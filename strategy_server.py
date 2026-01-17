import time
import zmq
import torch
import numpy as np
import ccxt
import csv
import os
from collections import deque
from model import DeepLOB_Simplified

MODEL_FILE = "model_balanced.pth"
SYMBOL = 'BTC/USDT'
CONFIDENCE_THRESHOLD = 0.60
COOLDOWN_SECONDS = 30
STOP_LOSS_PCT = 0.002
TAKE_PROFIT_PCT = 0.004
VOLATILITY_THRESHOLD = 15.0
LOG_FILE = "trade_log.csv"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Setup Logging
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Event", "Price", "Confidence", "PnL", "Balance"])

def log_event(event, price, conf, pnl, balance):
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), event, price, f"{conf:.4f}", f"{pnl:.2f}", f"{balance:.2f}"])

# 2. Initialize Binance
exchange = ccxt.binance()

# 3. Load Model
model = DeepLOB_Simplified().to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
    model.eval()
    print("Model Loaded.")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# 4. Setup ZeroMQ
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

# 5. State Variables
history = deque(maxlen=100)
price_history = deque(maxlen=50)
position = None 
entry_price = 0.0
entry_time = 0
balance = 10000.00 # Updated Balance
highest_pnl = 0.0 

print(f"Waiting for C++ Spotter data. (Logging to {LOG_FILE})")

while True:
    try:
        # A. Receive Data
        message = socket.recv()
        data = np.frombuffer(message, dtype=np.float32)
        
        if len(data) != 40:
            socket.send_string("ERROR")
            continue
            
        history.append(data)
        if len(history) < 100:
            socket.send_string("BUFFERING")
            continue
            
        # B. AI Inference
        input_tensor = torch.FloatTensor(np.array(history)).to(DEVICE)
        input_tensor = input_tensor.view(1, 100, 40)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, action_tensor = torch.max(probs, 1)
            action = action_tensor.item()
            conf_score = conf.item()
            
        # C. Fetch Price & Calc Volatility
        try:
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker['last']
            price_history.append(current_price)
        except:
            socket.send_string(f"{action},{conf_score:.4f}")
            continue

        if len(price_history) == 50:
            volatility = np.std(price_history)
        else:
            volatility = 999.0

        # D. Trading Logic
        timestamp = time.time()
        reply = f"{action},{conf_score:.4f}"
        
        # EXIT LOGIC
        if position is not None:
            duration = timestamp - entry_time
            pnl = 0
            if position == 'LONG': pnl = (current_price - entry_price) / entry_price
            elif position == 'SHORT': pnl = (entry_price - current_price) / entry_price
            
            if pnl > highest_pnl: highest_pnl = pnl

            close_reason = None
            
            # 1. Trailing Stop (Lock in profits)
            if highest_pnl > 0.0015 and pnl < 0.0005:
                close_reason = "Trailing Stop"
            
            # 2. Smart Boredom Exit (NEW)
            # If market is dead (Vol < 5) AND we are losing, get out.
            elif volatility < 5.0 and pnl < 0:
                close_reason = "Low Volatility Exit"

            # 3. Standard Exits
            elif duration >= COOLDOWN_SECONDS:
                if pnl > TAKE_PROFIT_PCT: close_reason = "Take Profit"
                elif pnl < -STOP_LOSS_PCT: close_reason = "Stop Loss"
                elif position == 'LONG' and action == 0 and conf_score > CONFIDENCE_THRESHOLD: close_reason = "Ai Flip Sell"
                elif position == 'SHORT' and action == 2 and conf_score > CONFIDENCE_THRESHOLD: close_reason = "Ai Flip Buy"
            
            # 4. Panic Stop
            elif pnl < -STOP_LOSS_PCT: 
                close_reason = "Panic Stop Loss"

            if close_reason:
                realized = pnl * 1000
                balance += realized
                print(f"[{time.strftime('%H:%M:%S')}] {close_reason} @ {current_price} (Vol: {volatility:.1f}) | Bal: ${balance:.2f}")
                log_event(close_reason, current_price, conf_score, realized, balance)
                position = None
                highest_pnl = 0.0
                reply = "CLOSE"

        # ENTRY LOGIC
        elif position is None and conf_score > CONFIDENCE_THRESHOLD and action != 1:
            if volatility > VOLATILITY_THRESHOLD:
                if action == 2:
                    position = 'LONG'
                    entry_price = current_price
                    entry_time = timestamp
                    highest_pnl = 0.0
                    print(f"[{time.strftime('%H:%M:%S')}] OPEN LONG @ {current_price} ({conf_score*100:.1f}%) [Vol: {volatility:.1f}]")
                    log_event("OPEN_LONG", current_price, conf_score, 0, balance)
                    reply = "BUY"
                elif action == 0:
                    position = 'SHORT'
                    entry_price = current_price
                    entry_time = timestamp
                    highest_pnl = 0.0
                    print(f"[{time.strftime('%H:%M:%S')}] OPEN SHORT @ {current_price} ({conf_score*100:.1f}%) [Vol: {volatility:.1f}]")
                    log_event("OPEN_SHORT", current_price, conf_score, 0, balance)
                    reply = "SELL"
                    
        socket.send_string(reply)
        
    except Exception as e:
        print(f"Error: {e}")
        try: socket.send_string("ERROR") 
        except: pass