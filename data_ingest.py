import asyncio
import websockets
import json
import csv
import os
from datetime import datetime

SYMBOL = "btcusdt"
FILENAME = "market_data_week1.csv"
BUFFER_SIZE = 100  

async def run_scraper():
    url = f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth20@100ms"
    file_exists = os.path.isfile(FILENAME)
    
    print(f"Initializing Scraper for {SYMBOL}")
    
    with open(FILENAME, mode='a', newline='') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["timestamp", "bids", "asks"])
            
        buffer = []
        count = 0
        
        while True:
            try:
                print(f"Connecting to Binance WebSocket")
                async with websockets.connect(url) as ws:
                    print("Connected!")
                    
                    while True:
                        msg = await ws.recv()
                        data = json.loads(msg)
                        
                        # 1. Check for "bids"
                        if 'bids' in data:
                            bids = data['bids']
                            asks = data['asks']
                        # 2. Check for "b"
                        elif 'b' in data:
                            bids = data['b']
                            asks = data['a']
                        # 3. Subscription Confirmations
                        else:
                            continue 
                        
                        # SAVE DATA 
                        now = datetime.now().timestamp()
                        buffer.append([now, json.dumps(bids), json.dumps(asks)])
                        
                        if len(buffer) >= BUFFER_SIZE:
                            writer.writerows(buffer)
                            f.flush()
                            count += len(buffer)
                            print(f"\rCollected {count} rows.)", end="", flush=True)
                            buffer = []

            except Exception as e:
                print(f"\n Error: {e}")
                print("Reconnecting in 5 seconds.")
                await asyncio.sleep(5)
                continue

if __name__ == "__main__":
    try:
        asyncio.run(run_scraper())
    except KeyboardInterrupt:
        print("\n Scraper Stopped.")
    except Exception as e:
        print(f"\n ERROR: {e}")