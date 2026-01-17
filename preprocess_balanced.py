import pandas as pd
import numpy as np
import json
from tqdm import tqdm

INPUT_FILE = "market_data_week1.csv"
OUTPUT_FILE = "training_data_balanced.npy"
LOOK_AHEAD = 100  # 10 seconds
THRESHOLD = 0.00015

def parse_json(text):
    try: return json.loads(text)
    except: return []

df = pd.read_csv(INPUT_FILE)

# 1. FEATURE EXTRACTION
print("Processing features.")
raw_data = df.to_numpy()
bid_idx = df.columns.get_loc("bids")
ask_idx = df.columns.get_loc("asks")

features = []
mid_prices = []

for i in tqdm(range(len(raw_data))):
    try:
        row = raw_data[i]
        bids = np.array(parse_json(row[bid_idx]), dtype=np.float32)
        asks = np.array(parse_json(row[ask_idx]), dtype=np.float32)
        
        if len(bids) < 10 or len(asks) < 10: continue
            
        bids, asks = bids[:10], asks[:10]
        mid_price = (bids[0][0] + asks[0][0]) / 2.0
        mid_prices.append(mid_price)
        
        # Normalize
        norm_bids_p = bids[:, 0] / mid_price
        norm_asks_p = asks[:, 0] / mid_price
        norm_bids_v = np.log1p(bids[:, 1])
        norm_asks_v = np.log1p(asks[:, 1])
        
        features.append(np.concatenate([norm_bids_p, norm_asks_p, norm_bids_v, norm_asks_v]))
    except:
        continue

# 2. LABEL GENERATION
print("Generating Labels.")
X = []
Y = []

count = len(features)
for i in range(count - LOOK_AHEAD):
    curr = mid_prices[i]
    fut = mid_prices[i + LOOK_AHEAD]
    change = (fut - curr) / curr
    
    label = 1 # HOLD
    if change > THRESHOLD: label = 2 # BUY
    elif change < -THRESHOLD: label = 0 # SELL
    
    X.append(features[i])
    Y.append(label)

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.int64)

# 3. THE BALANCER
print("Balancing Data.")
idx_0 = np.where(Y == 0)[0] # SELL
idx_1 = np.where(Y == 1)[0] # HOLD
idx_2 = np.where(Y == 2)[0] # BUY

print(f"Original Counts -> SELL: {len(idx_0)}, HOLD: {len(idx_1)}, BUY: {len(idx_2)}")

# Find the smallest class size
min_len = min(len(idx_0), len(idx_2)) 
# We allow HOLD to be slightly larger (1.5x) to represent "noise" but not dominate
target_hold_len = int(min_len * 1.5) 

# Randomly downsample
np.random.shuffle(idx_0)
np.random.shuffle(idx_1)
np.random.shuffle(idx_2)

idx_0 = idx_0[:min_len]
idx_1 = idx_1[:target_hold_len] # Cut the giant HOLD class down
idx_2 = idx_2[:min_len]

# Combine and Shuffle again
final_indices = np.concatenate([idx_0, idx_1, idx_2])
np.random.shuffle(final_indices)

X_final = X[final_indices]
Y_final = Y[final_indices]

print(f"Balanced Counts -> Total: {len(Y_final)}")
print(f"Saving to {OUTPUT_FILE}.")
np.save(OUTPUT_FILE, {"X": X_final, "Y": Y_final})
print("Done.")