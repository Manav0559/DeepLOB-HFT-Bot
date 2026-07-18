# CNN-LSTM on Binance order-book snapshots — experimental prototype

> **Status: learning-stage prototype, being rebuilt.** The next iteration is
> a faithful DeepLOB (Zhang et al., 2019) replication validated on the
> FI-2010 benchmark, with chronological embargoed splits and cost-aware
> replay evaluation. Until that lands, everything here should be read as a
> prototype with the limitations listed below.

A two-process experiment: a C++ client polls Binance's public REST depth
endpoint, normalizes the top 10 levels per side into 40 features, and ships
them over ZeroMQ (raw float32 payload) to a Python server running a small
CNN-LSTM classifier (`DeepLOB_Simplified` — a reduced stand-in, **not** the
full DeepLOB architecture) that paper-trades the resulting signal.

## Honest scope

- **Spot REST polling, not WebSockets.** Decision cadence is hundreds of
  milliseconds to seconds. This is not HFT, and no latency is measured
  anywhere in this repo.
- **Paper trading only.** There is no order execution. Fills are simulated at
  ticker price with zero fees, spread, or slippage, so any simulated PnL is
  optimistic by construction and no performance figures are published here.
- **Known training-pipeline flaw.** The committed preprocessing class-balances
  and shuffles rows *before* windowing and the train/test split, which
  scrambles the sequences and leaks across the split — reported validation
  accuracy is not meaningful. The rebuild replaces this with contiguous
  windows and chronological, embargoed splits.
- **The solid piece:** `data_ingest.py` — Binance `depth20@100ms` WebSocket
  collection to CSV with reconnect — carries over to the rebuild unchanged.

## Pipeline

1. `data_ingest.py` — collect 100ms depth snapshots to CSV
2. `preprocess_balanced.py` — features + fixed-horizon labels
3. `train.py` — train `DeepLOB_Simplified` (saves `model_balanced.pth`)
4. `strategy_server.py` (Python, ZMQ REP) + `bot.cpp` (C++ client) — live
   signal loop, paper only

## Build & run

```bash
# C++ client
cmake -B build && cmake --build build

# Python side
pip install -r requirements.txt
python strategy_server.py   # terminal 1
./build/hft_bot             # terminal 2
```
