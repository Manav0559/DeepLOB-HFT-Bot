# DeepLOB: High-Frequency Trading Bot (Hybrid C++ & Python)

A high-performance HFT system designed for **Binance Futures**. This project demonstrates a **Hybrid Architecture**: it leverages **C++** for ultra-low latency market data parsing and execution, while offloading complex decision-making to a **PyTorch (Python)** Deep Learning model.

## Architecture

* **The Engine (C++17):**
    * Connects to Binance WebSockets (Market Streams).
    * Parses the Limit Order Book (LOB) updates in microseconds.
    * Executes trades via HTTP (cURL) for minimal latency.
* **The Brain (Python 3.13):**
    * Hosts a **DeepLOB** Neural Network (CNN + LSTM).
    * Communicates with the C++ Engine via **ZeroMQ (ZMQ)** sockets.
    * Predicts short-term price movements (Alpha) based on Order Flow.

---

## Live Performance Results
*Test run performed on Binance Futures (BTC/USDT). Analysis generated via `analyze_trades.py`.*

| Metric | Value | Meaning |
| :--- | :--- | :--- |
| **Total Trades** | `76` | Sample size of the live session |
| **Win Rate** | `57.89%` | Percentage of profitable trades |
| **Profit Factor** | `1.55` | Gross Profit / Gross Loss (> 1.0 is profitable) |
| **Total PnL** | `+22.30 USDT` | Net Profit realized |
| **Max Drawdown** | `-13.80` | Worst peak-to-valley loss |
| **Avg Duration** | `7.4 min` | Average holding time per trade |

**Key Findings:**
* The **DeepLOB model** successfully identified market micro-structures, maintaining a win rate > 55%.
* **Latency:** The C++ engine processed order book updates fast enough to capture short-lived arbitrage opportunities.
* **Risk Management:** Stop-losses effectively capped downside, leading to a healthy Profit Factor of 1.55.

---

## Prerequisites

### Software Stack
* **C++:** Clang/GCC (C++17 standard), CMake (3.10+)
* **Python:** 3.8+ (PyTorch, NumPy, Pandas)
* **Libraries:** ZeroMQ, nlohmann/json, libcurl

---

## Installation

### 1. System Dependencies (MacOS)
Install the required C++ headers and build tools:
```bash
brew install zeromq nlohmann-json curl cmake

2. Python Environment
Install the AI and Data Science dependencies:

Bash

pip install -r requirements.txt
(Dependencies include: torch, numpy, pandas, requests, websockets)

⚡ How to Run
Phase 1: The Lab (Data & Training)
1. Collect Historical Data Scrape the live Limit Order Book to create a training dataset.

Bash

python data_ingest.py
Output: Generates training_data_balanced.npy.

2. Train the Model Train the Neural Network using the collected data.

Bash

python train.py
Output: Saves the trained weights to model_balanced.pth.

Phase 2: The Arena (Live Trading)
Step 1: Start the Inference Server This Python script loads the model and waits for data from the C++ engine.

Bash

python strategy_server.py
Output: Waiting for ZMQ request...

Step 2: Compile the C++ Engine Open a new terminal window.

Bash

mkdir build
cd build
cmake ..
make
Step 3: Launch the Bot Run the executable to start trading.

Bash

./hft_bot
Phase 3: The Audit (Analysis)
After the session, analyze your trade_log.csv to generate the performance table.

Bash

python analyze_trades.py
📂 Project Structure
Plaintext

HFT_Bot/
├── src/
│   ├── bot.cpp             # Main C++ Execution Engine (Binance connection)
│   └── ...                 # Helper classes (OrderManager, Logger)
├── model.py                # DeepLOB Architecture (CNN + LSTM)
├── train.py                # Training Loop & Data Normalization
├── strategy_server.py      # Python Bridge (ZMQ Server + Inference)
├── data_ingest.py          # WebSocket Scraper for Training Data
├── analyze_trades.py       # Performance Analytics Script
├── CMakeLists.txt          # Build Configuration
├── requirements.txt        # Python Dependencies
└── README.md               # Documentation