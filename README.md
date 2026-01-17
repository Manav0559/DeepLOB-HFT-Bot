![Language](https://img.shields.io/badge/Language-C%2B%2B17-00599C?style=for-the-badge&logo=c%2B%2B)
![Language](https://img.shields.io/badge/Language-Python_3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/AI-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Binance-F3BA2F?style=for-the-badge&logo=binance&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

# DeepLOB: High-Frequency Trading Bot (Hybrid C++ & Python)

A high-performance HFT system designed for **Binance Futures**. This project demonstrates a **Hybrid Architecture**: it leverages **C++** for ultra-low latency market data parsing and execution, while offloading complex decision-making to a **PyTorch (Python)** Deep Learning model.

---

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

## Installation & Usage

### 1. System Dependencies (MacOS)
Install the required C++ headers and build tools:
```bash
brew install zeromq nlohmann-json curl cmake
