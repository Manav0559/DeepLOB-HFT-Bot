![Language](https://img.shields.io/badge/Language-C%2B%2B17-00599C?style=for-the-badge&logo=c%2B%2B)
![Language](https://img.shields.io/badge/Language-Python_3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Framework](https://img.shields.io/badge/AI-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Binance-F3BA2F?style=for-the-badge&logo=binance&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

# ⚡ DeepLOB-HFT: Hybrid Algo Trading System

> **A sub-millisecond execution engine powered by Deep Learning.**

DeepLOB-HFT is a live trading bot targeting **Binance Futures**. It bridges two worlds: the raw speed of **C++** for market data processing and execution, and the predictive power of **PyTorch (Python)** for detecting micro-structure alpha in the Limit Order Book (LOB).

---

## 🏗️ System Architecture

The system utilizes a **Hybrid Decoupled Architecture**. The critical path (Market Data $\rightarrow$ Execution) is handled by C++ to minimize jitter, while the compute-heavy inference is offloaded to a dedicated Python process via ZeroMQ.

```mermaid
graph TD
    %% Define Styles
    classDef cpp fill:#00599C,stroke:#fff,stroke-width:2px,color:#fff;
    classDef py fill:#3776AB,stroke:#fff,stroke-width:2px,color:#fff;
    classDef ex fill:#F3BA2F,stroke:#333,stroke-width:2px,color:#000;
    classDef store fill:#ddd,stroke:#333,stroke-width:1px,stroke-dasharray: 5 5;

    subgraph "Exchange Layer"
        Binance[("🌊 Binance Futures API")]:::ex
    end

    subgraph "C++ Low-Latency Core"
        WS[WebSocket Handler]:::cpp
        LOB[LOB Reconstruction]:::cpp
        Risk[🛡️ Risk Engine]:::cpp
        Exec[Order Manager]:::cpp
        ZMQ_Pub[ZMQ Publisher]:::cpp
    end

    subgraph "Python Inference Layer"
        ZMQ_Sub[ZMQ Subscriber]:::py
        Pre[Tensor Preprocessing]:::py
        Model["🧠 DeepLOB (CNN-LSTM)"]:::py
    end

    %% Data Flow
    Binance == "WSS Feed (Diff Depth)" ==> WS
    WS --> LOB
    LOB -->|Snapshot 10x4x100| ZMQ_Pub
    ZMQ_Pub == "IPC / TCP" ==> ZMQ_Sub
    
    ZMQ_Sub --> Pre
    Pre --> Model
    Model -->|Alpha Signal Long/Short| Exec
    
    Exec --> Risk
    Risk -->|Valid Order| Binance
    
    %% Storage
    LOB -.->|Log| CSV[("Data Logger")]:::store
