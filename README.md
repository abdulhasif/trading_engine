# trading-engine

Real-time execution heartbeat for the Advanced Trading Model. Connects to Upstox for live tick data and automated order management.

## 🛠 Features
- **Live Renko Heartbeat**: Incremental brick formation from sub-second ticks.
- **Upstox Integration**: WebSocket connectivity for indices and equity symbols.
- **Paper Trading Simulator**: Bit-perfect broker simulation for strategy validation.
- **State Sync**: Real-time state persistence for dashboard monitoring.

## 🚀 Installation & Setup

This project requires **Python 3.12** specifically.

1. **Install Core Dependency**:
   Ensure `trading-core` is installed in your environment:
   ```bash
   py -3.12 -m pip install -e ../trading_core
   ```

2. **Install Project Requirements**:
   ```bash
   py -3.12 -m pip install -r requirements.txt
   ```

## 🔄 Execution

To start the live trading session:
```bash
py -3.12 -m trading_engine.src.engine_main
```

### Configuration
Set your API keys and trading parameters in `trading_engine/config.py`.
The engine reads from `data/brain/` for live ML inference.
