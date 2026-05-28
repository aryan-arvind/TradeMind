# TradeMind: Multimodal Deep Reinforcement Learning for Financial Trading

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-ee4c2c)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688)
![React](https://img.shields.io/badge/React-Vite-61dafb)

**TradeMind** is an advanced trading agent framework powered by Deep Reinforcement Learning (Proximal Policy Optimization). It uses a multimodal architecture to combine technical price indicators, market sentiment, and trading volume into a unified decision-making policy for portfolio allocation.

![Dashboard Screenshot](dashboard_test.png)

## 🚀 Key Features

- **Multimodal Architecture**: Processes data through a 3-branch model (Price LSTM, Sentiment FFN, Volume FFN).
- **PPO Actor-Critic**: Robust training pipeline utilizing Generalized Advantage Estimation (GAE) and clipping.
- **NSE Market Data**: Built and tested using real National Stock Exchange (NSE) data for major Indian tickers (RELIANCE, TCS, HDFCBANK, INFY) from 2021 to 2024/2025.
- **Ablation Studies**: Built-in support to run ablation modes (Price only, Sentiment only, Volume only, or All).
- **Interactive Dashboard**: A React/Vite frontend displaying real-time backtest metrics (Cumulative Return, Sharpe Ratio, Max Drawdown) and live portfolio weights powered by a FastAPI backend.

## 🛠️ Technology Stack

- **Core/AI:** PyTorch, Proximal Policy Optimization (PPO)
- **Backend:** FastAPI, Python
- **Frontend:** React, Vite, Node.js

---

## 💻 Quick Start

### 1. Install Dependencies
It's recommended to use Python 3.10 or 3.11. First, set up your virtual environment and install the required Python packages:

```bash
pip install -r requirements.txt
```
*(If the torch installation fails, install the CPU wheels directly: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`)*

### 2. Train the Model
You can start training the PPO agent from scratch. The pipeline will automatically generate model artifacts and ablation results:

```bash
python run.py --mode train --epochs 4
```
*Artifacts generated in the `artifacts/` folder: `ppo_multimodal.pt`, `ablation_results.json`, `latest_metrics.json`*

### 3. Start the Backend API
Serve the trained policy and backtest metrics over a FastAPI endpoint:

```bash
python run.py --mode serve --port 8000
```
*The API will be available at `http://localhost:8000/api/dashboard`*

### 4. Run the Frontend Dashboard
Open a new terminal, navigate to the `frontend` directory, and start the Vite dev server:

```bash
cd frontend
npm install
npm run dev
```

The frontend dashboard will automatically connect to the backend and display live portfolio weights and trading metrics.

## 📊 Evaluation & Metrics
The pipeline includes backtesting functionality that computes key financial metrics:
- **Cumulative Return**
- **Sharpe Ratio**
- **Max Drawdown**

These metrics are visually represented in the dashboard UI and updated in the artifacts folder.
