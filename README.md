# ContextOS ML Lab — Hyperopt Optimization Script

This repository contains early experimental scripts used to optimize trading strategy hyperparameters using `Hyperopt`, `Backtrader`, and PyTorch.

## 🧠 Purpose
This was part of the foundational R&D phase in developing data-driven market behavior models. This repo is intended as an experimental playground for iterating on ML ideas before they are integrated into production pipelines.

The primary goal of this repo is to:
- Test various ML model types against known backtest labels.
- Evaluate which contextual features hold real predictive power.
- Track improvements through consistent metrics and visualizations.
- Lay the groundwork for a fully-automated signature detection classifier.

The scripts aimed to:

- Run backtests using historical price data
- Evaluate strategies based on Sharpe ratio, drawdown, and net return
- Use Bayesian optimization (via `Hyperopt`) to find optimal parameter sets

## 📦 Tech Stack
- `Hyperopt` (Bayesian optimization)
- `Backtrader` (backtesting engine)
- `Pandas`, `NumPy`
- `SQLite` (data storage)
- Early ML tools: `torch`, `sklearn`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `numpy`

## 🚧 Notes
- This code predates the modular refactor.
- Environment variables are hardcoded in this version (do not use in production).
- Use Python 3.9+.

All experiments use CSV files exported from trading backtests (not included here).
Each script can be run independently with its own input format.
Feel free to fork, modify, and use as a base for new classifiers (LSTM, XGBoost, etc.).

## 🗂️ File List
- `hyperopt_optim_full.py`: Main experiment script (monolithic form)

## 📜 License
MIT

---

_This repo is no longer under active development but remains part of the ClarityX historical foundation for transparency and learning._
