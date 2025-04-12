# ContextOS ML Lab — Hyperopt Optimization Script

This repository contains an early experimental script used to optimize trading strategy hyperparameters using `Hyperopt`, `Backtrader`, and PyTorch.

## 🧠 Purpose
This was part of the foundational R&D phase in developing data-driven market behavior models. The script aimed to:

- Run backtests using historical price data
- Evaluate strategies based on Sharpe ratio, drawdown, and net return
- Use Bayesian optimization (via `Hyperopt`) to find optimal parameter sets

## 📦 Tech Stack
- `Hyperopt` (Bayesian optimization)
- `Backtrader` (backtesting engine)
- `Pandas`, `NumPy`
- `SQLite` (data storage)
- Early ML tools: `torch`, `sklearn`

## 🚧 Notes
- This code predates the modular refactor.
- Environment variables are hardcoded in this version (do not use in production).
- Use Python 3.9+.

## 🗂️ File List
- `hyperopt_optim_full.py`: Main experiment script (monolithic form)

## 📜 License
MIT

---

_This repo is no longer under active development but remains part of the ClarityX historical foundation for transparency and learning._
