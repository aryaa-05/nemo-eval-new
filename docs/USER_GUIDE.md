# NeMo Evaluator Benchmarking Platform: User Guide

## 1. Project Overview
This repository contains the complete implementation of the NeMo Evaluator multi-model benchmarking system.

## 2. Configuration Files
- `configs/models_config.yaml`: Define models, providers, context windows, and pricing.
- `configs/benchmarks_config.yaml`: Define benchmarks, scorers, sample files, and domains.
- `configs/eval_config_default.yaml`: Set execution parameters (concurrency, limit, models, benchmarks).

## 3. Connecting Real NVIDIA NIM API Keys
To evaluate models against live NVIDIA NIM endpoints:
1. Obtain an API key from https://build.nvidia.com
2. Set your environment variable:
   ```powershell
   $env:NVIDIA_API_KEY = "nvapi-..."
   ```
3. In `configs/models_config.yaml`, change the `provider` field for the model from `mock` to `nvidia_nim`.

## 4. Running the Web Dashboard
```powershell
streamlit run src/dashboard/app.py
```
Tabs:
- **Leaderboard**: Displays rankings, Wilson confidence intervals, and Pareto frontier.
- **Radar & Domain Analytics**: 5-domain radar charts.
- **Model-vs-Model Diff & Flips**: Run comparison, McNemar test, and item-by-item flip inspector.
- **Launch Evaluation**: Trigger interactive runs directly from the UI.
- **Automated Reports**: Download markdown and HTML reports.
