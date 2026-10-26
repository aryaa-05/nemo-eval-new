# ??? NeMo Evaluator: Multi-Model Benchmarking & Comparison Platform

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![NVIDIA NeMo Evaluator](https://img.shields.io/badge/Framework-NVIDIA%20NeMo%20Evaluator-green.svg)](https://github.com/NVIDIA-NeMo/Evaluator)
[![Industry Partner](https://img.shields.io/badge/Collaboration-Tata%20Consultancy%20Services%20(TCS)-orange.svg)](https://www.tcs.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade evaluation and benchmarking platform based on **NVIDIA NeMo Evaluator**, developed as a collaborative capstone project between **Vishwakarma Institute of Technology (VIT), Pune** and **Tata Consultancy Services (TCS)**.

---

## ?? Project Team & Supervision
- **Students**: 
  - **Ashish Kumar** (CS-A, Roll: 110, G.R.No: 12311937, `kumar.ashish23@vit.edu`)
  - **Aryaa Sharma** (CS-A, Roll: 149, G.R.No: 12311983, `aryaa.sharma23@vit.edu`)
- **Internal Guide**: **Prof. (Dr.) Yogesh Sharma** (`yogesh.sharma@vit.edu`)
- **Institution**: Vishwakarma Institute of Technology, Pune ? Department of Computer Engineering
- **Academic Year**: 2026?27 | **Semester**: VII | **Group No.**: 16

---

## ?? Key Features

1. **8-Layer Modular Architecture**: Implements Presentation, Control Plane, Core Engine, Benchmark Layer, Model Gateway, Scoring & Judgement, Results Store, and Reporting Engine.
2. **Standardized Benchmarks**:
   - **MMLU-Pro**: 10-option multi-discipline reasoning across STEM, Humanities, and Social Sciences.
   - **GSM8K**: Multi-step grade-school mathematical word problems with step-by-step verification.
   - **MATH-500**: Advanced high-school and competition-grade mathematics with LaTeX extraction.
   - **GPQA**: Graduate-level Google-proof scientific reasoning.
   - **HumanEval**: Isolated Python execution sandbox testing generated code on unit tests.
   - **TruthfulQA**: Evaluation of factual hallucination and misconception resistance.
   - **Custom BYOB**: Bring-Your-Own-Benchmark loader with n-gram contamination detection.
3. **Unified Model Gateway**: Supports NVIDIA NIM, Groq, OpenRouter, OpenAI, local Ollama, and a built-in zero-cost deterministic simulator. `provider: "auto"` picks the first configured route whose API key is present (free tiers before paid) and records the requested ID, the served ID and whether the route is a substitute.
4. **Rigorous Statistics & Judgement**:
   - 95% Wilson score confidence intervals.
   - Paired **McNemar's significance tests** for measuring whether performance improvements are statistically real.
   - Quality Gate decisions (`nel gate`: PASS / FAIL).
5. **Interactive Streamlit Web Dashboard**: Real-time leaderboards, 5-domain radar charts, item-level flip inspector, automated report exporter, and background runs that keep the page responsive while polling progress every 2 seconds.
6. **Command Line Interface (`nel`)**: Fast CLI workflow (`nel list`, `nel eval run`, `nel compare`, `nel gate`, `nel report`).
7. **Cloud Ready (Google Colab)**: Pre-configured Jupyter Notebook (`notebooks/nemo_evaluator_colab.ipynb`) for GPU execution in compliance with green computing principles (SDG 9 & SDG 12).

---

## ?? Installation & Quickstart

```bash
# Clone the repository
git clone https://github.com/ashish-vit/nemo-evaluator-benchmarking.git
cd nemo-evaluator-benchmarking

# Install dependencies
pip install -r requirements.txt
```

Create a git-ignored `.env` in the repo root with only the keys you need (never commit it):

```
NVIDIA_API_KEY=<your key>
GROQ_API_KEY=<your key>
OPENROUTER_API_KEY=<needed by the six real pair models' auto routes; paid>
```

The CLI and dashboard read `.env` automatically. A model with `provider: "mock"` makes no API call and is
simulated; any network-backed run, including a stand-in or substitute route, is LIVE.

### Run CLI Commands (`nel`)
```bash
# 1. List configured models and benchmarks
python src/cli/nel.py list

# 2. Run a quick evaluation
python src/cli/nel.py eval run --config configs/eval_config_default.yaml --limit 10

# 3. Compare two runs with McNemar test & flip analysis
python src/cli/nel.py compare --run-a <RUN_A_ID> --run-b <RUN_B_ID> --output-md comparison.md

# 4. Check deployment Quality Gate
python src/cli/nel.py gate --baseline <RUN_A_ID> --candidate <RUN_B_ID>
```

### Launch Interactive Web Dashboard
```bash
streamlit run src/dashboard/app.py
```

---

## ??? Model Pairs: Required vs Stand-in

The brief names three model pairs to compare. Those exact models cannot be called on the free providers
available to this project (NVIDIA NIM and Groq) - some are retired, the rest are not hosted there. To exercise
the full pipeline on **real API calls**, each required pair is mirrored by a **stand-in pair**: cheaper
look-alike models that play the same role (small instruct vs Llama; mid-size reasoning vs Llama; reasoning vs
Qwen).

> **Important:** a stand-in run makes real API calls, so it is a **live** run, not a simulated one. But a
> stand-in's numbers are results for the **stand-in model only** - they are **not** results for the required
> models and must not be presented as such (see `CONTEXT.md` sections 0 and 11). Only a run with provider
> `mock` is simulated.

### Required models (as named in the brief)

| Pair | Model A (real) | Model B (real) | Status on free providers |
|---|---|---|---|
| 1 | `mistralai/Ministral-3-8B-Instruct-2512` | `meta-llama/Llama-3.1-8B-Instruct` | Not callable (Ministral absent; Llama-3.1-8B retired HTTP 410) |
| 2 | `mistralai/Ministral-3-14B-Reasoning-2512` | `meta-llama/Llama-4-Scout-17B-16E` | Not callable (both absent from NIM/Groq) |
| 3 | `mistralai/Ministral-3-8B-Reasoning-2512` | `Qwen3.5` (size not specified) | Not callable (both absent from NIM/Groq) |

These six models are configured with `provider: "auto"` and an OpenRouter route. Setting `OPENROUTER_API_KEY`
makes them live; the route is paid and **no real OpenRouter call has been made yet**, so whether each route
answers is unverified (see `CONTEXT.md` section 3a). Several served models are substitutes, not the exact
requested variants: the run records both IDs and `model_substituted: true`, and must be reported as a result for
the served model, never as the exact requested model.

### Stand-in models (used now, free, real API calls)

| Pair | Model A (stand-in) | Provider | Model B (stand-in) | Provider |
|---|---|---|---|---|
| 1 | `google/gemma-3-12b-it` | NVIDIA NIM | `meta/llama-3.2-11b-vision-instruct` | NVIDIA NIM |
| 2 | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA NIM | `meta/llama-3.2-11b-vision-instruct` | NVIDIA NIM |
| 3 | `openai/gpt-oss-20b` | Groq | `qwen/qwen3.8-27b` | Groq |

Configs: `configs/standin_pair1.yaml`, `standin_pair2.yaml`, `standin_pair3.yaml`. Pair 1's model A
(`gemma-3-12b-it`) is currently **not callable** (HTTP 404), so pair 1 cannot run until it is replaced.

---

## ??? System Architecture
```
+-----------------------------------------------------------------------------+
| Presentation Layer: Streamlit Dashboard / CLI (nel)                         |
+-----------------------------------------------------------------------------+
| Orchestration Plane: YAML Config Manager / Job Scheduler / Run Registry     |
+-----------------------------------------------------------------------------+
| Core Evaluation Engine: Interceptor Proxy / Solvers (Chat, CoT) / Sandbox   |
+-----------------------------------------------------------------------------+
| Data & Model Gateway: Standard Benchmarks <---> Unified NIM / API Gateway   |
+-----------------------------------------------------------------------------+
| Scoring & Judgement: Rule Scorers / McNemar & Wilson Statistics            |
+-----------------------------------------------------------------------------+
| Storage & Comparison: SQLite Run Store / nel compare / Automated Reports   |
+-----------------------------------------------------------------------------+
```
