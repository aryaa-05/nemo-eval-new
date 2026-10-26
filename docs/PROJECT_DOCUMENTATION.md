# NEMO Evaluator Benchmarking Platform — Complete Project Documentation

**Project Title:** NEMO Evaluator Use Case: Benchmarking and Comparing the Performance of Various Models
**Institution:** Vishwakarma Institute of Technology (VIT), Pune — Department of Computer Engineering
**Industry Partner:** Tata Consultancy Services (TCS)
**Academic Year:** 2026–27 | **Semester:** VII | **Group No.:** 16
**Documentation Date:** 26 September 2026
**Document Type:** Consolidated technical + progress reference (single source of truth)

---

## Table of Contents

1. [Document Purpose & How to Read This File](#1-document-purpose--how-to-read-this-file)
2. [Project Identity, Team & Governance](#2-project-identity-team--governance)
3. [Problem Statement & Objectives](#3-problem-statement--objectives)
4. [Technology Stack & Verified Environment](#4-technology-stack--verified-environment)
5. [Repository Structure (Annotated)](#5-repository-structure-annotated)
6. [System Architecture — 8 Layers Mapped to Real Modules](#6-system-architecture--8-layers-mapped-to-real-modules)
7. [Layer-by-Layer Deep Dive](#7-layer-by-layer-deep-dive)
8. [Data Model & On-Disk Formats](#8-data-model--on-disk-formats)
9. [End-to-End Execution Flow (Numbered Trace)](#9-end-to-end-execution-flow-numbered-trace)
10. [Statistical Methodology](#10-statistical-methodology)
11. [Configuration Reference](#11-configuration-reference)
12. [Dataset Inventory](#12-dataset-inventory)
13. [Experimental Results & Progress Evidence](#13-experimental-results--progress-evidence)
14. [Case Study: Llama-3.1-8B vs 70B Comparison](#14-case-study-llama-31-8b-vs-70b-comparison)
15. [Test Suite & Verification Status](#15-test-suite--verification-status)
16. [Documentation & Report Inventory](#16-documentation--report-inventory)
17. [Feature Completeness Matrix (Claimed vs. Implemented)](#17-feature-completeness-matrix-claimed-vs-implemented)
18. [Known Gaps, Defects & Risks — Action Register](#18-known-gaps-defects--risks--action-register)
19. [Roadmap & Recommended Next Steps](#19-roadmap--recommended-next-steps)
20. [Appendices](#20-appendices)

---

## 1. Document Purpose & How to Read This File

This document is a **complete, consolidated explanation** of the NEMO Evaluator Benchmarking Platform, written to serve four audiences from one file:

| Audience | Read these sections |
|---|---|
| **Project reviewers / guide (Prof. Yogesh Sharma)** | §2, §3, §13, §17, §18, §19 |
| **Evaluators (TCS / industry panel)** | §6, §7, §10, §13, §14, §17 |
| **New team members / future maintainers** | §4, §5, §7, §8, §9, §11, §20 |
| **Anyone re-running or extending the system** | §9, §10, §11, §18, §19, §20.2 |

**Verification stance.** Every factual claim here was verified by reading source code and by executing the system locally. Claims about *what the code does* carry `file:line` references. Claims about *what the system produced* are backed by the persisted artifacts in `runs/`. Where documentation and code disagree, **both are reported and the disagreement is flagged** rather than silently reconciled (§18).

**Relationship to existing documents.** This file supersedes the others for technical detail: `README.md` (overview/quickstart), `docs/USER_GUIDE.md` (usage), `docs/REVIEW_1_PROGRESS_REPORT.md` and `docs/REVIEW_2_FINAL_REPORT.md` (academic review forms). Those remain the canonical submission artefacts for their respective forms; this document is the technical master.

---

## 2. Project Identity, Team & Governance

| Attribute | Detail |
|---|---|
| **Project Title** | NEMO Evaluator Use Case: Benchmarking and Comparing the Performance of Various Models |
| **Academic Year / Semester** | 2026–27 / VII |
| **Group Number** | 16 |
| **Department** | Computer Engineering |
| **Institute** | Vishwakarma Institute of Technology, Pune |
| **Industry Partner** | Tata Consultancy Services (TCS) |
| **Internal Guide** | Prof. (Dr.) Yogesh Sharma (`yogesh.sharma@vit.edu`) |
| **Student 1** | Ashish Kumar — CS-A, Roll 110, G.R.No. 12311937, `kumar.ashish23@vit.edu` |
| **Student 2** | Aryaa Sharma — CS-A, Roll 149, G.R.No. 12311983, `aryaa.sharma23@vit.edu` |
| **Review Forms** | Form FF No. 180 — Review 1 (Mid-Semester), Review 2 (End-Semester) |
| **Sustainability Framing** | UN SDG 9 (Industry, Innovation & Infrastructure) and SDG 12 (Responsible Consumption & Production) |
| **Package Version** | 1.0.0 (`pyproject.toml:7`) |
| **Python Requirement** | `>=3.10` (`pyproject.toml:10`) |

**Governance in practice:** the two students own the codebase jointly; the internal guide reviews at two formal checkpoints (Review 1, Review 2); TCS provides the industry use-case framing (enterprise LLM selection). The platform's Quality Gate (`nel gate`) mirrors a real release-readiness process: a candidate model must clear an absolute accuracy floor *and* not regress beyond a tolerance before deployment approval.

---

## 3. Problem Statement & Objectives

### 3.1 The problem

Organisations deploying LLMs face a recurring, expensive decision: **which model should we ship?** Public leaderboards answer this poorly for five reasons:

1. **Non-reproducible numbers.** Leaderboards are recomputed under different harnesses, prompts, few-shot settings and scorers, so a small gap may be an artefact of the harness rather than a capability difference.
2. **Single-metric fixation.** Aggregate accuracy hides *where* a model fails. A model can be strong on math and useless on code; one average conceals that.
3. **Non-significant claims.** Vendors routinely claim "Model B beats Model A" from raw accuracy deltas with no test of whether the difference exceeds sampling noise. On 10-item samples a 10-point delta is routinely noise.
4. **Black-box disagreements.** When two models differ, teams get a number but no *item-level* evidence — which specific questions flipped, and why.
5. **Cost blindness.** Accuracy is reported without latency and token cost, so expensive models with negligible accuracy gains look like winners.

### 3.2 What this project builds

A reproducible, statistically-grounded, multi-model benchmarking platform producing:

- **Standardised evaluation** of 6 LLMs across 6 standardised benchmarks spanning 5 capability domains.
- **Uncertainty quantification** — 95% Wilson score confidence intervals on every accuracy figure.
- **Paired significance testing** — McNemar's test to separate real capability differences from noise.
- **Item-level flip analysis** — the exact questions where two models disagree, with full prompts and both responses.
- **Cost/latency-aware ranking** — a Pareto accuracy-vs-latency frontier plus per-run token cost.
- **Release gating** — an automated PASS/FAIL deployment decision.
- **Zero-cost demonstrability** — a built-in simulator so the whole pipeline runs with no API spend.

### 3.3 Objectives mapped to evidence

| # | Objective | Where satisfied | Section |
|---|---|---|---|
| O1 | Unified access to heterogeneous providers behind one interface | `src/gateway/model_client.py` | §7.6 |
| O2 | Standardised ingestion of 6 public benchmarks | `src/benchmarks/` | §7.5, §12 |
| O3 | Per-benchmark scoring matched to answer type (MCQ / math / code execution) | `src/scoring/rule_scorers.py`, `src/engine/sandbox.py` | §7.4, §7.7 |
| O4 | Confidence intervals and paired significance testing | `src/scoring/statistics.py` | §10 |
| O5 | Item-level comparative analysis (flip inspector) | `src/comparison/diff_engine.py` | §7.9, §14 |
| O6 | Persisted, queryable run registry | `src/storage/` | §7.8 |
| O7 | Automated report generation (Markdown/HTML) | `src/reporting/report_generator.py` | §7.10 |
| O8 | Interactive visualisation (leaderboard, radar, Pareto) | `src/dashboard/app.py` | §7.2, §13 |
| O9 | Automated release quality gate | `src/cli/nel.py:196`, `src/scoring/statistics.py:47` | §7.1, §10.3 |
| O10 | Zero-cost execution without dedicated GPU infrastructure | `notebooks/nemo_evaluator_colab.ipynb` | §16 |

---

## 4. Technology Stack & Verified Environment

### 4.1 Declared dependencies

`requirements.txt` (13 packages) and `pyproject.toml:11-23` (11 packages) overlap but are not identical (§18-D12).

| Dependency | Purpose in this project |
|---|---|
| `pyyaml>=6.0.0` | Parses the three configuration files in `configs/` |
| `pydantic>=2.0.0` | Validates and serialises all run artifacts (`src/storage/schema.py`) |
| `rich>=13.0.0` | CLI tables, panels, coloured output (`src/cli/nel.py`) |
| `click>=8.1.0` | CLI command/option framework for `nel` |
| `tabulate>=0.9.0` | Declared but **never imported** (§18-D13) |
| `jinja2>=3.1.0` | Report templating (Markdown/HTML) |
| `requests>=2.31.0` | HTTP calls to NIM / OpenAI-compatible / Ollama endpoints |
| `httpx>=0.25.0` | Declared but **never imported** (§18-D13) |
| `pandas>=2.0.0` | Dashboard dataframes |
| `numpy>=1.24.0` | Transitive (scipy/pandas); not directly imported |
| `scipy>=1.11.0` | **Statistically load-bearing** — normal and chi-square distributions |
| `plotly>=5.18.0` | Leaderboard bars, Pareto scatter, radar charts |
| `streamlit>=1.30.0` | Web dashboard |

The only third-party dependency genuinely **critical to correctness** (as opposed to presentation) is `scipy`, used in exactly two places: `stats.norm.ppf` for the Wilson z-value (`src/scoring/statistics.py:10`) and `stats.chi2.cdf` for the McNemar p-value (`src/scoring/statistics.py:36`).

### 4.2 Verified local environment

Confirmed by execution on 26 September 2026:

| Component | Version / Status |
|---|---|
| OS | Windows (win32), PowerShell 5.1 |
| Python | **3.13.9** (exceeds the `>=3.10` requirement) |
| pydantic | 2.12.4 |
| PyYAML | 6.0.3 |
| click | 8.4.2 |
| Jinja2 | 3.1.6 |
| scipy | 1.16.3 |
| plotly | 6.3.0 |
| streamlit | 1.51.0 |
| pandas | 2.3.3 |
| requests | 2.32.5 |
| Git repository | **Not initialised** (no `.git` directory) |
| Test suite | **7/7 passing** (§15) |

> **Note on the `nel` console script.** `pyproject.toml:26` declares `nel = "src.cli.nel:cli"`, but the package is not installed in this environment, so `nel ...` is not on `PATH`. All commands therefore run as `python src/cli/nel.py ...` — which is also what `README.md:58-67` and the Colab notebook use. `pip install -e .` would enable the bare `nel` command.

---

## 5. Repository Structure (Annotated)

```
nemo-evaluator-benchmarking/
│
├── README.md                                  Overview, quickstart, architecture diagram
├── comparison.md                              Latest generated comparison report (8B vs 70B, batch 1)
├── pyproject.toml                             Package metadata + `nel` console-script entry point
├── requirements.txt                           Full dependency list (superset of pyproject)
│
├── configs/                                   ── Layer 7: Orchestration & Control Plane ──
│   ├── models_config.yaml                     6 model definitions (provider, params, pricing, strengths)
│   ├── benchmarks_config.yaml                 7 benchmark definitions (domain, scorer, sample file)
│   └── eval_config_default.yaml               Job definition: models, benchmarks, limits, gate thresholds
│
├── datasets/                                  ── Layer 4a: Data Plane ──
│   ├── mmlu_pro/sample.jsonl                  15 items — 10-option MCQ (A–J)
│   ├── gsm8k/sample.jsonl                     15 items — grade-school math word problems
│   ├── math_500/sample.jsonl                  15 items — competition math (LaTeX answers)
│   ├── gpqa/sample.jsonl                      15 items — graduate-level science MCQ (A–D)
│   ├── humaneval/sample.jsonl                 15 items — Python synthesis + unit tests
│   ├── truthfulqa/sample.jsonl                15 items — misconception resistance MCQ (A–D)
│   └── custom_byob/sample.jsonl                2 items — bring-your-own-benchmark demo
│
├── src/
│   ├── __init__.py                            "# NeMo Evaluator Package"
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── nel.py                             ── Layer 8: CLI ──  Click group: list / eval run / compare / gate
│   │
│   ├── engine/                                ── Layer 5: Core Evaluation Engine ──
│   │   ├── __init__.py
│   │   ├── runner.py                          EvaluationRunner — main orchestration loop
│   │   ├── solvers.py                         Prompt formatters: simple / cot / harbor
│   │   └── sandbox.py                         CodeSandbox — isolated Python execution for HumanEval
│   │
│   ├── gateway/                               ── Layer 4b: Model Gateway ──
│   │   ├── __init__.py
│   │   ├── model_client.py                    UnifiedModelClient — provider router
│   │   ├── interceptor.py                     InterceptorProxy — SHA-256 response cache
│   │   └── mock_models.py                     MockModelEngine — zero-cost simulator
│   │
│   ├── benchmarks/                            ── Layer 4a: Data Plane ──
│   │   ├── __init__.py
│   │   ├── base.py                            7 Benchmark classes + factory get_benchmark()
│   │   └── loader.py                          BenchmarkLoader — JSONL reader + n-gram contamination check
│   │
│   ├── scoring/                               ── Layer 3: Scoring & Judgement ──
│   │   ├── __init__.py
│   │   ├── rule_scorers.py                    MCQ + math answer extraction and scoring
│   │   ├── statistics.py                      Wilson CI, McNemar test, Quality Gate
│   │   └── llm_judge.py                       LLM-as-judge pairwise harness (not wired — §18-D4)
│   │
│   ├── storage/                               ── Layer 2: Results Store ──
│   │   ├── __init__.py
│   │   ├── schema.py                          Pydantic: RunArtifact, ItemResult, BenchmarkSummary
│   │   └── run_store.py                       RunStore — JSON artifact + SQLite registry
│   │
│   ├── comparison/                            ── Layer 6: Comparison Engine ──
│   │   ├── __init__.py
│   │   └── diff_engine.py                     DiffEngine — paired diff, flips, McNemar wiring
│   │
│   ├── reporting/                             ── Layer 1: Reporting Engine ──
│   │   ├── __init__.py
│   │   └── report_generator.py                Jinja2 Markdown + HTML report generator
│   │
│   └── dashboard/                             ── Layer 8: Dashboard ──
│       ├── __init__.py
│       └── app.py                             5-tab Streamlit application
│
├── tests/                                     3 unittest modules, 7 tests, all passing
│   ├── test_statistics.py                     Wilson CI, McNemar, Quality Gate
│   ├── test_scorers.py                        MCQ scorer, math scorer, code sandbox
│   └── test_diff.py                           DiffEngine integration (against persisted runs)
│
├── runs/                                      ── Layer 2: Persisted results ──
│   ├── runs_registry.db                       SQLite: 12 run summary rows
│   ├── run_20260916_19*.json  (6 files)       Batch 1 artifacts — 60 items each
│   ├── run_20260916_20*.json  (6 files)       Batch 2 artifacts — 60 items each
│   └── llama_8b_vs_70b_comparison.md          Earlier generated comparison report
│
├── docs/
│   ├── PROJECT_DOCUMENTATION.md               ← THIS FILE
│   ├── REVIEW_1_PROGRESS_REPORT.md            Form FF 180 — mid-semester submission
│   ├── REVIEW_2_FINAL_REPORT.md               Form FF 180 — final submission
│   └── USER_GUIDE.md                          Operational user guide
│
└── notebooks/
    └── nemo_evaluator_colab.ipynb             6-step Google Colab pipeline
```

**Code volume:** ~2,250 lines of hand-written Python across 14 modules, plus ~2,800 lines of tests/configs/docs and ~650 KB of persisted run artifacts.

**Notable absences:** no `.gitignore`, no `LICENSE` (despite the MIT badge at `README.md:6`), no CI configuration, no `conftest.py`, no `tests/__init__.py`, and stale `__pycache__/*.cpython-314.pyc` files (compiled by a different Python minor version than the installed 3.13.9).

---

## 6. System Architecture — 8 Layers Mapped to Real Modules

`README.md:24` advertises an "8-Layer Modular Architecture", but the ASCII diagram at `README.md:78-92` renders only **6** boxes. The design intent recorded in `docs/REVIEW_1_PROGRESS_REPORT.md:20-21` names all 8. Below is the authoritative mapping of the 8 conceptual layers onto the modules that actually implement them.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ LAYER 8 — PRESENTATION                                                       │
│   src/cli/nel.py  (Click CLI: list | eval run | compare | gate)              │
│   src/dashboard/app.py  (Streamlit: 5 tabs)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ LAYER 7 — ORCHESTRATION & CONTROL PLANE                                      │
│   configs/eval_config_default.yaml  → job selection, limits, gate thresholds │
│   src/cli/nel.py:96-146        → command-line control flow, CLI overrides     │
├──────────────────────────────────────────────────────────────────────────────┤
│ LAYER 5 — CORE EVALUATION ENGINE                                             │
│   src/engine/runner.py     → orchestration loop, aggregation, artifact build │
│   src/engine/solvers.py    → prompt construction (simple / cot / harbor)     │
│   src/engine/sandbox.py    → CodeSandbox for HumanEval execution             │
├──────────────────────────────────────────────────────────────────────────────┤
│ LAYER 4 — DATA PLANE & MODEL GATEWAY                                         │
│   4a Data:    src/benchmarks/base.py, loader.py, datasets/*.jsonl           │
│   4b Gateway: src/gateway/model_client.py, interceptor.py, mock_models.py    │
├──────────────────────────────────────────────────────────────────────────────┤
│ LAYER 3 — SCORING & JUDGEMENT                                                │
│   src/scoring/rule_scorers.py  → MCQ + math scorers                          │
│   src/scoring/statistics.py    → Wilson CI, McNemar, Quality Gate            │
│   src/scoring/llm_judge.py     → pairwise LLM judge (present, not wired)    │
├──────────────────────────────────────────────────────────────────────────────┤
│ LAYER 6 — COMPARISON ENGINE                                                  │
│   src/comparison/diff_engine.py → paired diff, flip detection                │
├──────────────────────────────────────────────────────────────────────────────┤
│ LAYER 2 — RESULTS STORE                                                      │
│   src/storage/schema.py   → Pydantic artifact models                        │
│   src/storage/run_store.py → JSON artifacts + SQLite registry + leaderboard  │
├──────────────────────────────────────────────────────────────────────────────┤
│ LAYER 1 — REPORTING ENGINE                                                   │
│   src/reporting/report_generator.py → Markdown + HTML reports                │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Data-flow summary (one line per hop):**

```
YAML config → EvaluationRunner → Solver builds prompt → UnifiedModelClient
   → InterceptorProxy cache lookup → provider (mock | NIM/OpenAI | Ollama)
   → raw text back → Benchmark.evaluate → RuleScorers | CodeSandbox
   → ItemResult accumulated → BenchmarkSummary (accuracy + Wilson CI)
   → RunArtifact → RunStore (JSON file + SQLite row) → DiffEngine | ReportGenerator | Dashboard
```

**Why the layering matters:** the scoring layer never imports the gateway, and the gateway never imports the scorers. A model backend can be swapped (mock → live NIM) without touching a scoring line, and a new scorer can be added without touching provider code. The only coupling points are the `RunArtifact` contract in `src/storage/schema.py` and the `Dict[str, Any]` response envelope returned by `UnifiedModelClient.generate`.

---

## 7. Layer-by-Layer Deep Dive

### 7.1 Layer 8 — Presentation: CLI (`nel`)

**File:** `src/cli/nel.py` (226 lines) · **Framework:** Click 8.x rendered with Rich 13.x.

**Bootstrap:** `src/cli/nel.py:1-2` inserts the repository root onto `sys.path` so `src.*` imports resolve when invoked as a script rather than an installed package. Lines 3-4 then re-import `os` and `sys` redundantly.

**Helper:** `load_yaml()` (`:19-22`) returns `{}` for a missing file rather than raising — so a mistyped `--config` path silently produces an all-default run instead of an error (§18-D5).

#### Commands

| Command | Line | Purpose |
|---|---|---|
| `nel list` | `:30` | Prints three Rich tables: configured models, configured benchmarks, 10 most recent runs |
| `nel eval run` | `:89` | Executes evaluation(s) and persists artifacts |
| `nel compare` | `:148` | Paired diff of two runs + McNemar + report export |
| `nel gate` | `:196` | Release-readiness decision |

Verified surface: `python src/cli/nel.py --help` lists exactly these four commands plus the `eval` subgroup. **`nel report` does not exist**, despite `README.md:39` advertising it.

#### `nel list` (`:31-82`)

Three tables:
- **Models** (`:39-48`): Model ID, Display Name, Provider, Context Window, Strengths — from `configs/models_config.yaml`.
- **Benchmarks** (`:52-61`): Benchmark ID, Name, Domain, Scorer, Description — from `configs/benchmarks_config.yaml`.
- **Runs** (`:65-82`): 10 most recent SQLite rows — Run ID, Model, Accuracy, Mean Latency, Tokens, Timestamp.

Columns are hardcoded rather than config-driven, so a model missing `name` or `strengths` renders empty cells rather than failing.

#### `nel eval run` (`:89-146`)

Options: `--config`, `--model`, `--bench` (multi-value), `--limit`, `--solver {simple,cot,harbor}`, `--job-name`.

Resolution logic (`:101-104`): an explicit `--model`/`--bench`/`--limit` **overrides** the config; otherwise the config's `models`, `benchmarks`, `limit_per_benchmark` and `seed` are used, with hardcoded fallbacks if the config is empty.

Control flow (`:115-133`): iterates model IDs **sequentially** — no concurrency, despite `eval_config_default.yaml:27` declaring `concurrency: 4`. For each model it falls back to a synthetic `provider: "mock"` config if the ID is absent from `models_config.yaml` (`:116-121`) — helpful, but a typo'd model name produces a plausible-looking mock run rather than an error.

After each run it prints a per-benchmark Rich table (`:136-146`): Score/Total, Accuracy + 95% CI, Mean Latency.

#### `nel compare` (`:148-193`)

Loads both runs by ID (exiting 1 with a red message if either is missing), calls `DiffEngine.compare_runs`, and renders a summary Panel with accuracy delta, latency delta, flip counts, and the McNemar p-value plus significance verdict. Then a per-benchmark delta table with green/red delta colouring. Finally writes Markdown/HTML reports if `--output-md` / `--output-html` are given.

#### `nel gate` (`:196-223`)

Options `--baseline`, `--candidate`, `--min-acc` (default **0.65**), `--max-reg` (default **0.03**). These defaults are **hardcoded in the CLI** and are *not* read from the `quality_gate:` block in `configs/eval_config_default.yaml:35-40`, even though that block declares the same values — a duplicated source of truth (§18-D6). Renders a Panel with colour-coded PASS (green) / FAIL (red) plus the rationale string.

### 7.2 Layer 8 — Presentation: Streamlit Dashboard

**File:** `src/dashboard/app.py` (273 lines) · **Launch:** `streamlit run src/dashboard/app.py` · **Page config:** wide layout, expanded sidebar (`:17-22`).

**Path resolution:** `load_yaml()` (`:25-30`) resolves configs relative to the repo root; the run store opens at `../../runs` relative to the module (`:34-35`), so the dashboard is location-independent within the repo. Five tabs are declared at `:41-47`.

#### Tab 1 — Leaderboard (`:50-100`)

- Reads `RunStore.get_leaderboard()` (SQL grouped by model).
- Formats accuracy, Wilson CI, latency, cost, run count.
- **Bar chart** (`:74-85`): accuracy per model, text labels outside bars, y-axis fixed to `[0, 1.05]`.
- **Scatter** (`:88-100`): the **Pareto frontier** — x = mean latency, y = best accuracy, bubble size = total tokens, one colour per model, model name as text label. This is the most decision-relevant visualisation in the project: it exposes models that are accurate but needlessly slow.

#### Tab 2 — Radar & Domain Analytics (`:103-149`)

- Multi-select of runs (default: first 4).
- Hardcoded 5 categories (`:117`): General Knowledge, Mathematical Reasoning, Logical Reasoning, Code Generation, Question Answering.
- For each selected run, benchmark accuracies are grouped by `BenchmarkSummary.domain` and averaged within each domain (`:126-133`), then plotted as a closed `Scatterpolar` trace (`:137-142`) with `fill='toself'`.
- **Gap:** `custom_byob`'s domain (`Custom Evaluation`, `src/benchmarks/base.py:67`) is not in the hardcoded category list, so BYOB results are silently excluded from the radar (§18-D8).

#### Tab 3 — Model-vs-Model Diff & Flips (`:152-216`)

- Requires ≥2 runs, else shows a warning.
- Two selectboxes (baseline A, candidate B) labelled with model, run ID and accuracy.
- Four metric tiles (`:172-177`): accuracy delta, latency delta (with `delta_color="inverse"` so a slowdown reads as bad), candidate-win flip count, baseline-win flip count.
- McNemar significance line (`:179-181`).
- Per-benchmark delta dataframe (`:184-195`).
- **Flip Inspector** (`:197-216`): a radio toggle for which side's wins to show, then up to 10 `st.expander` cards, each rendering the prompt, ground truth, and side-by-side the two models' extracted answers (`st.error` / `st.success`) and full raw responses (`st.code`). This is the item-level evidence mechanism from §3.1(4).

#### Tab 4 — Launch Evaluation (`:219-251`)

A `st.form` allowing model selection (display-name formatted), solver selection, multi-benchmark selection (default: first 4), a sample-limit slider (2–15, default 5), and a job-name field. On submit it runs `EvaluationRunner.run_evaluation` inside a spinner, saves the artifact, then `st.rerun()`.

**Trade-off:** this runs **synchronously in the Streamlit server process**. With mock models and 4 benchmarks × 5 items it completes in ~1 second; with live NIM endpoints (`timeout=45s` per call, `src/gateway/model_client.py:59`) 20 items could block the UI for minutes with no incremental progress. There is no job queue, background worker, or cancellation (§18-D9).

#### Tab 5 — Automated Reports (`:254-273`)

If ≥2 runs exist, it compares the **oldest** run (A) against the **newest** (B) — `runs[-1]` vs `runs[0]`, correct because `list_runs()` orders by timestamp descending — generates the Markdown report, offers it as a download button, and renders a live preview.

### 7.3 Layer 7 — Orchestration & Control Plane (Configuration)

Three YAML files under `configs/`, all hand-authored. They are the **only** input surface; there is no programmatic job API, no database-backed job table, and no scheduler despite the "Job Scheduler" claim in `README.md:82`.

Responsibilities in practice:
1. **Model registry** — `models_config.yaml`: identity, provider, API base, context window, sampling params, per-million-token pricing, declared strengths.
2. **Benchmark registry** — `benchmarks_config.yaml`: identity, domain, description, scorer name, sample file, option count.
3. **Job definition** — `eval_config_default.yaml`: job name/description, seed, per-benchmark limit, model list, benchmark list, execution parameters, quality-gate thresholds.

**Two design observations worth noting for reviewers:**

- **The benchmark registry is descriptive, not authoritative.** `benchmarks_config.yaml` declares a `scorer:` field (`mcq_pro`, `math_cot`, `math_exact`, `mcq`, `code_execution`, `mcq_truthful`, `rule_or_judge`) but no code ever reads it. Scorer selection is hardcoded inside each `BaseBenchmark` subclass's `evaluate()` method (`src/benchmarks/base.py`). Adding a new scorer requires a code change, not a config change.
- **The execution block is aspirational.** `execution:` in `eval_config_default.yaml:26-32` declares `concurrency: 4`, `timeout_seconds: 30`, `retry_attempts: 3`, `cache_responses: true`. Of these, only the last is honoured (`InterceptorProxy` caching defaults on, `src/engine/runner.py:19`); the other three are read by nothing. The actual HTTP timeout is hardcoded to 45s for NIM/OpenAI and 60s for Ollama (`src/gateway/model_client.py:59,96`).

### 7.4 Layer 5 — Core Evaluation Engine

#### `src/engine/runner.py` — `EvaluationRunner` (157 lines)

The orchestrator. One instance is created per CLI invocation (`src/cli/nel.py:112`) and per dashboard submission; it owns a single `InterceptorProxy` shared across all models in that invocation (`:19`).

`run_evaluation()` signature (`:21-31`): `model_config`, `benchmark_ids`, `job_name`, `description`, `limit_per_benchmark`, `solver_type`, `seed`, `show_progress`.

**Run identity** (`:32`): `run_{UTC timestamp}_{6 hex of uuid4}` — e.g. `run_20260916_194736_474bc0`. Uniqueness guaranteed; sortability guaranteed by the timestamp prefix.

**Per-benchmark loop** (`:42-52`): resolve the benchmark class via the factory, load items with the limit, reset per-benchmark accumulators (correct count, latencies, tokens, cost).

**Per-item loop** (`:51-100`):
1. `solver.format_prompt(item, b_id)` — build the prompt.
2. `client.generate(prompt, benchmark_id, ground_truth, item_meta)` — query the model. Note `ground_truth` is passed *into* the gateway; this is what lets the mock answer at a controlled rate (§7.6), and it is a **deliberate design smell for live evaluation** — a real model client must never receive the answer (§18-D2).
3. Unpack `text`, `latency_sec`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `cost` from the response envelope.
4. `bench.evaluate(item, raw_text)` → `(is_correct, score, extracted_answer, error)`.
5. Accumulate and build an `ItemResult` (`:81-99`) carrying the full prompt, raw response, extracted answer, ground truth, correctness, all token counts, cost, latency, solver type and domain. **Storing the full prompt and response per item is what makes the flip inspector and audit trail possible** — the reason artifacts are ~54 KB each.

**Per-benchmark aggregation** (`:102-122`): accuracy, Wilson CI via `StatisticsEngine.wilson_confidence_interval(correct, n)`, mean and median latency, total tokens, total cost — all rounded for stable serialisation.

**Overall aggregation** (`:127-132`): micro-average over all items. Note this is a **pooled item-level** accuracy — every question counts equally, not a mean of per-benchmark accuracies. Defensible, but benchmarks with more samples would dominate if sizes were unequal.

**Artifact assembly** (`:134-156`): `RunArtifact` with run identity, model identity, overall metrics, per-benchmark summaries, all item results, and an `execution_params` echo (`solver_type`, `limit_per_benchmark`, `benchmarks`) for provenance.

`seed` is accepted and stored but **never used to seed any random number generator** (§18-D1).

#### `src/engine/solvers.py` — prompt formatters (35 lines)

| Solver | Class | Behaviour |
|---|---|---|
| `simple` | `SimpleChatSolver` (`:7-13`) | For MCQ, renders `A. …` … `J. …` and asks for the letter **directly**. For non-MCQ, returns the bare question. |
| `cot` | `ChainOfThoughtSolver` (`:16-23`) | For MCQ, asks to "think through step-by-step" and state `Answer: [OPTION]` on the final line. For non-MCQ, asks for step-by-step solution with the final answer last. |
| `harbor` | `HarborAgenticSolver` (`:25-28`) | Prepends `[Harbor Agentic Workflow]`, asks the model to "explore hypotheses, verify intermediate bounds". Does **not** preserve the options block. |

Solvers are pure prompt transformers — the correct extension point for few-shot exemplars, system prompts, or decoding strategies.

#### `src/engine/sandbox.py` — `CodeSandbox` (43 lines)

Executes model-generated Python in a separate process.

Composition (`:22-27`):
```python
full_program = f"""{code}

candidate = {entry_point}
{test_code}
check(candidate)
"""
```

Worker (`:5-17`): `exec(program, exec_globals)` in a fresh globals dict; if both `check` and `candidate` are present it calls `check(candidate)` again (a harmless redundancy — the composed program already calls it); then puts `(True, None)` on a queue. `AssertionError` is caught separately from generic `Exception`, so test failures are reported distinctly from runtime errors.

Isolation (`:29-38`): `multiprocessing.get_context("spawn")` — a fresh interpreter, the right choice on Windows, preventing inherited module state from masking missing imports in generated code. `proc.join(timeout=3.0)`; if still alive, `terminate()` then `join(1.0)`, returning a `TimeoutExpired` failure. Otherwise the queue is drained; an empty queue means the child died without reporting (segfault, `os._exit`, memory exhaustion), correctly surfaced as a failure rather than a false pass.

**Verified working:** `tests/test_scorers.py:40-50` passes a real `add` function through the sandbox and asserts success. An additional ad-hoc check confirmed that mock "successful" HumanEval generations execute and pass the dataset's unit tests 10/10 for both Llama-8B and Llama-70B in a single process.

**Security posture (state this honestly in a report):** this is a *timeout-isolated subprocess*, not a security sandbox. There is no memory limit, no CPU rlimit, no network namespace, and no syscall filtering. `exec` on untrusted text inside a subprocess bounds *hangs*, not *malicious behaviour* (§18-D3).

### 7.5 Layer 4a — Data Plane: Benchmark Layer

#### `src/benchmarks/base.py` (97 lines)

`BaseBenchmark` (`:7-18`) holds identity (`bench_id`, `name`, `domain`), resolves `sample_file` against an optional `base_dir`, and delegates loading to `BenchmarkLoader.load_jsonl`. Subclasses implement only `evaluate()`.

| Class | Benchmark | Domain | Scorer path | Line |
|---|---|---|---|---|
| `MMLUProBenchmark` | MMLU-Pro | General Knowledge | `score_mcq(valid_options="ABCDEFGHIJ")` | `:20-27` |
| `GSM8KBenchmark` | GSM8K | Mathematical Reasoning | `score_math` | `:29-36` |
| `MATH500Benchmark` | MATH-500 | Mathematical Reasoning | `score_math` | `:38-45` |
| `GPQABenchmark` | GPQA | Logical Reasoning | `score_mcq("ABCD")` | `:47-54` |
| `TruthfulQABenchmark` | TruthfulQA | Question Answering | `score_mcq("ABCD")` | `:56-63` |
| `CustomBYOBBenchmark` | Custom BYOB | Custom Evaluation | `score_mcq("ABCD")` | `:65-72` |
| `HumanEvalBenchmark` | HumanEval | Code Generation | `CodeSandbox.execute_code` | `:74-82` |

`get_benchmark(bench_id, base_dir)` (`:84-97`) is the factory: a dict maps ID → class, instantiates with `base_dir`, and raises `ValueError` listing valid IDs for unknown keys. Clean, extensible, and correctly fails loudly.

**Design note:** the 5 radar domains in `README.md:38` are produced by exactly these `domain` strings, so domain analytics are driven by the benchmark classes, not by config.

#### `src/benchmarks/loader.py` (27 lines)

`load_jsonl` (`:7-18`): streams line-by-line, skips blanks, `json.loads` each line, and applies the limit via a post-append check (`if 0 < limit <= len(items): break`). Correct for `limit=0` meaning "all".

`check_contamination` (`:20-27`): splits text into words, builds all n-grams of width `n` (default 8), and returns the fraction present in a supplied token set. This is the **contamination detector** referenced in `README.md:32`. Fully implemented and unit-testable but **never called by any pipeline component** — there is no training-corpus token set anywhere in the repo (§18-D4).

### 7.6 Layer 4b — Model Gateway

#### `src/gateway/model_client.py` — `UnifiedModelClient` (110 lines)

Constructor (`:9-16`) reads `id`, `provider` (default `mock`), `api_base` (default NVIDIA's `https://integrate.api.nvidia.com/v1`), the API key from the environment variable named by `api_key_env` (default `NVIDIA_API_KEY`), sampling `params`, and an optional interceptor.

`generate()` (`:18-42`) is a three-step pipeline:
1. **Cache lookup** — `interceptor.get_cached(model_id, prompt, params)`; return immediately on hit.
2. **Provider routing** (`:25-38`):
   - `provider == "mock"` **or** `api_key` empty → `MockModelEngine`
   - `provider in {"nvidia_nim", "openai"}` → `_call_openai_compatible`
   - `provider == "ollama"` → `_call_ollama`
   - anything else → `MockModelEngine` (silent fallback)
3. **Cache store** — `interceptor.set_cached(...)`.

> **Two consequences of the routing condition, both worth documenting:**
> (a) Because the check is `provider == "mock" or not self.api_key`, a config correctly set to `nvidia_nim` but run **without** `NVIDIA_API_KEY` set **silently produces mock results** rather than erroring. A user could believe they evaluated a live model when they did not. This is the highest-risk silent-failure path in the system (§18-D2).
> (b) `provider: "huggingface"` is advertised in `README.md:33` and in the `ModelConfig` docstring (`src/storage/schema.py:24`) but is **not implemented** — it falls through to the `else` branch and is served by the mock, again silently (§18-D4).

`_call_openai_compatible` (`:44-89`): POSTs `{model, messages:[{role:user,content:prompt}], temperature, max_tokens, top_p}` to `{api_base}/chat/completions` with a Bearer token and 45s timeout. Measures wall-clock latency with `time.time()`. Extracts `choices[0].message.content` and `usage.{prompt,completion}_tokens`, falling back to word-count estimates when `usage` is absent. **Cost model:** `(p_tokens/1e6 × cost_per_1m_input) + (c_tokens/1e6 × cost_per_1m_output)` from the model config. All exceptions are caught and returned as a **normal-shaped response** with `text = "API Error: ..."` and `error` set — so a total network outage produces a run full of failures rather than a crash. Defensible for batch evaluation, but the error string is then handed to the scorer, which simply marks it incorrect.

`_call_ollama` (`:91-110`): POSTs `{model, prompt, stream: false}` to `{api_base}/api/generate` with a 60s timeout, reads `response` plus Ollama's `prompt_eval_count`/`eval_count` counters, and hardcodes `cost = 0.0` (correct — local inference is free).

#### `src/gateway/interceptor.py` — `InterceptorProxy` (24 lines)

Cache key (`:10-12`): `sha256(f"{model_id}|{prompt}|{sorted(params.items())}")`. Including `model_id` and serialised params means different models or sampling settings never collide, and `sorted()` makes params serialisation order-independent.

Storage is a plain in-memory `dict` (`:8`) with no size bound, no TTL, and no persistence — appropriate for a single-process evaluation run, though a long-lived dashboard process would grow unbounded (bounded in practice by the number of distinct prompts, which is small here).

`README.md:24` and `docs/REVIEW_1_PROGRESS_REPORT.md:24` describe the interceptor as providing "response caching, rate limiting, and millisecond-level latency tracking". **Only caching exists** — no rate limiter, no latency instrumentation in this class; latency is measured by the callers (§18-D4).

A subtle behavioural note: because cached values include the originally-measured `latency_sec`, a cache hit reports the *first* call's latency, not the near-zero lookup time. This does not affect the saved runs (prompts are unique within a run) but would distort latency statistics if a prompt repeated.

#### `src/gateway/mock_models.py` — `MockModelEngine` (121 lines)

The zero-cost simulator that makes the whole platform demonstrable without API spend — and, per `docs/USER_GUIDE.md` and the notebook, the default mode for every model in `configs/models_config.yaml`.

**`MODEL_PROFILES`** (`:6-43`) — one entry per model, each with:
- `accuracy_bias`: per-benchmark probability of "succeeding" (e.g. Qwen-2.5-72B: `mmlu_pro 0.85, gsm8k 0.92, math_500 0.75, gpqa 0.65, humaneval 0.88, truthfulqa 0.82`).
- `mean_latency`: a latency scale in seconds.
- `cost_in` / `cost_out`: per-million-token prices, mirroring `models_config.yaml`.

These profiles are **hand-tuned to reproduce the qualitative shape of published leaderboards**, which is what makes the demonstration results plausible.

**The success/failure roll** (`:50-54`):
```python
q_id = item_meta.get("question_id", "default")
seed_str = f"{model_id}_{benchmark_id}_{q_id}"
rand_val = (hash(seed_str) % 10000) / 10000.0
should_succeed = (rand_val < bench_acc)
```
The intent is a deterministic per-question roll. **The intent is not achieved:** CPython randomises `hash()` for `str` per process (PYTHONHASHSEED), so `hash()` of the same string differs between runs. Verified: `hash('meta/llama-3.1-70b-instruct_mmlu_pro_mmlu_01') % 10000` returned `983` in one process and `873` in the next. Empirically, four identical invocations produced `{mmlu_pro: 7, gsm8k: 6, gpqa: 6}`, `{7, 7, 3}`, `{3, 7, 7}`, and `{8, 9, 5}`. This is the root cause of the variance documented in §13.4 and the reason `seed` has no effect (§18-D1).

**Latency simulation** (`:56-57`): `max(0.05, mean_latency × (0.85 + (rand_val % 0.3)))` yields 0.85–1.15× the profile latency, then `time.sleep(min(0.05, latency × 0.05))` adds a small real delay so the timing path is genuinely exercised.

**Response synthesis by benchmark family:**
- **MCQ** (`:62-78`): on success, emits the ground-truth letter with a generic justification; on failure, picks a deterministic non-answer option. DeepSeek models are given a `<think>…</think>` block to emulate chain-of-thought. Token counts from word count + 15.
- **Math** (`:80-96`): on success, emits a 3-step scaffold ending in the true value; on failure, perturbs the numeric answer by +3 (integers) or +2.5 (non-integers), or appends `_err`. Non-DeepSeek models use the GSM8K `#### value` convention; DeepSeek uses `\boxed{value}` — both conventions the real scorer supports.
- **HumanEval** (`:98-107`): on success, concatenates the function prompt with the canonical solution; on failure, injects a deliberate bug by rewriting `==`→`!=` and/or `<`→`>`, falling back to `return False` if neither operator is present.
- **Fallback** (`:108-110`): a generic templated string.

**Cost** (`:112`): same formula as the live client, but using the profile's prices. Token counts are synthetic but *consistently* so (prompt = word count + 25, completion = word count + 15/20/25 by family), making cost and token comparisons across models internally coherent.

### 7.7 Layer 3 — Scoring & Judgement

#### `src/scoring/rule_scorers.py` — `RuleScorers` (104 lines)

**`extract_mcq_answer`** (`:6-33`) — a **cascading, priority-ordered** pattern list, the right design because explicit answer markers should always beat incidental letters:
1. `(?:the\s+correct\s+answer\s+is|…|answer\s*:|choice\s*:|option\s*:)\s*[\[\(\{]?([A-J])[\]\)\}]?`
2. `\\boxed\{([A-J])\}`
3. `\*\*([A-J])\*\*` (bolded)
4. `[\[\(]([A-J])[\]\)]` (bracketed)
5. `^([A-J])(?:\.|\)|\:|$|\s)` (line-initial)
6. `(?:^|\s)([A-J])\s*$` (line-final, standalone)

Then a **fallback**: take the *last* standalone `\b[A-J]\b` token in the text. Every candidate is validated against the `valid_options` set before being returned, so a model answering "J" on a 4-option GPQA question yields `None` rather than a false match. Returns `None` when nothing matches — correctly scored incorrect rather than crashing. Taking the *last* token in the fallback is the right heuristic for CoT output, where the conclusion appears at the end.

**`score_mcq`** (`:36-40`): case-insensitive equality of extracted vs ground truth; `(False, extracted)` if either is missing — note it still returns the extraction, so the flip inspector can show *what* the model said even when it was wrong.

**`extract_math_answer`** (`:43-69`): `\boxed{...}` → `#### value` (GSM8K) → "the final answer is …" phrasing → a trailing `= number` → last numeric literal. Ordered most- to least-explicit, and the numeric fallback correctly handles the very common case of a model that reasons to a number with no marker.

**`normalize_math_str`** (`:72-77`): lowercases; strips `$`, thousands commas and spaces; unwraps `\text{...}`; removes `\left`/`\right`. This is what makes `\boxed{\text{23}}` compare equal to `23` and `1,000` equal to `1000`.

**`score_math`** (`:80-104`): three-tier acceptance — (1) normalised string equality, (2) numeric equality within `1e-5` after converting `"a/b"` to a float, (3) otherwise incorrect. Tier 2 is what makes the unit test `score_math("The probability is 2/12", "1/6") == True` pass (`tests/test_scorers.py:36-38`) — genuine mathematical equivalence, not string matching. This is a meaningfully better scorer than most quick implementations.

#### `src/scoring/statistics.py` — `StatisticsEngine` (73 lines)

See §10 for the full methodology.

#### `src/scoring/llm_judge.py` — `LLMJudge` (57 lines)

An LLM-as-a-judge harness for pairwise comparison, implementing **position-bias debiasing**. `evaluate_pairwise_debiased()` (`:35-56`) asks the judge the same question twice with the responses swapped, then reports a winner only if both orders agree:
- forward says A **and** reverse says B → `Model_1`
- forward says B **and** reverse says A → `Model_2`
- any disagreement, or any tie → `Tie`

This is the correct debiasing pattern and mirrors MT-Bench-style methodology. The prompt template (`:7-21`) requests `Winner: [[A]] / [[B]] / [[Tie]]` plus a brief explanation; `parse_judge_winner` (`:25-32`) parses exactly that token.

**Status: implemented but not integrated.** No module imports `LLMJudge`; no pipeline stage invokes pairwise judging; the `custom_byob` config advertises scorer `rule_or_judge` (`configs/benchmarks_config.yaml:59`) but `CustomBYOBBenchmark.evaluate` only ever calls `score_mcq` (`src/benchmarks/base.py:71`) (§18-D4).

### 7.8 Layer 2 — Results Store

#### `src/storage/schema.py` (94 lines) — Pydantic v2 contracts

| Model | Purpose | Key fields |
|---|---|---|
| `EvalItem` (`:5-19`) | Dataset item schema | `question_id`, `question`, `options`, `answer`, `reasoning`, `solution`, `subject`, `domain`, `canonical_solution`, `test`, `entry_point`, `metadata` |
| `ModelConfig` (`:21-31`) | Model registry schema | mirrors `models_config.yaml` + `cost_per_1m_*`, `strengths` |
| `BenchmarkConfig` (`:33-40`) | Benchmark registry schema | mirrors `benchmarks_config.yaml` |
| `ItemResult` (`:42-60`) | One graded model response | prompt, `raw_response`, `extracted_answer`, `ground_truth`, `is_correct`, `score`, latency, all token counts, `cost`, `error`, `solver_type`, `domain` |
| `BenchmarkSummary` (`:62-74`) | Per-benchmark rollup | `correct_samples`/`total_samples`, `accuracy`, `ci_lower`/`ci_upper`, mean/median latency, tokens, cost |
| `RunArtifact` (`:76-94`) | One complete evaluation run | identity (`run_id`, `job_name`, `timestamp`, `git_hash`, `seed`), model identity, `overall_accuracy` + CI, `mean_latency`, `total_tokens`, `total_cost`, `benchmark_summaries`, `item_results`, `execution_params` |

Design strengths: every numeric field is explicitly rounded before storage, so artifacts are byte-stable for identical inputs; `metadata: Dict[str, Any]` escape hatches prevent schema churn; `git_hash` exists for provenance (though see §18-D7).

Design weakness: `EvalItem`, `ModelConfig` and `BenchmarkConfig` are **never used** — the pipeline passes raw `Dict[str, Any]` from YAML and JSONL straight through, so these three models are decorative documentation of the intended shape (§18-D4).

#### `src/storage/run_store.py` — `RunStore` (94 lines)

**Dual persistence strategy** — the key architectural decision of this layer:
- **JSON artifact per run** (`runs/{run_id}.json`, `save_run` `:35-38`): the complete, self-describing, portable record including all 60 `ItemResult`s, ~54 KB each. `get_run` reads this (`:64-70`), which is what makes runs diffable, archivable, and independently verifiable.
- **SQLite registry row** (`runs/runs_registry.db`, `:40-61`): a 13-column summary for fast querying and leaderboard aggregation.

`CREATE TABLE IF NOT EXISTS` (`:16-32`) is idempotent; `INSERT OR REPLACE` (`:42`) keyed on `run_id` makes `save_run` safely re-runnable for the same run.

`list_runs` (`:72-76`): `SELECT * … ORDER BY timestamp DESC` with `sqlite3.Row` → dicts. This feeds the CLI runs table and all dashboard selectors.

`get_leaderboard` (`:78-94`): groups by `model_id`, taking `MAX(overall_accuracy) as best_accuracy`, `AVG(mean_latency)`, `SUM(total_tokens)`, `SUM(total_cost)`, `COUNT(run_id) as total_runs`, and selecting bare `ci_lower`, `ci_upper`, `run_id as latest_run_id`.

> **Subtlety worth knowing before modifying this query:** the bare `ci_lower`/`ci_upper`/`run_id` columns have no aggregate function. SQLite's documented min/max special case makes them take values from the row that produced the `MAX`, so the CI shown does correspond to the best run — but this is engine-specific, it silently breaks on PostgreSQL/MySQL, and the alias `latest_run_id` is a **misnomer**: it is the *best-accuracy* run, not the most recent (§18-D10).

### 7.9 Layer 6 — Comparison / Diff Engine

#### `src/comparison/diff_engine.py` — `DiffEngine` (101 lines)

`compare_runs(run_a, run_b)` is a static method returning one dictionary consumed identically by the CLI, the dashboard, and the report generator. Five sections:

**1. Aggregate deltas** (`:9-11`): accuracy, mean latency, total cost, all as `b - a`.

**2. Per-benchmark comparison** (`:14-33`): iterates the **union** of benchmark IDs from both runs, so a benchmark present in only one run still appears (with 0.0 for the missing side) instead of vanishing. Each entry carries both accuracies, the delta, both confidence intervals, and both latencies.

**3. Item alignment** (`:36-39`): builds `{question_id: ItemResult}` maps for both runs and intersects the keys — `common_keys`. Alignment by `question_id` is exactly what makes the paired McNemar test valid: both models are scored on *the same questions in the same order*. Caveat: `question_id` must be globally unique across benchmarks for this to hold. The shipped datasets use distinct prefixes — `mmlu_`, `gsm_`, `math_`, `gpqa_`, `tqa_`, `byob_`, and `HumanEval/N` — so it does hold, but a BYOB set reusing IDs would silently corrupt the pairing (§18-D11).

**4. McNemar test** (`:45`): delegates the aligned boolean vectors to `StatisticsEngine.mcnemar_test`.

**5. Flip classification** (`:48-78`): every aligned item is placed into exactly one of four buckets — `a_wins_flips` (A correct, B wrong), `b_wins_flips` (A wrong, B correct), `both_correct`, `both_failed`. Flip entries are **rich dicts**, not just IDs: prompt, ground truth, both raw responses, both extracted answers, and both latencies. This is precisely the object the dashboard's Flip Inspector renders and the report's Sample Flips section prints.

The returned dict (`:80-101`) has 21 keys — the full comparison contract for every downstream consumer.

### 7.10 Layer 1 — Reporting Engine

#### `src/reporting/report_generator.py` — `ReportGenerator` (93 lines)

A Jinja2 `Template` (`MARKDOWN_TEMPLATE`, `:5-61`) rendered with a single `comparison` variable.

**Report structure (4 sections):**
1. **Executive Summary** — run IDs and model IDs, aligned item count, both accuracies, net delta, and the McNemar statistic / p-value / significance verdict.
2. **Benchmark Breakdown** — a Markdown table with Benchmark, Domain, both accuracies, and a signed delta. Jinja whitespace control (`{% for %}` on its own line) produces the blank-line-separated rows visible in `comparison.md`.
3. **Flip Analysis** — the four bucket counts, then the **first 3** B-wins flips expanded in full: question ID, benchmark, complete prompt, ground truth, and each model's extracted answer with a correct/incorrect annotation. The model names are interpolated into the "(Incorrect)"/"(Correct)" annotations, so the labels are correct *by construction* (these come from `b_wins_flips`, where B is correct and A is not).
4. **Resource & Computational Efficiency** — mean latency delta and inference cost delta, closing with the SDG 9 / SDG 12 compliance line.

**Formatting idioms:** `"%+0.1f"|format(x)` for signed percentages and `"%+0.6f"|format(cost)` for the six-decimal USD delta. Both used consistently, so the report's numbers are machine-aligned and diff-friendly.

`generate_html` (`:69-93`) wraps the Markdown in a minimal, self-contained HTML document — inline CSS only (system font stack, collapsed-border table, subtle `h1/h2/h3` bottom borders, `<code>`/`<pre>` styling), no external assets or JavaScript, so the file is portable and printable. Honest limitation: the Markdown is embedded in a `<pre>` block rather than converted to HTML, so it renders as formatted text rather than a styled document (§18-D14).

---

## 8. Data Model & On-Disk Formats

### 8.1 Run artifact JSON structure

Verified against `runs/run_20260916_201103_5a9d07.json`:

```jsonc
{
  "run_id": "run_20260916_201103_5a9d07",
  "job_name": "baseline_multi_model_evaluation",
  "description": "Comprehensive evaluation across 6 flagship LLMs and 6 standard benchmarks",
  "timestamp": "2026-09-16T20:11:22.821848Z",
  "git_hash": "HEAD",                    // static default, not real provenance
  "seed": 42,                            // recorded but never used to seed anything
  "model_id": "deepseek/deepseek-r1-distill-llama-70b",
  "model_name": "DeepSeek-R1-Distill-Llama-70B",
  "provider": "mock",
  "overall_accuracy": 0.8833,
  "overall_ci_lower": 0.7782,
  "overall_ci_upper": 0.9423,
  "mean_latency": 1.452,
  "total_tokens": 5707,
  "total_cost": 0.004562,
  "benchmark_summaries": {
    "mmlu_pro": {
      "benchmark_id": "mmlu_pro", "benchmark_name": "MMLU-Pro",
      "domain": "General Knowledge",
      "total_samples": 10, "correct_samples": 8, "accuracy": 0.8,
      "ci_lower": 0.4909, "ci_upper": 0.8817,
      "mean_latency": 1.476, "median_latency": 1.47,
      "total_tokens": 906, "total_cost": 0.000825
    }
    // … gsm8k, math_500, gpqa, humaneval, truthfulqa
  },
  "item_results": [ /* 60 × ItemResult */ ],
  "execution_params": {
    "solver_type": "simple",
    "limit_per_benchmark": 10,
    "benchmarks": ["mmlu_pro","gsm8k","math_500","gpqa","humaneval","truthfulqa"]
  }
}
```

### 8.2 SQLite registry schema

```sql
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, job_name TEXT, timestamp TEXT,
    model_id TEXT, model_name TEXT, provider TEXT,
    overall_accuracy REAL, ci_lower REAL, ci_upper REAL,
    mean_latency REAL, total_tokens INTEGER, total_cost REAL,
    artifact_path TEXT
);
```

12 rows currently persisted; `runs_registry.db` is 12 KB.

### 8.3 Dataset JSONL schemas (verified per benchmark)

| Benchmark | ID field | Question field | Options | Answer | Extra fields |
|---|---|---|---|---|---|
| MMLU-Pro | `question_id` (`mmlu_01`) | `question` | 10 strings | letter `A`–`J` | `domain` (e.g. `CS`) |
| GSM8K | `question_id` (`gsm_01`) | `question` | — | numeric string | — |
| MATH-500 | `question_id` (`math_01`) | `question` | — | numeric string | `subject` (e.g. `Algebra`) |
| GPQA | `question_id` (`gpqa_01`) | `question` | 4 strings | letter `A`–`D` | — |
| TruthfulQA | `question_id` (`tqa_01`) | `question` | 4 strings | letter `A`–`D` | — |
| HumanEval | `task_id` (`HumanEval/0`) | **`prompt`** | — | — | `canonical_solution`, `test`, `entry_point` |
| Custom BYOB | `question_id` (`byob_01`) | `question` | 4 strings | letter `A` | `domain` (`Enterprise Architecture`) |

**Two schema details the code depends on:**
- HumanEval uses `task_id` and `prompt` rather than `question_id` and `question`. The codebase handles this defensively in two places: `get_solver` uses `item.get("question") or item.get("prompt", "")` (`src/engine/solvers.py:9`) and `runner.py:79` uses `item.get("question_id") or item.get("task_id", …)`. The BYOB dataset also carries a `domain` field that is **not** propagated to `BenchmarkSummary.domain` (that comes from the class, `src/benchmarks/base.py:67`).
- The `EvalItem` Pydantic model documents this union schema but is not used to validate loaded data, so a malformed dataset line fails with a raw `json.JSONDecodeError` rather than a validation error.

---

## 9. End-to-End Execution Flow (Numbered Trace)

**Command:** `python src/cli/nel.py eval run --config configs/eval_config_default.yaml --limit 10`

| # | Step | Code location | Detail |
|---|---|---|---|
| 1 | CLI group dispatch | `src/cli/nel.py:96` | `run_cmd` receives parsed Click options |
| 2 | Load job config | `:98` | `load_yaml(config)` → dict; missing file ⇒ `{}` |
| 3 | Load model registry | `:99` | `configs/models_config.yaml` → `models` dict |
| 4 | Resolve selections | `:101-104` | flags override config; config overrides hardcoded defaults |
| 5 | Print job panel | `:106-110` | Rich Panel: job name, models, benchmarks, limit, solver |
| 6 | Construct runner + store | `:112-113` | `EvaluationRunner()` creates the shared `InterceptorProxy`; `RunStore("runs")` creates `runs/` if absent and opens the SQLite registry |
| 7 | **Per model** loop | `:115` | sequential; no parallelism |
| 8 | Resolve model config | `:116-121` | registry lookup, else synthetic mock config |
| 9 | Call `run_evaluation` | `:124-131` | see below |
| 10 | Persist artifact | `:132` → `run_store.py:35` | JSON dump + `INSERT OR REPLACE` |
| 11 | Print per-benchmark table | `:136-146` | Score/Total, Accuracy + Wilson CI, Mean Latency |
| 12 | **Per benchmark** loop | `runner.py:42` | for each of the 6 benchmark IDs |
| 13 | Instantiate benchmark | `runner.py:43` → `base.py:84` | factory lookup; unknown ID ⇒ `ValueError` |
| 14 | Load items | `runner.py:44` → `loader.py:7` | stream JSONL, stop at `limit` |
| 15 | **Per item** loop | `runner.py:51` | 10 iterations |
| 16 | Build prompt | `runner.py:52` → `solvers.py` | MCQ ⇒ lettered options + instruction; HumanEval ⇒ raw prompt |
| 17 | Query model | `runner.py:56` → `model_client.py:18` | cache lookup → provider route → response envelope |
| 18 | *(mock path)* | `mock_models.py:46` | hash-roll success; synthesise text; simulate latency; compute cost |
| 19 | *(cache store)* | `model_client.py:41` | SHA-256 key → dict |
| 20 | Score | `runner.py:71` → `base.py` | MCQ/math ⇒ `RuleScorers`; HumanEval ⇒ `CodeSandbox` (spawn subprocess, 3s timeout) |
| 21 | Accumulate | `runner.py:72-77` | correctness, latency, tokens, cost |
| 22 | Build `ItemResult` | `runner.py:81-99` | full prompt + response + extraction + metrics |
| 23 | Roll up benchmark | `runner.py:102-122` | accuracy, **Wilson CI**, mean/median latency, tokens, cost |
| 24 | Roll up run | `runner.py:127-132` | micro-averaged accuracy + overall Wilson CI, aggregate latency/tokens/cost |
| 25 | Build `RunArtifact` | `runner.py:134-156` | 19 top-level fields incl. all 60 `ItemResult`s |
| 26 | Return to CLI | `runner.py:157` | |
| 27 | Save + report | `run_store.py:35-62`, `nel.py:136` | disk persistence, then the summary table |

**Downstream flows:**
- **Compare:** `nel compare` → `RunStore.get_run` ×2 → `DiffEngine.compare_runs` → intersect `question_id` maps → `StatisticsEngine.mcnemar_test` → flip classification → Rich output → optional `ReportGenerator.generate_markdown` / `generate_html`.
- **Gate:** `nel gate` → `get_run` ×2 → `StatisticsEngine.evaluate_quality_gate` → PASS/FAIL + rationale.
- **Dashboard:** module import → `RunStore(runs_dir)` → tab renders from `get_leaderboard()` / `list_runs()` / `get_run()`; Tab 4 can call `run_evaluation` in-process and then `st.rerun()`.

---

## 10. Statistical Methodology

All statistics live in `src/scoring/statistics.py` (73 lines).

### 10.1 Wilson score confidence interval (`:6-17`)

**Purpose:** accuracy on 10 samples is extremely imprecise. Reporting `80%` alone is misleading; the Wilson interval makes the uncertainty visible and is what allows the leaderboard's "95% CI" column to be honest.

**Formula as implemented:**
```
z       = stats.norm.ppf((1 + confidence) / 2)          # 1.959964 for 95%
p       = k / n
denom   = 1 + z²/n
centre  = (p + z²/(2n)) / denom
margin  = z · sqrt( p(1-p)/n + z²/(4n²) ) / denom
CI      = [ max(0, centre - margin), min(1, centre + margin) ]
```

**Why Wilson rather than the textbook normal approximation:** the Wald interval is `p ± z√(p(1-p)/n)`, which is degenerate at the boundaries — at 10/10 it returns the impossible upper bound 1.0 exactly and an interval extending below 0. Wilson is a genuine improvement in exactly the regime this project operates in, since high-scoring models on 10-item samples hit the boundary routinely.

**Behaviour at the extremes (n = 10, 95%):**

| k/n | Wilson 95% CI | Interpretation |
|---|---|---|
| 0/10 | [0.000, 0.278] | Correctly excludes high accuracy |
| 5/10 | [0.237, 0.763] | Honest about a 50% score on 10 items |
| 8/10 | [0.491, 0.882] | Wide, as it must be |
| 10/10 | [0.723, 1.000] | **Not** the impossible [1.0, 1.0] |

**Verified:** `tests/test_statistics.py:8-11` asserts `wilson_confidence_interval(8, 10)` ∈ [0.40, 0.60] × [0.85, 1.0]. The CI values throughout the leaderboard (§13.2) match this formula exactly.

Edge cases: `n == 0` returns `(0.0, 0.0)` (`:8-9`); bounds clamped to [0, 1]; results rounded to 4 dp.

### 10.2 McNemar's paired significance test (`:19-45`)

**Purpose:** the statistical core of the project's central claim. A raw accuracy delta between two models is meaningless without knowing whether it exceeds chance. McNemar is correct here because **both models are evaluated on the same questions**, so observations are paired and the only informative data is the *discordant* pairs.

**2×2 contingency construction** (`:28-29`):
- `b` = items where **A correct, B wrong** (baseline wins)
- `c` = items where **A wrong, B correct** (candidate wins)

Items both models get right, and items both get wrong, carry **no information** about which is better and are correctly discarded from the test statistic. This is what makes McNemar far more statistically efficient than an unpaired two-proportion test at small n.

**Statistic as implemented** (`:35`):
```
chi2 = (|b - c| - 1)² / (b + c)
```
This is McNemar with **continuity correction** (Yates), standard practice that improves the normal approximation's behaviour when `b + c` is small — again a regime this project operates in constantly.

**p-value** (`:36`): `1 - stats.chi2.cdf(chi2, df=1)`, the upper tail of a chi-square with 1 degree of freedom. **Significance threshold** (`:41`): hardcoded `p < 0.05`.

**Edge cases:** mismatched lengths raise `ValueError` (`:21-22`); `n == 0` returns a null result (`:25-26`); `b + c == 0` (complete agreement, no flips) returns `chi2 = 0.0, p = 1.0, significant = False` (`:31-33`) — the statistically correct answer, since with no discordant pairs there is no evidence of a difference.

**Return payload** (`:38-45`): `statistic`, `p_value`, `significant`, plus `a_better_than_b_count`, `b_better_than_a_count`, `total_flips` — so the caller gets the effect size alongside the p-value, the correct pairing of inferential and descriptive evidence.

**Verified:** `tests/test_statistics.py:13-21` feeds 10 paired outcomes with `b=2, c=4` and asserts both counts.

**Known statistical limitations (state these in a viva):**
- The **continuity correction makes the test conservative** — it slightly reduces power, biasing toward "not significant". With `b + c < 10` (routine at `limit=10`) the exact binomial McNemar test would be more appropriate. Not implemented.
- **α is hardcoded** to 0.05 at `:41`; `eval_config_default.yaml:40` declares `significance_alpha: 0.05` but nothing reads it (§18-D6).
- **Multiple comparisons are unhandled.** Six models compared pairwise means up to 15 tests. At α = 0.05 uncorrected the family-wise error rate reaches ~26%, so roughly one in four "significant" pairwise claims would be a false positive. A Bonferroni or Holm correction across the comparison family is the rigorous addition (§18-D15).

### 10.3 Quality Gate (`:47-73`)

`evaluate_quality_gate(baseline_summary, candidate_summary, min_accuracy=0.65, max_regression=0.03)` returns `status` ∈ {`PASS`, `FAIL`} plus both accuracies, the delta, and a human-readable rationale.

**Decision logic:**
1. If `candidate_accuracy < min_accuracy` ⇒ **FAIL** ("below required minimum").
2. If `delta < -max_regression` ⇒ **FAIL** ("regression exceeds maximum allowable").
3. Otherwise ⇒ **PASS**, with a rationale citing achieved accuracy and delta.

The two failure modes are independent and both are reported, so a candidate that is both too weak *and* regressing yields a rationale naming both problems. This maps directly to a real release process: an absolute quality floor plus a regression guard against the incumbent.

> **Documentation discrepancy:** `README.md:37` advertises three outcomes — "PASS / FAIL / **INCONCLUSIVE**". The implementation only ever returns PASS or FAIL; there is no inconclusive branch (§18-D7).

---

## 11. Configuration Reference

### 11.1 `configs/models_config.yaml` (85 lines, 6 models)

| Model ID | Display Name | Provider | Ctx | Temp | Max Tok | $/1M in | $/1M out | Declared Strengths |
|---|---|---|---|---|---|---|---|---|
| `meta/llama-3.1-70b-instruct` | Llama-3.1-70B-Instruct | mock | 131072 | 0.1 | 1024 | 0.88 | 0.88 | General Knowledge, Math, Reasoning, Code |
| `meta/llama-3.1-8b-instruct` | Llama-3.1-8B-Instruct | mock | 131072 | 0.1 | 1024 | 0.18 | 0.18 | High Throughput, Low Latency, Cost-Efficiency |
| `qwen/qwen-2.5-72b-instruct` | Qwen-2.5-72B-Instruct | mock | 131072 | 0.1 | 1024 | 0.90 | 0.90 | Coding, Math, Multilingual |
| `mistralai/mixtral-8x7b-instruct` | Mixtral-8x7B-Instruct-v0.1 | mock | 32768 | 0.1 | 1024 | 0.60 | 0.60 | Fast MoE Inference, General Knowledge |
| `deepseek/deepseek-r1-distill-llama-70b` | DeepSeek-R1-Distill-Llama-70B | mock | 65536 | **0.6** | **2048** | 0.80 | 0.80 | Deep CoT, MATH-500, Reasoning |
| `openai/gpt-4o-mini` | GPT-4o-Mini | mock | 128000 | 0.1 | 1024 | 0.15 | **0.60** | Reliable Instruction Following, Cost-Efficiency |

All six default to `provider: "mock"`. Live switching is documented in `docs/USER_GUIDE.md:11-18`: obtain a key from `https://build.nvidia.com`, set `NVIDIA_API_KEY`, change `provider` to `nvidia_nim`.

**Notable:** DeepSeek is the only model with elevated sampling params (temp 0.6, 2048 max tokens), correctly reflecting its chain-of-thought reasoning mode. GPT-4o-Mini is the only model with asymmetric pricing (cheap input, pricier output). Mixtral's 32768 context is 4× smaller than the Llamas'. These are realistic, deliberate choices that make the mock profiles plausible.

### 11.2 `configs/benchmarks_config.yaml` (60 lines, 7 benchmarks)

| ID | Name | Domain | Scorer (declared) | Sample File | Options |
|---|---|---|---|---|---|
| `mmlu_pro` | MMLU-Pro | General Knowledge | `mcq_pro` | `datasets/mmlu_pro/sample.jsonl` | 10 |
| `gsm8k` | GSM8K | Mathematical Reasoning | `math_cot` | `datasets/gsm8k/sample.jsonl` | — |
| `math_500` | MATH-500 | Mathematical Reasoning | `math_exact` | `datasets/math_500/sample.jsonl` | — |
| `gpqa` | GPQA | Logical Reasoning | `mcq` | `datasets/gpqa/sample.jsonl` | 4 |
| `humaneval` | HumanEval | Code Generation | `code_execution` | `datasets/humaneval/sample.jsonl` | — |
| `truthfulqa` | TruthfulQA | Question Answering | `mcq_truthful` | `datasets/truthfulqa/sample.jsonl` | 4 |
| `custom_byob` | Custom BYOB | Custom Evaluation | `rule_or_judge` | `datasets/custom_byob/sample.jsonl` | — |

Note `custom_byob` is present here but **absent from the default job's benchmark list** (`eval_config_default.yaml:17-23`), so it does not run in the standard pipeline.

### 11.3 `configs/eval_config_default.yaml` (40 lines)

| Key | Value | Consumed by |
|---|---|---|
| `job_name` | `baseline_multi_model_evaluation` | `nel.py:104` |
| `description` | "Comprehensive evaluation across 6 flagship LLMs and 6 standard benchmarks" | passed to `run_evaluation` |
| `seed` | `42` | stored in artifact only — **not used for RNG** |
| `limit_per_benchmark` | `15` | `nel.py:103` |
| `models` | all 6 model IDs | `nel.py:101` |
| `benchmarks` | 6 IDs (no `custom_byob`) | `nel.py:102` |
| `execution.concurrency` | `4` | **nothing** |
| `execution.timeout_seconds` | `30` | **nothing** (hardcoded 45/60s) |
| `execution.retry_attempts` | `3` | **nothing** (no retry logic exists) |
| `execution.cache_responses` | `true` | matched by behaviour, not read |
| `execution.temperature` / `max_tokens` | `0.1` / `1024` | **nothing** (per-model `params` win) |
| `quality_gate.min_overall_accuracy` | `0.65` | **nothing** (CLI default 0.65) |
| `quality_gate.max_regression_delta` | `0.03` | **nothing** (CLI default 0.03) |
| `quality_gate.significance_alpha` | `0.05` | **nothing** (hardcoded 0.05) |

**This table is the single most important configuration finding in the project:** 9 of the 15 declared config keys are inert. The values that *are* honoured happen to match the hardcoded defaults, so behaviour is correct today — but the config file is not the source of truth it appears to be, and editing it will silently do nothing (§18-D6).

### 11.4 CLI flags overriding config

```
--config PATH     job definition            (default: configs/eval_config_default.yaml)
--model ID        single model              (overrides config.models)
--bench ID        repeatable                (overrides config.benchmarks)
--limit N         samples per benchmark     (overrides config.limit_per_benchmark)
--solver TYPE     simple | cot | harbor
--job-name NAME   tags the artifact
```

---

## 12. Dataset Inventory

| Benchmark | Items | Domain | Answer format | Ground-truth source | Notes |
|---|---|---|---|---|---|
| MMLU-Pro | **15** | General Knowledge (CS, Math, Physics, Law, …) | letter A–J | Public MMLU-Pro style, 10 options | Widest option space; hardest extraction case |
| GSM8K | **15** | Mathematical Reasoning | integer | Public GSM8K style | `#### value` convention |
| MATH-500 | **15** | Mathematical Reasoning (Algebra, …) | integer/expression | Public MATH-500 style | `\boxed{}` / LaTeX |
| GPQA | **15** | Logical Reasoning | letter A–D | GPQA-style graduate science | 4 options; graduate-level |
| HumanEval | **15** | Code Generation | executable Python | HumanEval-style (`HumanEval/0` …) | `canonical_solution` + `test` + `entry_point` |
| TruthfulQA | **15** | Question Answering | letter A–D | TruthfulQA-style misconceptions | 4 options |
| Custom BYOB | **2** | Custom Evaluation | letter A–D | Hand-authored enterprise sample | 2 items only |
| **Total** | **92** | 5 domains + 1 custom | — | — | 90 in the default job |

**Assessment of the dataset tier.** Each benchmark ships 15 curated items — enough to demonstrate the full pipeline end-to-end and to produce a leaderboard with real structure, and appropriate for a semester project where evaluation must run repeatedly at zero cost. They are **not** statistically sufficient for the accuracy claims to be read literally: at n = 10 (the actual `limit` used in every saved run) the 95% Wilson interval spans roughly 50 percentage points, as §10.1 shows. Any presentation of these results must state the sample size and CI alongside the point estimate — which is precisely why the platform computes them, and the reason the dashboards and reports display them.

The `custom_byob` set at 2 items is a **format demonstration** of the bring-your-own-benchmark path, not an evaluation set.

---

## 13. Experimental Results & Progress Evidence

### 13.1 Run inventory

Twelve runs are persisted in `runs/` and registered in `runs_registry.db`, forming **two complete batches of 6 models**, both on 16 September 2026, both at `limit_per_benchmark = 10` (60 graded items per run, 720 item evaluations total).

| # | Run ID | Model | Overall Acc. | 95% Wilson CI | Mean Latency | Tokens | Cost (USD) | Batch |
|---|---|---|---|---|---|---|---|---|
| 1 | `run_20260916_194736_474bc0` | Llama-3.1-70B-Instruct | 80.0% | [68.2%, 88.2%] | 0.741 s | 5,454 | 0.004802 | 1 |
| 2 | `run_20260916_194800_892c41` | Llama-3.1-8B-Instruct | 50.0% | [37.7%, 62.3%] | **0.282 s** | 5,354 | **0.000965** | 1 |
| 3 | `run_20260916_194822_f892ab` | Qwen-2.5-72B-Instruct | 86.7% | [75.8%, 93.1%] | 0.818 s | 5,458 | 0.004919 | 1 |
| 4 | `run_20260916_194845_49254a` | Mixtral-8x7B-Instruct-v0.1 | 71.7% | [59.2%, 81.5%] | 0.480 s | 5,419 | 0.003248 | 1 |
| 5 | `run_20260916_194907_9bfad6` | DeepSeek-R1-Distill-Llama-70B | 81.7% | [70.1%, 89.4%] | 1.415 s | 5,686 | 0.004546 | 1 |
| 6 | `run_20260916_194930_747a8a` | GPT-4o-Mini | 78.3% | [66.4%, 86.9%] | 0.343 s | 5,447 | 0.001865 | 1 |
| 7 | `run_20260916_200951_38bb2d` | Llama-3.1-70B-Instruct | 83.3% | [72.0%, 90.7%] | 0.745 s | 5,450 | 0.004797 | 2 |
| 8 | `run_20260916_201010_a149d8` | Llama-3.1-8B-Instruct | 58.3% | [45.7%, 69.9%] | 0.277 s | 5,365 | 0.000967 | 2 |
| 9 | `run_20260916_201027_dd439f` | Qwen-2.5-72B-Instruct | 73.3% | [61.0%, 82.9%] | 0.793 s | 5,444 | 0.004906 | 2 |
| 10 | `run_20260916_201046_a7f38f` | Mixtral-8x7B-Instruct-v0.1 | 46.7% | [34.6%, 59.1%] | 0.481 s | 5,334 | 0.003201 | 2 |
| 11 | `run_20260916_201103_5a9d07` | DeepSeek-R1-Distill-Llama-70B | 88.3% | [77.8%, 94.2%] | 1.452 s | 5,707 | 0.004562 | 2 |
| 12 | `run_20260916_201122_63080d` | GPT-4o-Mini | 61.7% | [49.0%, 72.9%] | 0.352 s | 5,361 | 0.001813 | 2 |

Every run records `provider = "mock"`, `solver_type = "simple"`, `seed = 42`, `git_hash = "HEAD"`.

### 13.2 Leaderboard (as the dashboard renders it)

Produced by `RunStore.get_leaderboard()` — best accuracy per model across both batches, averaged latency, summed cost.

| Rank | Model | Best Accuracy | Achieved on Run | Mean Latency (2 runs) | Total Cost (2 runs) | Runs |
|---|---|---|---|---|---|---|
| 1 | **DeepSeek-R1-Distill-Llama-70B** | **88.3%** | #11 (batch 2) | 1.434 s | $0.009108 | 2 |
| 2 | **Qwen-2.5-72B-Instruct** | **86.7%** | #3 (batch 1) | 0.806 s | $0.009825 | 2 |
| 3 | Llama-3.1-70B-Instruct | 83.3% | #7 (batch 2) | 0.743 s | $0.009599 | 2 |
| 4 | GPT-4o-Mini | 78.3% | #6 (batch 1) | 0.348 s | $0.003678 | 2 |
| 5 | Mixtral-8x7B-Instruct-v0.1 | 71.7% | #4 (batch 1) | 0.481 s | $0.006449 | 2 |
| 6 | Llama-3.1-8B-Instruct | 58.3% | #8 (batch 2) | **0.280 s** | **$0.001932** | 2 |

**Three decision-relevant readings of this table:**

1. **Accuracy leader:** DeepSeek-R1-Distill-70B at 88.3%, followed by Qwen-2.5-72B at 86.7% — consistent with `docs/REVIEW_2_FINAL_REPORT.md:22`, which named those two as top performers (citing batch-1 figures of 86.7% and 81.7% respectively).
2. **Efficiency leader:** Llama-3.1-8B wins decisively on both latency (0.280 s, ~5× faster than DeepSeek) and cost ($0.0019 for two runs, ~5× cheaper). Its 58.3% best accuracy is the price.
3. **Pareto structure:** the scatter plot in Tab 1 makes the trade-off concrete. GPT-4o-Mini (78.3% @ 0.348 s, $0.0018/run) is the most *efficient* point on the frontier — it beats the 8B on accuracy by 20 points at only +0.07 s. Llama-70B (83.3% @ 0.745 s) is dominated by GPT-4o-Mini on this evidence. Mixtral is dominated outright. Only DeepSeek and Qwen sit above the efficient frontier, and DeepSeek pays 1.43 s and 5× the cost of GPT-4o-Mini for ~10 points.

**This Pareto reading is the platform's most commercially useful output** and the part of the project that maps most directly to a real TCS-style model-selection engagement.

### 13.3 Per-benchmark matrix (correct out of 10, both batches)

| Benchmark | Domain | Llama-70B | Llama-8B | Qwen-72B | Mixtral-8x7B | DeepSeek-R1 | GPT-4o-Mini |
|---|---|---|---|---|---|---|---|
| **Batch 1** (19:47–19:49) | | | | | | | |
| MMLU-Pro | General Knowledge | 8/10 | 6/10 | **9/10** | 8/10 | 7/10 | 8/10 |
| GSM8K | Math. Reasoning | **10/10** | 7/10 | 9/10 | 7/10 | **10/10** | 9/10 |
| MATH-500 | Math. Reasoning | 7/10 | 7/10 | 8/10 | 5/10 | 6/10 | 7/10 |
| GPQA | Logical Reasoning | 4/10 | 2/10 | 7/10 | 3/10 | **8/10** | 3/10 |
| HumanEval | Code Generation | **10/10** | 0/10 | **10/10** | **10/10** | **10/10** | **10/10** |
| TruthfulQA | Question Answering | 9/10 | 8/10 | 9/10 | **10/10** | 8/10 | **10/10** |
| **Batch 2** (20:09–20:11) | | | | | | | |
| MMLU-Pro | General Knowledge | **10/10** | 8/10 | 5/10 | 5/10 | 8/10 | 6/10 |
| GSM8K | Math. Reasoning | 8/10 | 9/10 | **9/10** | 6/10 | 9/10 | 8/10 |
| MATH-500 | Math. Reasoning | 8/10 | 6/10 | 7/10 | 5/10 | **10/10** | 6/10 |
| GPQA | Logical Reasoning | 5/10 | 4/10 | 6/10 | 4/10 | **8/10** | **8/10** |
| HumanEval | Code Generation | **10/10** | 0/10 | **10/10** | 0/10 | **10/10** | 0/10 |
| TruthfulQA | Question Answering | 9/10 | 8/10 | 7/10 | 8/10 | 8/10 | 9/10 |

**Domain-level observations:**

- **HumanEval is bimodal, never partial.** Every HumanEval cell is either 10/10 or 0/10 — never 3/10 or 7/10. This follows directly from the mock's failure injection: the buggy rewrite (`==`→`!=`, `<`→`>`, or `return False`) breaks *every* assertion in a unit-test file, so one injected fault fails the whole problem. Real models produce intermediate HumanEval scores. This is a **simulation artefact that must be disclosed** if these tables are shown externally (§18-D16).
- **GPQA is the hardest and most discriminating domain.** Range 2/10–8/10 — the widest spread of any benchmark, and the only column where both batches agree on a clear ordering (DeepSeek consistently top). A 4-option MCQ at graduate level behaves as expected.
- **Llama-8B scored 0/10 on HumanEval in both batches** while its configured `humaneval` bias is 0.62. Verified cause: the sandbox is *not* at fault — mock "successful" HumanEval generations execute and pass the dataset's unit tests 10/10 when tested directly (§7.4). The 0/10 is entirely the hash-randomisation roll described below.
- **MMLU-Pro is the least stable column** (Qwen: 9/10 then 5/10; Llama-70B: 8/10 then 10/10). With 10 options and a 0.85 bias, small-sample variance compounds.

### 13.4 Run-to-run variance — a critical, quantified finding

Comparing batch 1 to batch 2 for the **same model, same config, same seed, same solver, same limit**:

| Model | Batch 1 | Batch 2 | Δ | Relative swing |
|---|---|---|---|---|
| Llama-3.1-70B | 80.0% | 83.3% | +3.3 pp | 4.2% |
| Llama-3.1-8B | 50.0% | 58.3% | +8.3 pp | 16.7% |
| **Qwen-2.5-72B** | **86.7%** | **73.3%** | **−13.3 pp** | **15.4%** |
| **Mixtral-8x7B** | **71.7%** | **46.7%** | **−25.0 pp** | **34.9%** |
| DeepSeek-R1-Distill-70B | 81.7% | 88.3% | +6.7 pp | 8.2% |
| **GPT-4o-Mini** | **78.3%** | **61.7%** | **−16.7 pp** | **21.3%** |

Mean absolute swing: **12.5 percentage points**. Mixtral moves by a third of its own score between two nominally identical runs.

**Root cause (verified experimentally):** `src/gateway/mock_models.py:53` derives the per-question success roll from Python's built-in `hash()`. CPython randomises string hashing per process, so the roll differs on every invocation despite `seed = 42` being recorded in every artifact. Confirmed directly: `hash('meta/llama-3.1-70b-instruct_mmlu_pro_mmlu_01') % 10000` returned `983` in one process and `873` in the next; four identical script invocations produced four different per-benchmark score sets.

**Why this matters for the project, stated plainly:**

1. **The word "reproducible" in `README.md` and the abstract is not currently supported by the mock path.** A reviewer who runs the pipeline twice and gets different numbers has found a real defect, not a misunderstanding.
2. **The batch-1 numbers in `docs/REVIEW_2_FINAL_REPORT.md` are not "the" results** — they are one sample from a distribution with ~12.5 pp spread. The report's claim that "Qwen-2.5-72B (86.7%)" leads is not robust: in batch 2 Qwen scored 73.3% and ranked 4th of 6, while DeepSeek led at 88.3%.
3. **The fix is small and high-value:** replace `hash(seed_str)` with a stable digest — `int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) % 10000`, or seed a `random.Random` derived from the same string. That makes the "deterministic zero-cost simulator" claim true, makes `seed` meaningful, and makes every number in the reports exactly reproducible. This is the highest-priority item in §19.
4. **Interaction with the statistics module:** the Wilson intervals correctly communicate that a single 10-item measurement is imprecise, but they assume the *sample* is the only source of uncertainty. Here the **generator** is also random, so true uncertainty is wider than the reported CI. Once determinism is fixed the reported CIs become honest; until then they understate uncertainty.

### 13.5 Progress against the Review 1 → Review 2 milestone plan

`docs/REVIEW_1_PROGRESS_REPORT.md:29-33` set four Review 2 milestones:

| # | Review 1 milestone | Status | Evidence |
|---|---|---|---|
| M1 | Execute comprehensive cross-model benchmark runs across 6 LLMs | **Achieved** | 12 persisted runs covering all 6 models × 6 benchmarks × 10 items; 720 item evaluations; `runs/runs_registry.db` has 12 rows |
| M2 | Implement statistical significance testing (McNemar paired test, Wilson CIs) | **Achieved** | `src/scoring/statistics.py` (both tests + gate); 3 passing unit tests; McNemar result published in `comparison.md` and both Review reports |
| M3 | Deploy interactive Streamlit Dashboard with item-level flip inspector | **Achieved** | `src/dashboard/app.py`, 5 tabs including the Flip Inspector (`:197-216`) rendering full prompts and both responses |
| M4 | Prepare Google Colab notebook for GPU cloud execution | **Achieved with caveats** (§18-D17) | `notebooks/nemo_evaluator_colab.ipynb`, 6 steps, GPU metadata set; depends on an uninitialised git remote |

### 13.6 Review 1 claimed deliverables, verified

| `REVIEW_1` claim (lines 16-27) | Verified? | Detail |
|---|---|---|
| Literature survey: HELM, MMLU-Pro, MT-Bench, lm-evaluation-harness, NeMo Evaluator | Documentation only | Cited in the report; no bibliography file in the repo |
| 8-layer architecture finalised | **Partially** | 8 layers identifiable in code (§6); README diagram shows 6 (§18-D18) |
| `UnifiedModelClient` supporting NIM, OpenAI-compatible, Ollama, Mock | **Yes, with one exception** | All four implemented; `huggingface` advertised but unimplemented (§18-D4) |
| `InterceptorProxy` with **caching, rate limiting, and millisecond-level latency tracking** | **Caching only** | No rate limiter, no latency instrumentation in this class (§18-D4) |
| Standardized benchmarks for all 7 sets | **Yes** | 92 items across 7 JSONL files |
| Safe multiprocessing code execution sandbox | **Yes, with a caveat** | Spawn isolation + 3s timeout works; not a security sandbox (§18-D3) |

### 13.7 Review 2 claimed results, verified against artifacts

| `REVIEW_2` claim (lines 20-28) | Verified? | Detail |
|---|---|---|
| 6 flagship models supported | **Yes** | All 6 in `models_config.yaml` and the mock profiles |
| Top accuracy: Qwen 86.7%, DeepSeek 81.7% | **Batch 1 only** | Correct for batch 1. Batch 2 reverses it (DeepSeek 88.3%, Qwen 73.3%); leaderboard best puts DeepSeek 1st (§13.2) |
| DeepSeek: 100% GSM8K, highest GPQA (80%) | **Yes** | Batch 1: GSM8K 10/10, GPQA 8/10. Batch 2: GSM8K 9/10, GPQA 8/10 (tied with GPT-4o-Mini) |
| Qwen & Llama-70B: 100% HumanEval pass@1 | **Yes** | Both 10/10 in both batches |
| Llama-8B: lowest latency (0.28 s) and lowest cost | **Latency yes; cost figure wrong** | Latency 0.280 s ✓. But lowest cost is **$0.00097/run**, not the "$0.0019/run" stated — $0.0019 is GPT-4o-Mini's figure |
| McNemar: p = 0.00052, Chi2 = 12.04 for 70B over 8B | **Yes** | Reproduced exactly in `comparison.md:17-19` |
| Flip Inspector: 21 items where 70B corrected 8B errors | **Yes** | `comparison.md:45` |
| Pareto accuracy-vs-latency frontier | **Yes** | `src/dashboard/app.py:88-100` |
| Colab notebook enables zero-cost cloud GPU execution | **Partial** | Notebook exists; the pipeline is pure-Python CPU work, so "GPU execution" is not used by any code path, and it depends on a GitHub remote that does not yet exist (§18-D17) |

---

## 14. Case Study: Llama-3.1-8B vs 70B Comparison

This is the project's worked example, fully materialised in `comparison.md` (root) and `runs/llama_8b_vs_70b_comparison.md`.

**Inputs:** Run A = `run_20260916_194800_892c41` (Llama-3.1-8B-Instruct, 50.0%); Run B = `run_20260916_194736_474bc0` (Llama-3.1-70B-Instruct, 80.0%). Both batch 1, `limit=10`, `solver=simple`.

### 14.1 Headline result

| Metric | Value |
|---|---|
| Total aligned benchmark questions | **60** |
| Llama-3.1-8B-Instruct accuracy | 50.0% |
| Llama-3.1-70B-Instruct accuracy | 80.0% |
| **Net accuracy delta** | **+30.0 percentage points** |
| McNemar χ² (continuity-corrected) | **12.0417** |
| **p-value** | **0.00052** |
| Statistically significant at α = 0.05 | **YES** |
| Mean latency delta | +0.459 s |
| Inference cost delta | +$0.003837 |

### 14.2 Per-benchmark breakdown

| Benchmark | Domain | 8B | 70B | Delta |
|---|---|---:|---:|---:|
| GSM8K | Mathematical Reasoning | 70.0% | 100.0% | **+30.0 pp** |
| MATH-500 | Mathematical Reasoning | 70.0% | 70.0% | 0.0 pp |
| GPQA | Logical Reasoning | 20.0% | 40.0% | +20.0 pp |
| HumanEval | Code Generation | 0.0% | 100.0% | **+100.0 pp** |
| MMLU-Pro | General Knowledge | 60.0% | 80.0% | +20.0 pp |
| TruthfulQA | Question Answering | 80.0% | 90.0% | +10.0 pp |

**The headline 30 pp aggregate delta is not evenly distributed.** It decomposes into one dominant driver (HumanEval +100 pp, itself a bimodal 0-vs-10 artefact) plus four moderate gains and one exact tie (MATH-500, where both scored 7/10). A reviewer asking "is the 70B really 30 points better?" should be answered with this table, not the aggregate — which is exactly the analytical habit the platform exists to teach.

### 14.3 Flip analysis — the item-level evidence

| Bucket | Count | Share of 60 |
|---|---:|---:|
| **Only 70B succeeded** (B-wins) | **21** | 35.0% |
| **Only 8B succeeded** (A-wins) | 3 | 5.0% |
| Both succeeded | 27 | 45.0% |
| Both failed | 9 | 15.0% |

Discordant pairs = 21 + 3 = **24**. The 21:3 asymmetry is what McNemar's test converts into χ² = 12.04, p = 0.00052.

**Cross-check of the statistic by hand:** with continuity correction, χ² = (|b − c| − 1)² / (b + c) = (|3 − 21| − 1)² / 24 = (18 − 1)² / 24 = 289/24 = **12.0417** — exactly the reported value, confirming the implementation matches the documented method. The 27 common successes and 9 common failures are correctly excluded from the statistic, as they must be.

### 14.4 Sampled flips (verbatim from `comparison.md:52-114`)

The report prints the first 3 B-win flips in full. All three are MMLU-Pro items, and all three are cases where the 8B picked a *plausible-sounding but wrong* option while the 70B picked the ground truth:

| Question | Topic | Ground Truth | 8B | 70B |
|---|---|---|---|---|
| `mmlu_05` | Max Byzantine faulty nodes f in N nodes (consensus theory) | **B** (`f < N/3`) | F (`f < √N`) ✗ | B ✓ |
| `mmlu_07` | Brønsted–Lowry base definition | **D** (Proton acceptor) | E (`OH⁻ in water`) ✗ | D ✓ |
| `mmlu_10` | Action producing greatest balance of happiness | **C** (Utilitarianism) | D (Egoism) ✗ | C ✓ |

`mmlu_05` and `mmlu_10` are excellent illustration points for a report: in both, the 8B's error is a *superficially* reasonable answer (a real asymptotic complexity bound; a real ethical doctrine) that fails on a technicality. This demonstrates that the flip inspector surfaces genuine reasoning-quality differences rather than random noise — with 10 options there is a 10% random-guess floor, so a systematic wrong-letter pattern is informative.

**`mmlu_07` warrants a data-quality note:** the ground-truth key is `D` = "Proton acceptor". By the Brønsted–Lowry definition a base **is** a proton acceptor, so `D` is the correct key and the 8B's `E` ("OH⁻ in water") is genuinely wrong. The item is sound; the distractor `B` ("Electron donor") additionally captures the *Lewis* base definition, making this a good discrimination item. Worth confirming that the full 15-item MMLU-Pro set has similarly vetted keys before external publication.

### 14.5 How to reproduce this case study

```powershell
python src/cli/nel.py compare `
  --run-a run_20260916_194800_892c41 `
  --run-b run_20260916_194736_474bc0 `
  --output-md comparison.md `
  --output-html comparison.html
```

Note: `comparison.md` (root) and `runs/llama_8b_vs_70b_comparison.md` are **two generations of the same report for the same run pair**; they differ only in benchmark row ordering (Jinja dict-iteration order at generation time). Content is otherwise identical.

---

## 15. Test Suite & Verification Status

**Result: 7/7 tests pass.** Command: `python -m unittest discover -s tests -v` → `Ran 7 tests in 0.198s / OK` on Python 3.13.9.

| Test | File:line | Asserts | Status |
|---|---|---|---|
| `test_wilson_ci` | `tests/test_statistics.py:8-11` | `wilson_confidence_interval(8, 10)` lower ∈ [0.40, 0.60], upper ∈ [0.85, 1.0] | **PASS** |
| `test_mcnemar` | `tests/test_statistics.py:13-21` | Result dict has `p_value`/`statistic`; `a_better_than_b_count == 2`, `b_better_than_a_count == 4` | **PASS** |
| `test_quality_gate` | `tests/test_statistics.py:23-36` | 0.50 → 0.80 with min 0.65 ⇒ PASS; 0.80 → 0.60 with min 0.65 ⇒ FAIL | **PASS** |
| `test_mcq_scorer` | `tests/test_scorers.py:9-23` | 3 cases: "The correct answer is B" ⇒ B ✓; `\boxed{F}` with 10 options ⇒ F ✓; "Option A" vs truth C ⇒ False, extraction still A | **PASS** |
| `test_math_scorer` | `tests/test_scorers.py:25-38` | 3 cases: GSM8K `#### 72` ✓; `\boxed{23}` ✓; **fraction equivalence** `2/12` == `1/6` ✓ | **PASS** |
| `test_code_sandbox` | `tests/test_scorers.py:40-50` | Real `add(a,b)` passes positive and negative assertions through the spawn-based sandbox; `err is None` | **PASS** |
| `test_diff` | `tests/test_diff.py:9-17` | Against the real persisted runs: comparison dict has `delta_accuracy`, `mcnemar`, `a_wins_flips`, `b_wins_flips` | **PASS** |

**What is well covered:**
- The two **statistically load-bearing** functions (Wilson, McNemar) have direct tests including flip-count effect sizes.
- The **scorers** are tested against all three real answer conventions (`####`, `\boxed{}`, prose), plus the mathematically-equivalent-fractions case, which is the hardest case.
- The **sandbox** is tested end-to-end with a real subprocess spawn and a real assertion-failure surface.
- `test_diff` is a genuine **integration test** against the 12 persisted runs, not a mock — it validates that artifacts on disk deserialise through Pydantic and that the diff engine produces the full contract.

**Gaps in coverage (candidates for the next test module):**

| Untested area | Why it matters |
|---|---|
| `InterceptorProxy` cache-key stability and hit/miss behaviour | A key collision would silently serve one model's response for another |
| `BenchmarkLoader.load_jsonl` limit semantics, missing file, malformed JSON | `limit=0` meaning "all" and the early-`break` boundary are easy to regress |
| `BenchmarkLoader.check_contamination` | Zero-call, zero-test utility — either test and wire it, or remove it |
| `LLMJudge.parse_judge_winner` and the debiasing agreement rule | Pure functions — trivially testable |
| `RuleScorers.extract_mcq_answer` negative cases (out-of-range letters, empty/None input) | Boundary robustness for untrusted model output |
| `MockModelEngine` determinism | **Would have caught the hash-randomisation defect.** A test asserting "same item → same verdict across two separate subprocesses" is the direct regression guard for §18-D1 |
| `CodeSandbox` timeout and crash paths | The 3s timeout and the "process died without reporting" branch are safety-critical and entirely untested |
| `RunStore` round-trip and `get_leaderboard` aggregation | The bare-column/`MAX` subtlety in the leaderboard SQL deserves a test with 2+ runs per model |
| `DiffEngine` `question_id` collision handling | A BYOB set reusing another benchmark's IDs would silently corrupt McNemar pairing |

**Infrastructure note:** there is no `pytest.ini` / pyproject test config, no `conftest.py`, and no `tests/__init__.py`. The suite runs correctly via `unittest discover` because each test file performs its own `sys.path` insertion (`tests/test_*.py:2-3`) — workable but repetitive. Adding `pytest` as a dev dependency would also allow parametrised scorer cases, which is where most of the value above lies.

---

## 16. Documentation & Report Inventory

| Artefact | Path | Contents | Audience |
|---|---|---|---|
| **This document** | `docs/PROJECT_DOCUMENTATION.md` | Complete technical + progress reference | All |
| Project README | `README.md` (92 lines) | Badges, team, 7 key features, quickstart, 6-box architecture diagram | External / evaluators |
| User guide | `docs/USER_GUIDE.md` (29 lines) | Config file purposes, NIM API key setup, dashboard tab reference | Operators |
| Review 1 report | `docs/REVIEW_1_PROGRESS_REPORT.md` (33 lines) | Form FF 180 mid-semester: literature survey, architecture, work completed, Review 2 milestones | Academic panel |
| Review 2 report | `docs/REVIEW_2_FINAL_REPORT.md` (30 lines) | Form FF 180 final: deliverables, results, significance, sustainability | Academic panel |
| Comparison report (current) | `comparison.md` (122 lines) | 8B vs 70B: exec summary, benchmark table, 3 sampled flips, efficiency deltas | Stakeholders |
| Comparison report (earlier) | `runs/llama_8b_vs_70b_comparison.md` (122 lines) | Same content, different row ordering | Archive |
| Colab notebook | `notebooks/nemo_evaluator_colab.ipynb` (6 cells) | 6-step pipeline: install → API key → clone → run → compare/gate → plot | Cloud users |

**Colab notebook step-by-step:**
1. Markdown overview: project identity, team, and the six benchmarks with one-line descriptions.
2. `!pip install -q git+https://github.com/NVIDIA-NeMo/Evaluator.git` plus the Python dependencies.
3. `getpass` prompt for `NVIDIA_API_KEY`, with a clear "leave blank for mock mode" path — good UX for a zero-cost demo.
4. `git clone` the project repo and `cd` into it.
5. `nel list` then `nel eval run --config … --limit 10`.
6. Glob `runs/*.json` by mtime, then `nel compare` and `nel gate` on the two newest runs, auto-emitting a Markdown report.
7. Load the leaderboard into pandas and render a Plotly bar chart inline.

Notebook metadata requests a GPU accelerator and declares Python 3.10.

**Documentation quality observations:**
- The README's `?` characters (lines 1, 12, 22, 44, 77) and the dashboard's `page_icon="??"` / tab labels are **mojibake** — emoji lost in a non-UTF-8 write. Same artefact in `comparison.md:51` and in the Review reports' "2026?27". Cosmetic but visible in a submitted document; trivially fixed by writing as UTF-8 (§18-D19).
- The README's architecture diagram shows 6 boxes while the prose claims 8 layers (§18-D18).
- `README.md:39` advertises `nel report`, which is not a command (§18-D20).
- `README.md:6` links to a `LICENSE` file that does not exist (§18-D21).

---

## 17. Feature Completeness Matrix (Claimed vs. Implemented)

Legend: **OK** implemented & exercised · **PARTIAL** implemented but not wired / mis-scoped · **MISSING** claimed but not implemented

| # | Claimed capability | Source of claim | Status | Evidence / Gap |
|---|---|---|---|---|
| 1 | 8-layer modular architecture | `README.md:24` | **PARTIAL** | 8 layers identifiable in code; README diagram shows 6 |
| 2 | MMLU-Pro, GSM8K, MATH-500, GPQA, HumanEval, TruthfulQA | `README.md:26-31` | **OK** | 90 items; all 6 load and score |
| 3 | Custom BYOB loader with n-gram contamination detection | `README.md:32` | **PARTIAL** | BYOB loads and scores; `check_contamination` never called |
| 4 | NVIDIA NIM via `integrate.api.nvidia.com` | `README.md:33` | **OK** | `_call_openai_compatible`; needs `NVIDIA_API_KEY` |
| 5 | OpenAI-compatible endpoints | `README.md:33` | **OK** | Same code path as NIM |
| 6 | Hugging Face Inference | `README.md:33` | **MISSING** | No `huggingface` branch; silently falls back to mock |
| 7 | Local Ollama | `README.md:33` | **OK** | `_call_ollama` |
| 8 | Built-in **deterministic** simulator | `README.md:33` | **PARTIAL** | Fully functional but **not deterministic** across processes (§13.4) |
| 9 | 95% Wilson confidence intervals | `README.md:35` | **OK** | Computed per benchmark and overall; shown in CLI, leaderboard, diff |
| 10 | Bootstrap confidence intervals | `README.md:35` | **MISSING** | No bootstrap implementation anywhere |
| 11 | Paired McNemar significance tests | `README.md:36` | **OK** | Verified by hand against `comparison.md` (§14.3) |
| 12 | Quality Gate PASS / FAIL / **INCONCLUSIVE** | `README.md:37` | **PARTIAL** | PASS and FAIL only; no INCONCLUSIVE branch |
| 13 | Streamlit dashboard: leaderboards | `README.md:38` | **OK** | Tab 1 + bar chart + Pareto scatter |
| 14 | Streamlit dashboard: 5-domain radar | `README.md:38` | **OK** | Tab 2; `custom_byob` domain excluded |
| 15 | Item-level flip inspector | `README.md:38` | **OK** | Tab 3; up to 10 expandable cards with both responses |
| 16 | Automated report exporter | `README.md:38` | **OK** | Markdown + HTML, CLI flags and dashboard download |
| 17 | CLI `nel list` | `README.md:39` | **OK** | Verified via `--help` |
| 18 | CLI `nel eval run` | `README.md:39` | **OK** | All options functional |
| 19 | CLI `nel compare` | `README.md:39` | **OK** | Reproduces `comparison.md` |
| 20 | CLI `nel gate` | `README.md:39` | **OK** | PASS/FAIL with rationale |
| 21 | CLI `nel report` | `README.md:39` | **MISSING** | Command does not exist |
| 22 | Google Colab notebook for GPU execution | `README.md:40` | **PARTIAL** | Notebook complete; no code path uses a GPU; depends on an uninitialised remote |
| 23 | Job scheduler | `README.md:82` | **MISSING** | Runs are sequential in-process; no queue, worker or scheduler |
| 24 | Concurrency (`concurrency: 4`) | `eval_config_default.yaml:27` | **MISSING** | Never read; execution strictly sequential |
| 25 | Retry with `retry_attempts: 3` | `eval_config_default.yaml:29` | **MISSING** | No retry logic exists |
| 26 | Configurable `timeout_seconds` | `eval_config_default.yaml:28` | **MISSING** | Hardcoded 45s / 60s in the client |
| 27 | LLM-as-a-judge scoring | `llm_judge.py` | **PARTIAL** | Debiased pairwise harness complete; no caller, no pipeline stage |
| 28 | 6 flagship models | `REVIEW_2:19` | **OK** | All 6 configured, profiled and run |
| 29 | Pareto accuracy-vs-latency frontier | `REVIEW_2:29` | **OK** | Dashboard Tab 1 scatter |
| 30 | `seed` controls reproducibility | `eval_config_default.yaml:4` | **MISSING** | Recorded in every artifact; never seeds an RNG |
| 31 | Unit test suite | this doc §15 | **OK** | 7/7 passing on Python 3.13.9 |
| 32 | Package installable as `nel` | `pyproject.toml:26` | **PARTIAL** | Entry point declared; package not installed in the current env |

**Summary tally: 20 OK · 9 PARTIAL · 8 MISSING.** The MISSING set is almost entirely (a) an over-claiming README, (b) an over-specified config file, and (c) two genuinely absent library capabilities (bootstrap CI, HF provider). The core spine — evaluation, scoring, statistics, comparison, storage, reporting — is complete and exercised.

---

## 18. Known Gaps, Defects & Risks — Action Register

Ordered by severity. Each entry: what, where, evidence, impact, fix.

### CRITICAL

#### D1 — The mock simulator is not deterministic; `seed` has no effect
- **Where:** `src/gateway/mock_models.py:53` (`hash(seed_str)`); `src/engine/runner.py:30` (`seed` accepted but unused); `src/storage/schema.py:82` (`seed` stored).
- **Evidence:** `hash()` of the same string returned 983 then 873 in consecutive processes. Four identical invocations gave `{mmlu_pro: 7, gsm8k: 6, gpqa: 6}`, `{7, 7, 3}`, `{3, 7, 7}`, `{8, 9, 5}`. Two full 6-model batches differ by a mean of 12.5 pp per model, up to 25 pp (Mixtral) — §13.4.
- **Impact:** Directly contradicts the project's central "reproducible benchmarking" claim. Every number in `comparison.md` and both Review reports is one draw from a wide distribution. A reviewer re-running the pipeline gets different figures, which is the fastest way to lose credibility on a reproducibility-themed project.
- **Fix:** replace with a stable digest — `int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest(), 16) % 10000` — and seed a `random.Random` from the same string for latency jitter. Cost: ~3 lines. Then regenerate all 12 artifacts and the reports.
- **Regression guard:** add a test that runs the same item through `MockModelEngine` in two separate subprocesses and asserts identical verdicts.

#### D2 — Silent fallback to mock when a live provider is configured without an API key
- **Where:** `src/gateway/model_client.py:25` — `if self.provider == "mock" or not self.api_key:`.
- **Evidence:** direct reading; `docs/USER_GUIDE.md:11-18` instructs users to set `provider: nvidia_nim` and `NVIDIA_API_KEY`, implying the provider field is what selects the backend.
- **Impact:** a user who flips `provider` to `nvidia_nim` but forgets (or mistypes) the env var gets a **complete set of mock results presented as a live evaluation**, with no warning. This is the most dangerous silent failure in the system because the output looks entirely normal.
- **Fix:** if `provider != "mock"` and the key is empty, either raise a clear error or set an `error` field on every response and surface a banner. Also log the *resolved* provider per run into `execution_params` so the artifact self-documents which backend actually served it.

#### D3 — `CodeSandbox` is a timeout, not a security boundary
- **Where:** `src/engine/sandbox.py:5-17` (`exec` on model output), `:29-38` (spawn + 3s join + terminate).
- **Evidence:** no `resource.setrlimit`, no container, no seccomp, no network restriction. `docs/REVIEW_1_PROGRESS_REPORT.md:27` and `README.md:30` both describe it as a "safe" / "isolated" sandbox.
- **Impact:** bounded against infinite loops and hangs, the realistic failure mode for LLM-generated code. **Not** bounded against `os.system`, file writes or fork bombs. Acceptable for a curriculum project with mock models; **not** acceptable if the platform is pointed at a live endpoint where model output is untrusted external input.
- **Fix (proportional):** (a) reword the documentation to "timeout-isolated subprocess" rather than "safe sandbox"; (b) if live evaluation is enabled, add `resource.setrlimit` for CPU/memory/address space and run with a scrubbed environment. Note Windows lacks `setrlimit`, so a Docker-based executor is the portable answer.

### HIGH

#### D4 — Six capabilities are implemented but disconnected from the pipeline

| Capability | Location | Status |
|---|---|---|
| LLM-as-judge (debiased pairwise) | `src/scoring/llm_judge.py` (all 57 lines) | No importer; no pipeline stage |
| n-gram contamination detection | `src/benchmarks/loader.py:20-27` | No caller; no training-corpus token set exists |
| `EvalItem` / `ModelConfig` / `BenchmarkConfig` schemas | `src/storage/schema.py:5-40` | Never used for validation |
| Hugging Face provider | `README.md:33`, `schema.py:24` | No branch; silently mocked |
| Rate limiting | `README.md:24`, `REVIEW_1:24` | Not implemented |
| Latency instrumentation in the interceptor | `README.md:24` | Latency measured by callers, not the interceptor |

- **Impact:** over-claims in the documentation (§17 has 9 PARTIAL rows). The most valuable to connect is the LLM judge, which would give TruthfulQA and BYOB a genuinely different scoring path and make the "Scoring & Judgement" layer name truthful.
- **Fix:** either wire each capability in behind a config flag, or remove it and correct the documentation. Leaving dead code that the documentation advertises is the worst of the three options.

#### D5 — Missing or mistyped config paths fail silently
- **Where:** `src/cli/nel.py:19-22` — `load_yaml` returns `{}` if the file is absent. The dashboard's `load_yaml` (`src/dashboard/app.py:25-30`) behaves the same.
- **Impact:** `nel eval run --config confg/eval_config_default.yaml` (typo) silently runs the hardcoded fallback — 1 model, 6 benchmarks, limit 10 — producing a valid-looking run of the wrong experiment. Similarly, an unknown `--model` ID fabricates a synthetic mock config (`src/cli/nel.py:116-121`).
- **Fix:** raise `FileNotFoundError` for an explicitly-specified `--config`; warn loudly (not silently) when a model ID is absent from the registry.

#### D6 — Nine configuration keys are inert; the config is not the source of truth
- **Where:** `configs/eval_config_default.yaml` `execution:` (6 keys) and `quality_gate:` (4 keys) vs. hardcoded values in `src/cli/nel.py:199-200` and `src/scoring/statistics.py:41,48`.
- **Impact:** a reviewer who tunes `concurrency`, `retry_attempts`, `timeout_seconds`, `temperature`, `max_tokens`, `min_overall_accuracy`, `max_regression_delta` or `significance_alpha` in the config will observe **no change in behaviour** and reasonably conclude the system is broken. Today the duplicated values happen to agree, so nothing is numerically wrong — the trap is purely structural.
- **Fix:** thread these through from config into `run_evaluation` and `evaluate_quality_gate`, or delete the keys. A startup warning for unrecognised config keys would also future-proof the format.

#### D7 — Quality Gate has no `INCONCLUSIVE` outcome; `git_hash` is not real provenance
- **Where:** `src/scoring/statistics.py:47-73` (two outcomes only) vs. `README.md:37`; `src/storage/schema.py:81` (`git_hash: Optional[str] = "HEAD"` hardcoded).
- **Impact:** documentation mismatch. A genuinely useful third state is being missed: when the McNemar test is *inconclusive* (p ≥ 0.05) but the point estimate improves, "FAIL" is too harsh and "PASS" is unjustified — `INCONCLUSIVE` is the honest answer, and it maps well onto a real release process that escalates ambiguous cases for human review. Separately, every artifact claims `git_hash: "HEAD"`, which is misleading provenance on a project whose selling point is reproducibility.
- **Fix:** add an `INCONCLUSIVE` branch keyed on `p >= alpha`; populate `git_hash` from a real `git rev-parse HEAD` when a repo is present, else `None`.

### MEDIUM

#### D8 — Dashboard radar silently drops the BYOB domain
- **Where:** `src/dashboard/app.py:117` hardcodes 5 categories; `src/benchmarks/base.py:67` sets BYOB's domain to `Custom Evaluation`.
- **Impact:** BYOB results are invisible in the radar. If a BYOB run is the *only* run for a model, its radar trace reads all-zeros.
- **Fix:** derive categories from the observed domains in the selected runs, or add `Custom Evaluation` and document the 5-domain framing as the default.

#### D9 — Dashboard evaluation runs block the UI thread
- **Where:** `src/dashboard/app.py:239-251` — `runner.run_evaluation` inside `st.spinner`.
- **Impact:** fine for mock models (~1 s). With live NIM at 45 s timeout per call, 20 items can freeze the browser for minutes with no progress indication and no cancellation.
- **Fix:** a background job with a progress placeholder polled via `st.fragment(run_every=...)`, or a simple job-submission + polling pattern.

#### D10 — Leaderboard SQL relies on SQLite-specific bare-column behaviour; `latest_run_id` is misnamed
- **Where:** `src/storage/run_store.py:82-89` — bare `ci_lower`, `ci_upper`, `run_id` alongside `MAX(overall_accuracy)`.
- **Impact:** correct on SQLite via the documented min/max special case, but silently wrong if the backend changes, and the alias `latest_run_id` actually returns the **best-accuracy** run, not the most recent — a genuine misreading risk on the dashboard.
- **Fix:** use a window function or a subquery selecting the row via `ORDER BY overall_accuracy DESC LIMIT 1`; rename the alias to `best_run_id`.

#### D11 — `question_id` collisions would silently corrupt McNemar pairing
- **Where:** `src/comparison/diff_engine.py:36-39` — maps keyed by `question_id` alone, with no benchmark prefix.
- **Impact:** safe for the shipped datasets (all IDs prefixed: `mmlu_`, `gsm_`, `math_`, `gpqa_`, `tqa_`, `byob_`, `HumanEval/`). **Unsafe** for a BYOB set that reuses another benchmark's IDs — items would be silently overwritten, `total_aligned_items` would under-report, and the McNemar test would be computed on misaligned pairs. Given BYOB is a headline feature, this is a real trap.
- **Fix:** key on `(benchmark, question_id)`, and reject or namespace duplicate IDs at load time.

#### D12 — `requirements.txt` and `pyproject.toml` dependencies disagree
- **Where:** `requirements.txt` (13 packages) vs. `pyproject.toml:11-23` (11).
- **Impact:** `pip install -r requirements.txt` and `pip install -e .` produce different environments. `requests` and `httpx` are in the first but not the second, while `requests` is a genuine runtime dependency (`src/gateway/model_client.py:3`).
- **Fix:** make `pyproject.toml` the single source and generate `requirements.txt` from it, or delete one.

#### D13 — Two declared dependencies are never imported
- **Where:** `tabulate` and `httpx` in `requirements.txt` (and `tabulate` in `pyproject.toml:19`).
- **Impact:** trivial supply-chain surface and confusion about what the project actually needs. Rich is used for all table rendering instead of `tabulate`.
- **Fix:** remove both.

### LOW

#### D14 — HTML report embeds Markdown in a `<pre>` block
- **Where:** `src/reporting/report_generator.py:87-89`.
- **Impact:** the HTML output is readable and printable but not a styled document — headings, tables and emphasis are not rendered. A stakeholder-facing report would look unfinished.
- **Fix:** add a Markdown renderer to the dependencies, or ship a hand-written HTML template with a real table.

#### D15 — No multiple-comparison correction
- **Where:** `src/scoring/statistics.py:41` — a single hardcoded α = 0.05.
- **Impact:** the project compares 6 models, implying up to 15 pairwise tests. At α = 0.05 uncorrected the family-wise error rate reaches ~26%, so roughly one in four "significant" pairwise claims would be a false positive. This is a genuine methodological gap a statistics-literate reviewer could raise.
- **Fix:** report Holm–Bonferroni adjusted p-values across the comparison family, or restrict headline claims to pre-registered pairwise comparisons (e.g. the baseline-vs-candidate pair the Quality Gate already defines) and say so explicitly.

#### D16 — HumanEval results are bimodal by construction
- **Where:** `src/gateway/mock_models.py:104-106` — the injected fault rewrites `==`→`!=` / `<`→`>`, or substitutes `return False`.
- **Evidence:** every HumanEval cell in §13.3 is exactly 0/10 or 10/10; never intermediate.
- **Impact:** HumanEval deltas are always ±100 pp, which visually dominates aggregate comparisons and inflates the 8B-vs-70B headline (§14.2). Real HumanEval pass@1 is continuous.
- **Fix:** inject subtler faults (off-by-one, wrong variable, a single inverted comparison in a longer function) so failures are partial. Failing that, **disclose the artefact explicitly** in any report showing HumanEval numbers.

#### D17 — Colab notebook depends on a remote that does not exist, and requests a GPU nothing uses
- **Where:** `notebooks/nemo_evaluator_colab.ipynb` cells 2 and 4, and `metadata.accelerator`.
- **Evidence:** (a) the working copy has **no `.git` directory**, so the `git clone https://github.com/ashish-vit/nemo-evaluator-benchmarking.git` referenced in `README.md:48` and notebook cell 4 cannot be assumed to resolve; (b) cell 2 installs from `https://github.com/NVIDIA-NeMo/Evaluator.git`, whose canonical path should be verified against NVIDIA's current repository naming; (c) the entire pipeline is pure-Python CPU work (HTTP calls, regex, `sqlite3`, `subprocess`), so `"accelerator": "GPU"` requests hardware nothing benefits from.
- **Impact:** the notebook is the project's "turnkey" claim (`REVIEW_2:30`). As written, a reviewer running it hits a clone failure or a wrong package URL at step 2 or 4.
- **Fix:** initialise the git repo and push to the stated remote; verify the NeMo Evaluator install URL; change the accelerator to `None` and reframe the notebook's benefit as "zero-cost, no-GPU-required" — the honest and stronger claim, and one that matches SDG 12.

#### D18 — README architecture diagram shows 6 boxes for 8 claimed layers
- **Where:** `README.md:78-92` vs. `README.md:24`.
- **Fix:** replace the ASCII block with the 8-layer diagram from §6 of this document.

#### D19 — Mojibake (`?`) throughout user-facing text
- **Where:** `README.md` lines 1, 12, 22, 44, 77; `src/dashboard/app.py:19,38,41-46,181,212,215,265`; `comparison.md:51`; `docs/REVIEW_*` ("2026?27").
- **Impact:** emoji were lost in a non-UTF-8 write. Visible in a submitted document and in the dashboard's tab labels and page icon.
- **Fix:** rewrite the affected files as UTF-8 with either proper emoji or plain ASCII labels.

#### D20 — `nel report` is advertised but does not exist
- **Where:** `README.md:39`; actual commands verified via `nel.py --help`.
- **Fix:** add a `report` command wrapping `ReportGenerator` over a run pair (~15 lines), or remove it from the README.

#### D21 — `LICENSE` referenced but absent; no `.gitignore`; stale bytecode
- **Where:** `README.md:6` links to `LICENSE`; the file does not exist. There is no `.gitignore`, and `__pycache__/*.cpython-314.pyc` files are present from a different Python minor version than the installed 3.13.9.
- **Fix:** add the MIT `LICENSE` text (matching the badge), add a Python `.gitignore`, delete the stale bytecode.

---

## 19. Roadmap & Recommended Next Steps

### 19.1 Priority 0 — before any further evaluation runs (≈1 day)

These change the meaning of the project's headline claims, so they come first.

| Step | Addresses | Effort |
|---|---|---|
| Replace `hash()` with a stable digest in `MockModelEngine`; wire `seed` through | D1 | ~30 min |
| Add a cross-subprocess determinism regression test | D1, §15 | ~30 min |
| Regenerate all 12 run artifacts and both comparison reports from the now-deterministic pipeline | D1, §13 | ~30 min |
| Correct `docs/REVIEW_2_FINAL_REPORT.md` to quote regenerated figures, and fix the "$0.0019/run" cost error | D1, §13.7 | ~20 min |
| Raise on an explicitly-specified but missing `--config`; warn on unknown model IDs | D5 | ~20 min |
| Add a loud warning when `provider != "mock"` but no API key is set; record the resolved provider in `execution_params` | D2 | ~30 min |
| Reword "safe sandbox" to "timeout-isolated subprocess" in README and Review 1 | D3 | ~10 min |

### 19.2 Priority 1 — correctness of claims (≈2 days)

| Step | Addresses | Effort |
|---|---|---|
| Add the `INCONCLUSIVE` gate outcome, keyed on McNemar p ≥ α; report Holm-adjusted p-values across comparison families | D7, D15 | ~1 hr |
| Thread `execution:` and `quality_gate:` config keys through to the engine and the gate; warn on unrecognised keys | D6 | ~2 hrs |
| Populate `git_hash` from a real `git rev-parse HEAD` when available | D7 | ~20 min |
| Key `DiffEngine` alignment on `(benchmark, question_id)`; reject duplicate IDs at load | D11 | ~40 min |
| Make the mock's HumanEval fault injection subtler so scores are continuous; disclose any residual artefact | D16 | ~45 min |
| Replace the leaderboard's bare-column SQL with a window function; rename `latest_run_id` → `best_run_id` | D10 | ~30 min |
| Add `LICENSE` (MIT, matching the badge), `.gitignore`; delete stale `__pycache__` | D21 | ~10 min |

### 19.3 Priority 2 — closing advertised gaps (≈3 days)

| Step | Addresses | Effort |
|---|---|---|
| Implement the `huggingface` provider branch (or remove it from the README) | D4 | ~2 hrs |
| Wire `LLMJudge` into TruthfulQA / BYOB behind a config flag | D4 | ~3 hrs |
| Wire `check_contamination` into dataset loading with a supplied corpus token set | D4 | ~2 hrs |
| Add the `nel report` command | D20 | ~30 min |
| Implement the `execution:` concurrency with a thread pool (HTTP-bound, so threads suffice) and retry with backoff | §17 rows 24-26 | ~3 hrs |
| Replace the 6-box README diagram with the 8-layer diagram | D18 | ~15 min |
| Fix mojibake across README, dashboard, and Review reports | D19 | ~30 min |
| Reconcile `requirements.txt` with `pyproject.toml`; drop `tabulate` and `httpx` | D12, D13 | ~15 min |
| Initialise git, push to the stated remote, verify the notebook clone and NeMo Evaluator install URLs; drop the GPU accelerator request | D17 | ~1 hr |
| Render real HTML in the report generator | D14 | ~1 hr |

### 19.4 Priority 3 — depth and scale (stretch)

| Step | Addresses | Rationale |
|---|---|---|
| Expand datasets toward statistically meaningful sizes (100+ items per benchmark) | §12 | At n=10 the CI spans ~50 pp; the platform's own statistics argue for this |
| Implement bootstrap confidence intervals as advertised | §17 row 10 | Completes the README claim; useful for non-proportion metrics |
| Replace the timeout-based sandbox with a container-based executor (Docker / gVisor) | D3 | Required before pointing the platform at live endpoints |
| Add background job execution + progress polling to the dashboard | D9 | Required for real-provider evaluation via the UI |
| Add a genuine rate limiter to `InterceptorProxy` | D4 | Completes the Review 1 claim; needed to respect NIM quotas |
| Add a `pytest` suite with parametrised scorer cases and the untested areas in §15 | §15 | Closes the largest coverage gaps cheaply |
| Add CI running the test suite on every push | §15 | Makes the "7/7 passing" claim self-maintaining |

### 19.5 Reporting guidance for the next review

When presenting results, the following framing is both honest and technically strong:

1. **Always present n and the Wilson CI next to every accuracy figure.** At n=10 the interval is ~50 pp wide; the platform computes it for exactly this reason, and showing it demonstrates methodological maturity rather than weakness.
2. **Present the Pareto frontier as the primary decision artefact**, not the raw leaderboard. "GPT-4o-Mini delivers 78.3% at 0.348 s and $0.0018/run; DeepSeek delivers 88.3% at 1.434 s and $0.0046/run" is the actual decision a stakeholder faces, and it is the clearest demonstration of the platform's value.
3. **Attribute the headline delta to its components** (as in §14.2) rather than quoting the aggregate, and disclose the HumanEval bimodality.
4. **Report the McNemar statistic with its effect size** (21 vs 3 discordant pairs), and note the exact-binomial alternative for small `b + c`.
5. **State the simulation status explicitly.** All 12 runs record `provider = "mock"`. The pipeline, statistics and comparison machinery are real and exercised; the model responses are synthetic. Presenting mock outputs as model findings would be the single most damaging thing a reviewer could discover independently.
6. **After fixing D1, present a multi-seed variance study** — the same 6 models across 3-5 seeds with mean ± spread — as direct evidence that the platform now quantifies its own uncertainty. That is a materially stronger contribution than a single point estimate.

---

## 20. Appendices

### 20.1 Complete CLI command reference

```powershell
# From the repository root, with the package NOT pip-installed:
python src/cli/nel.py <command> [options]

# --- List configured models, benchmarks, and the 10 most recent runs ---
python src/cli/nel.py list

# --- Run an evaluation ---
python src/cli/nel.py eval run                                  # all config defaults
python src/cli/nel.py eval run --config configs/eval_config_default.yaml --limit 10
python src/cli/nel.py eval run --model meta/llama-3.1-8b-instruct --limit 15
python src/cli/nel.py eval run --bench mmlu_pro --bench gsm8k --limit 10
python src/cli/nel.py eval run --solver cot --job-name cot_ablation

# --- Compare two runs ---
python src/cli/nel.py compare --run-a <RUN_A> --run-b <RUN_B>
python src/cli/nel.py compare --run-a <RUN_A> --run-b <RUN_B> --output-md comparison.md
python src/cli/nel.py compare --run-a <RUN_A> --run-b <RUN_B> --output-html comparison.html

# --- Release quality gate ---
python src/cli/nel.py gate --baseline <RUN_A> --candidate <RUN_B>
python src/cli/nel.py gate --baseline <RUN_A> --candidate <RUN_B> --min-acc 0.70 --max-reg 0.01

# --- Dashboard ---
streamlit run src/dashboard/app.py

# --- Tests ---
python -m unittest discover -s tests -v
```

**Solver values:** `simple` (direct answer) · `cot` (chain-of-thought) · `harbor` (agentic framing).

### 20.2 Quick-start from a clean clone

```powershell
git clone https://github.com/ashish-vit/nemo-evaluator-benchmarking.git
cd nemo-evaluator-benchmarking
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Verify the environment
python -m unittest discover -s tests -v      # expect: Ran 7 tests ... OK

# Optional: enable the bare `nel` command
pip install -e .

# Inspect the environment
python src/cli/nel.py list

# Run the full default job (6 models x 6 benchmarks x 15 items)
python src/cli/nel.py eval run --config configs/eval_config_default.yaml

# Compare the two most recent runs and export a report
python src/cli/nel.py compare --run-a run_20260916_194800_892c41 --run-b run_20260916_194736_474bc0 --output-md comparison.md
```

To evaluate against **live NVIDIA NIM** instead of the simulator: get a key from <https://build.nvidia.com>, set `$env:NVIDIA_API_KEY = "nvapi-..."`, and change the target model's `provider` from `mock` to `nvidia_nim` in `configs/models_config.yaml`. Then confirm the resolved provider in the run artifact's `execution_params` (D2).

### 20.3 Module-to-layer index

| Module | Lines | Layer | Primary class / function |
|---|---|---|---|
| `src/cli/nel.py` | 226 | 8 | Click group `cli`; `list_cmd`, `run_cmd`, `compare_cmd`, `gate_cmd` |
| `src/dashboard/app.py` | 273 | 8 | 5-tab Streamlit app |
| `src/engine/runner.py` | 157 | 5 | `EvaluationRunner.run_evaluation` |
| `src/engine/solvers.py` | 35 | 5 | `get_solver`, `SimpleChatSolver`, `ChainOfThoughtSolver`, `HarborAgenticSolver` |
| `src/engine/sandbox.py` | 43 | 5 | `CodeSandbox.execute_code` |
| `src/gateway/model_client.py` | 110 | 4b | `UnifiedModelClient.generate` |
| `src/gateway/interceptor.py` | 24 | 4b | `InterceptorProxy.get_cached` / `set_cached` |
| `src/gateway/mock_models.py` | 121 | 4b | `MockModelEngine.generate_response`, `MODEL_PROFILES` |
| `src/benchmarks/base.py` | 97 | 4a | `get_benchmark`, 7 benchmark classes |
| `src/benchmarks/loader.py` | 27 | 4a | `BenchmarkLoader.load_jsonl`, `check_contamination` |
| `src/scoring/rule_scorers.py` | 104 | 3 | `RuleScorers.extract_mcq_answer`, `score_mcq`, `extract_math_answer`, `score_math` |
| `src/scoring/statistics.py` | 73 | 3 | `StatisticsEngine.wilson_confidence_interval`, `mcnemar_test`, `evaluate_quality_gate` |
| `src/scoring/llm_judge.py` | 57 | 3 | `LLMJudge.evaluate_pairwise_debiased` (not wired) |
| `src/storage/schema.py` | 94 | 2 | `RunArtifact`, `ItemResult`, `BenchmarkSummary` |
| `src/storage/run_store.py` | 94 | 2 | `RunStore.save_run`, `get_run`, `list_runs`, `get_leaderboard` |
| `src/comparison/diff_engine.py` | 101 | 6 | `DiffEngine.compare_runs` |
| `src/reporting/report_generator.py` | 93 | 1 | `ReportGenerator.generate_markdown`, `generate_html` |

### 20.4 Glossary

| Term | Meaning in this project |
|---|---|
| **Run** | One complete evaluation of one model across a set of benchmarks, persisted as a JSON artifact plus a SQLite registry row |
| **Run ID** | `run_{UTC timestamp}_{6 hex}` — e.g. `run_20260916_194736_474bc0` |
| **Artifact** | The full JSON record of a run, including every prompt, response, extraction and metric |
| **Item** | One benchmark question and its graded outcome (`ItemResult`) |
| **Flip** | A benchmark question where exactly one of two compared models was correct |
| **Discordant pair** | A flip — the only data that carries information in a McNemar test |
| **Wilson interval** | A binomial confidence interval that stays valid at 0% and 100%, unlike the normal approximation |
| **McNemar test** | A paired significance test for the difference between two proportions on the same items |
| **Continuity correction** | The `(|b−c|−1)²` term in McNemar's statistic; improves the approximation at small `b+c` |
| **Pareto frontier** | The set of models not dominated on both accuracy and latency by another model |
| **Quality Gate** | The automated PASS/FAIL release decision comparing a candidate run to a baseline run |
| **BYOB** | "Bring Your Own Benchmark" — the custom-dataset ingestion path |
| **MCQ / CoT / NIM / MoE / SDG** | Multiple-choice question · Chain-of-Thought · NVIDIA Inference Microservice · Mixture-of-Experts · UN Sustainable Development Goal |
| **Mock / simulator** | The zero-cost built-in model backend used when `provider: "mock"` or no API key is present |

---

*End of document. Compiled 26 September 2026 from source code inspection, live execution of the test suite and evaluation pipeline, and direct query of the 12 persisted run artifacts in `runs/`.*

