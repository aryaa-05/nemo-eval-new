# NeMo Evaluator

A command-line and Streamlit platform for running benchmarks against language models, scoring replies, and comparing two runs statistically.

## Contents

- Benchmarks: `mmlu_pro`, `gsm8k`, `math_500`, `gpqa`, `humaneval`, `truthfulqa`, `custom_byob`, `kyc_india`, `rbi_regulations`, `image_processing`.
- Providers: NVIDIA NIM, Groq, OpenRouter, OpenAI, Ollama, and a built-in simulator.
- Scoring: multiple-choice, math, code execution, reasoning-text stripping, truncation and API-error flags.
- Statistics: per-model accuracy with 95% Wilson intervals, paired McNemar test, flips, and a PASS/FAIL quality gate.
- Interfaces: the `nel` CLI and a Streamlit dashboard.
- Source documents for the India-specific datasets are included in `_sources/`.

## Requirements

- Python 3.10 or newer.
- Git.
- API keys only for live providers.

## Setup

```powershell
git clone https://github.com/aryaa-05/nemo-eval-new.git
cd nemo-eval-new

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

Run commands from the repository root. Dataset and config paths are relative to the current directory.

## API keys

Create a `.env` file in the repository root and add only the keys you need:

```text
NVIDIA_API_KEY=<your key>
GROQ_API_KEY=<your key>
OPENROUTER_API_KEY=<your key>
OPENAI_API_KEY=<your key>
```

The CLI and dashboard load `.env` automatically. Never commit `.env` or any API key.

| Provider | Key variable | Notes |
|---|---|---|
| `mock` | none | Offline simulator; no API call |
| `nvidia_nim` | `NVIDIA_API_KEY` | Free tier |
| `groq` | `GROQ_API_KEY` | Free tier |
| `openrouter` | `OPENROUTER_API_KEY` | Paid; required by the real model routes |
| `openai` | `OPENAI_API_KEY` | Paid; not used by the real pair routes |
| `ollama` | none | Local server; no key |
| `auto` | route key | First configured route whose key is present |

## CLI guide

```powershell
# List models, benchmarks, and saved runs
python src/cli/nel.py list

# Quick simulated run
python src/cli/nel.py eval run --config configs/eval_config_default.yaml --limit 10

# Run a stand-in pair (real API calls; pipeline validation only)
python src/cli/nel.py eval run --config configs/standin_pair3.yaml --limit 5 --job-name standin_pair3_quick_check

# Compare two completed runs
python src/cli/nel.py compare --run-a <RUN_ID_A> --run-b <RUN_ID_B> --output-md comparison.md

# Apply the quality gate
python src/cli/nel.py gate --baseline <RUN_ID_A> --candidate <RUN_ID_B>
```

`eval run` options:

```text
--config <job.yaml>     Job file
--model <model-id>      Override the job's model list
--bench <benchmark>     Override the job's benchmarks; can be repeated
--limit N               First N rows per benchmark; 0 means all rows
--solver simple|cot|harbor
--job-name <name>
```

A model ID that is not in `configs/models_config.yaml` gets a simulated fallback. A mistyped `--config` path may silently fall back to defaults, so check the starting-job panel.

## Dashboard

```powershell
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501`. The dashboard provides:

- **Leaderboard**: accuracy, confidence interval, latency, tokens, cost, mode, job name, and warnings.
- **Radar & domains**: benchmark-domain comparison.
- **Model diff & flips**: questions where two runs disagree.
- **Launch evaluation**: background run with a progress poll every 2 seconds.
- **Reports**: Markdown/HTML comparison output.

Saved runs live in `runs/` and are git-ignored. A fresh clone starts with no saved results; copy the `runs/` folder locally if you need the existing leaderboard and report data.

## Models

The three required model pairs are configured with `provider: "auto"` and OpenRouter routes. Those exact models are not available on the free NIM/Groq tiers, and no real OpenRouter call has been verified yet. Several OpenRouter routes serve a substitute model; the run records both the requested and served IDs.

The free-tier stand-in pairs are:

| Pair | Model A | Model B |
|---|---|---|
| 1 | `google/gemma-3-12b-it` (NIM) | `meta/llama-3.2-11b-vision-instruct` (NIM) |
| 2 | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (NIM) | `meta/llama-3.2-11b-vision-instruct` (NIM) |
| 3 | `openai/gpt-oss-20b` (Groq) | `qwen/qwen3.8-27b` (Groq) |

Pair 1 cannot run: `google/gemma-3-12b-it` returns HTTP 404. Use pair 2 or pair 3 for pipeline checks.

A stand-in run makes real API calls and is therefore a **live** run, but its scores are results for the stand-in model only. They are **not** results for the required models. A simulated run uses `provider: "mock"` and makes no API call. A substitute route serves a different model ID than the one requested.

## Datasets

| Benchmark | Type | Items |
|---|---|---|
| `mmlu_pro` | 10-option MCQ | 15 |
| `gsm8k` | Free answer | 15 |
| `math_500` | Free answer | 15 |
| `gpqa` | 4-option MCQ | 15 |
| `humaneval` | Code | 15 |
| `truthfulqa` | 4-option MCQ | 15 |
| `custom_byob` | 4-option MCQ demo | 2 |
| `kyc_india` | 4-option MCQ | 100 |
| `rbi_regulations` | 4-option MCQ | 100 |
| `image_processing` | 4-option MCQ | 100 |

Each benchmark reads `datasets/<benchmark_id>/sample.jsonl`. `kyc_india`, `rbi_regulations`, and `image_processing` include `source` and `source_excerpt` fields. The source PDFs, text copies, and review worksheets are in `_sources/`.

Dataset rules:

- Every regulatory question must be written from source text in front of the author.
- Exactly one option must be correct.
- Answer positions must be balanced.
- Questions and keys must be unique.
- `review: true` marks a row that a person must check.
- Never invent a source section or answer key.

Validate a dataset:

```powershell
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --source-text _sources/kyc_master_direction.txt
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --final --source-text _sources/kyc_master_direction.txt
```

## Tests

```powershell
python -m pytest -q
```

The current suite has 402 tests.

## Documentation

- `CONTEXT.md`: full project context, rules, architecture, and known limitations.
- `docs/PROJECT_HANDOVER.md`: current status, decisions, blockers, and handover notes.
- `docs/PROJECT_DOCUMENTATION.md`, `docs/PRACTICAL_OVERVIEW.md`, and the review reports are older and may be out of date.
- `_sources/SOURCES.md`: source URLs and extraction notes.

## Reporting rules

- Never present simulated or stand-in results as results for the required models.
- Never present a substitute-model result as a result for the exact requested model.
- Never hand-write or guess benchmark scores.
- Never commit `.env`, API keys, or run files containing secrets.
- If something was not verified, say so.
