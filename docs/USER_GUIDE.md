# NeMo Evaluator Benchmarking Platform: User Guide

## 1. Project overview

NeMo Evaluator runs a fixed set of benchmark questions against language models, scores the replies automatically, and compares two runs statistically. It provides a CLI (`python src/cli/nel.py`) and a Streamlit dashboard.

The platform is text-only. The `image_processing` benchmark is multiple-choice questions about image-processing theory, not a vision-model test.

## 2. Configuration files

- `configs/models_config.yaml`: model IDs, providers, routes, parameters, and prices. This is the source of truth for model status.
- `configs/benchmarks_config.yaml`: benchmark names and descriptions. Its `sample_file` and `scorer` values are descriptive only.
- `configs/eval_config_default.yaml`: default job settings.
- `configs/eval_pair*.yaml`: the three required model-pair jobs. They use an OpenRouter key; all six routes were probed with one item, and full runs are pending.
- `configs/free_tier_pair*.yaml`: three pipeline-check jobs using LIVE and/or SIMULATED models.
- `configs/live_free_models.yaml`: 13 callable NIM/Groq models probed LIVE with one item each.

## 3. Run modes and labels

| Label | Meaning |
|---|---|
| `LIVE` | At least one real API call was made. Scores belong to the exact served model. |
| `SIMULATED` | The built-in `mock` provider answered; no API call was made. Scores are software tests only. |
| `SUBSTITUTE` | An `auto` route served a different model ID than requested. The run is still LIVE. |

“Free-tier” describes a zero-cost provider, not a model category or run mode. Never present a LIVE free-tier or SIMULATED run as a result for the required pair models.

## 4. API keys

Put keys in a repository-root `.env` file (git-ignored) or in environment variables. Never commit or print keys.

```powershell
$env:NVIDIA_API_KEY = "..."
$env:GROQ_API_KEY = "..."
```

Supported providers are `mock`, `nvidia_nim`, `groq`, `openrouter`, `openai`, `ollama`, and `auto`. `huggingface` is not implemented and raises an error. `auto` picks the first configured route whose key is present, in the order NIM, Groq, OpenRouter, OpenAI.

The required pair routes need `OPENROUTER_API_KEY`. On 20 November 2026, all six routes completed one-item LIVE probes; full benchmark runs are still pending, so probe scores are not results.

## 5. CLI

Run all commands from the repository root.

```powershell
python src/cli/nel.py list
python src/cli/nel.py eval run --config configs/eval_config_default.yaml --limit 10
python src/cli/nel.py eval run --config configs/live_free_models.yaml --limit 1
python src/cli/nel.py compare --run-a <RUN_ID_A> --run-b <RUN_ID_B> --output-md comparison.md
python src/cli/nel.py gate --baseline <RUN_ID_A> --candidate <RUN_ID_B>
python src/tools/rescore_run.py <RUN_ID>
python -m pytest -q
```

`eval run` options are `--config`, `--model`, repeatable `--bench`, `--limit` (`0` means all rows), `--solver simple|cot|harbor`, and `--job-name`. Explicit flags override the job file. A mistyped config path falls back to defaults silently, so check the printed “Starting job” panel.

A model ID not found in `configs/models_config.yaml` gets a simulated fallback. `compare` and `gate` refuse failed or incomplete runs.

## 6. Dashboard

```powershell
python -m streamlit run src/dashboard/app.py
```

Tabs:

- **Leaderboard**: accuracy, Wilson interval, latency, tokens, cost, mode, job name, and warnings.
- **Radar & domains**: per-domain accuracy from the selected runs; new domains appear automatically.
- **Model diff & flips**: paired McNemar test and item-level flips, matched by `question_id`.
- **Launch evaluation**: starts a background job and polls status every 2 seconds. Model-set choices are **Live free-tier models** and **All configured models**.
- **Reports**: Markdown/HTML comparison output.

The job status is held in the Streamlit process only; a restart loses the status panel but not the saved run file.

## 7. Datasets

Each benchmark reads `datasets/<benchmark_id>/sample.jsonl`.

| Benchmark | Type | Rows |
|---|---|---|
| `mmlu_pro` | 10-option MCQ | 15 |
| `gsm8k`, `math_500`, `gpqa`, `humaneval`, `truthfulqa` | math / code / 4-option MCQ | 15 each |
| `custom_byob` | 4-option MCQ demo | 2 |
| `kyc_india`, `rbi_regulations`, `image_processing` | 4-option sourced MCQ | 100 each |

The three 100-row datasets include `source` and `source_excerpt`. Their answer keys are written from source text but most were not independently human-checked. Validate them with `src/tools/validate_dataset.py`.

## 8. Reporting rules

- State whether each run is LIVE or SIMULATED and which provider/model served it.
- State when a served model is a substitute for the requested one.
- Never report simulated or free-tier results as required-pair results.
- Never invent scores; scores come only from executed runs.
- State any failure or unverified item next to the numbers.
