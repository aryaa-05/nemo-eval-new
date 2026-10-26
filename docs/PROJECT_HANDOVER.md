# NeMo Evaluator - Project Handover and Current Status

**Last updated:** 26 October 2026
**Audience:** A new team member, reviewer, or AI assistant picking up the project.
**Repository:** `C:\Users\Aryaa\Documents\Projects\In Progress\nemo-eval-new`  
**Remote:** `https://github.com/aryaa-05/nemo-eval-new.git`
**Primary source of truth:** the code, then `CONTEXT.md`, then this handover. The older files in `docs/` predate the latest work and contain outdated statements.

This document explains:

- what the project is trying to prove;
- which models were required, which models are actually being used, and why;
- what data has been built and how it was checked;
- how the pipeline works and how to run it;
- the current live smoke results;
- what is still missing or unverified;
- how a new person should start.

---

## 1. Read this first: rules that must not be broken

1. **Never present simulated or stand-in results as results for the required models.**
   - A run is **simulated** only when `provider: mock` and no API call is made.
   - A **stand-in** run does make real API calls. It is a **LIVE** run, and its numbers are real results **for the stand-in model only**. They are not results for the models named in the brief.
   - A **substitute route** is a live route where the provider serves a different model ID than the one requested. The run records both IDs and `model_substituted: true`. Its scores are results for the **served** model, not the exact requested model.
2. **Never write regulatory or factual benchmark content from memory.** Every question must come from source text the author can read, and must record the document and section/paragraph.
3. **Never invent or hand-write benchmark scores.** Scores come from running the code.
4. **Never commit or print an API key.** Keys live in `.env` (git-ignored) or environment variables.
5. **If something is unverified, say so.** Especially dataset keys, model availability, and OpenRouter routes.
6. **Report the sample size and reliability flags with any score.** A 2-item or 3-item result is a smoke check, not a model ranking.

---

## 2. Executive summary

The platform is built and working:

- It gives models the same benchmark questions, extracts answers, scores them, stores every item, and compares two models with a paired McNemar test.
- It reports accuracy with 95% Wilson confidence intervals, latency, tokens, cost, flips, and reliability flags.
- It has a CLI (`python src/cli/nel.py`) and a Streamlit dashboard.
- It supports NVIDIA NIM, Groq, OpenRouter, OpenAI, Ollama, a deterministic simulator, and `provider: "auto"` route selection.

The data is now in its intended 100-question shape:

- `kyc_india`: 100 items, validated.
- `rbi_regulations`: 100 items, validated.
- `image_processing`: 100 items, validated.
- All three have balanced answer positions (25 of each letter), source citations, and source excerpts that match the supplied source text.

The central blocker is model availability:

- The six models named in the brief cannot be called on the free NVIDIA NIM or Groq tiers.
- They are configured with `provider: "auto"` and an OpenRouter route, but there is currently **no `OPENROUTER_API_KEY`**, and **no real OpenRouter call has ever been made**.
- The project therefore runs free-tier **stand-ins** to validate the live pipeline.
- Pair 1 cannot currently run at all because its stand-in model A, `google/gemma-3-12b-it`, returned HTTP 404.

**There is still no result for any required model.** The results in section 9 are live stand-in smoke checks, not the real comparison.

---

## 3. What the project is

This is a university capstone project, with an industry use case of choosing between LLMs for enterprise work. The focus includes India-specific banking/KYC compliance questions.

The platform:

1. Loads a fixed benchmark question set.
2. Sends each question to a model with the same prompt settings.
3. Extracts and scores the answer automatically.
4. Stores the full prompt, reply, extracted answer, tokens, cost, latency, and flags.
5. Reports per-benchmark and overall accuracy with a 95% confidence interval.
6. Compares two runs question by question, finds cases where one model is right and the other is wrong ("flips"), and runs a paired McNemar test.
7. Applies a quality gate (PASS/FAIL) for release decisions.
8. Shows results in a CLI and a Streamlit dashboard.

Interfaces:

```text
python src/cli/nel.py list
python src/cli/nel.py eval run ...
python src/cli/nel.py compare ...
python src/cli/nel.py gate ...
streamlit run src/dashboard/app.py
```

There is **no `nel report` command**. Reports are produced by `compare --output-md ...` or `compare --output-html ...`.

---

## 4. Required models vs models actually used

### 4.1 The three required pairs

These are the model pairs exactly as named in the brief. They use Hugging Face-style IDs.

| Pair | Required model A | Required model B | Why it cannot simply run now |
|---|---|---|---|
| 1 | `mistralai/Ministral-3-8B-Instruct-2512` | `meta-llama/Llama-3.1-8B-Instruct` | Ministral is absent from NIM/Groq; Llama-3.1-8B is retired on NIM (HTTP 410) and absent from Groq. |
| 2 | `mistralai/Ministral-3-14B-Reasoning-2512` | `meta-llama/Llama-4-Scout-17B-16E` | Both are absent from NIM/Groq. |
| 3 | `mistralai/Ministral-3-8B-Reasoning-2512` | `Qwen3.5` (size not specified) | Both are absent from NIM/Groq. The brief does not state which Qwen3.5 size is meant. |

Important naming details:

- The repository uses NIM-style lower-case IDs, for example `meta/llama-3.1-8b-instruct`, not the Hugging Face spelling.
- Pair 2's second model is configured with an `-instruct` suffix: `meta/llama-4-scout-17b-16e-instruct`. The brief does not have that suffix.
- Pair 3's second model is configured as `qwen/qwen3.5-8b`. The "8B" is an assumption that came from the old configuration, not from the brief.
- The brief also asks for vision models, but gives no models, images, or tasks. Nothing vision-specific is implemented.

### 4.2 Repository model IDs corresponding to the brief

| Pair | Brief model | Repository model ID |
|---|---|---|
| 1 | `mistralai/Ministral-3-8B-Instruct-2512` | `mistralai/ministral-3-8b-instruct-2512` |
| 1 | `meta-llama/Llama-3.1-8B-Instruct` | `meta/llama-3.1-8b-instruct` |
| 2 | `mistralai/Ministral-3-14B-Reasoning-2512` | `mistralai/ministral-3-14b-reasoning-2512` |
| 2 | `meta-llama/Llama-4-Scout-17B-16E` | `meta/llama-4-scout-17b-16e-instruct` |
| 3 | `mistralai/Ministral-3-8B-Reasoning-2512` | `mistralai/ministral-3-8b-reasoning-2512` |
| 3 | `Qwen3.5` | `qwen/qwen3.5-8b` (8B assumption) |

### 4.3 OpenRouter routes and substitutes

The six required models are configured with `provider: "auto"` and an OpenRouter route. Auto tries routes in this order: NIM, Groq, OpenRouter, OpenAI. For these six models, only the OpenRouter route is configured, so `OPENROUTER_API_KEY` is the key that would make them live.

OpenRouter's model list was checked in October 2026. It has no exact Ministral-3 variant and no Qwen3.5-8B. The configured substitutions are:

| Requested model | OpenRouter served model | Substitute? | Note |
|---|---|---|---|
| `mistralai/ministral-3-8b-instruct-2512` | `mistralai/ministral-8b-2512` | Yes | OpenRouter does not distinguish Instruct vs Reasoning. |
| `mistralai/ministral-3-8b-reasoning-2512` | `mistralai/ministral-8b-2512` | Yes | Same served model as pair 1 model A. |
| `mistralai/ministral-3-14b-reasoning-2512` | `mistralai/ministral-14b-2512` | Yes | OpenRouter does not confirm the Reasoning variant. |
| `qwen/qwen3.5-8b` | `qwen/qwen3.5-9b` | Yes | 9B stands in for the unspecified 8B. |
| `meta/llama-3.1-8b-instruct` | `meta-llama/llama-3.1-8b-instruct` | No | Exact model. |
| `meta/llama-4-scout-17b-16e-instruct` | `meta-llama/llama-4-scout` | No | Exact family/size. |

OpenRouter costs in `configs/models_config.yaml` are per 1M tokens at the time of checking:

| Requested model | Input USD / 1M | Output USD / 1M | `max_tokens` | Timeout |
|---|---:|---:|---:|---:|
| Ministral-3-8B-Instruct | 0.15 | 0.15 | 1024 | default 120 s |
| Ministral-3-8B-Reasoning | 0.15 | 0.15 | 4096 | 240 s |
| Ministral-3-14B-Reasoning | 0.20 | 0.20 | 4096 | 240 s |
| Llama-3.1-8B-Instruct | 0.05 | 0.08 | 1024 | default 120 s |
| Llama-4-Scout-17B-16E | 0.10 | 0.30 | 1024 | default 120 s |
| Qwen3.5-9B substitute | 0.17 | 0.25 | 4096 | 240 s |

**No real OpenRouter call has been made.** The model list is not proof that a route answers. Whether each route works, which upstream provider OpenRouter chooses, and whether a separate reasoning field arrives are all unverified.

### 4.4 Free-tier stand-in models actually used

Because the required models are not available, the project uses cheaper look-alike models that mirror the roles of the required pairs. These make real API calls and are live runs, but they are **pipeline validation only**.

| Pair | Role mirrored | Stand-in model A | Provider | Stand-in model B | Provider | Status |
|---|---|---|---|---|---|---|
| 1 | Small instruct vs Llama ~8B | `google/gemma-3-12b-it` | NVIDIA NIM | `meta/llama-3.2-11b-vision-instruct` | NVIDIA NIM | **NOT RUNNABLE**: model A returns HTTP 404. |
| 2 | Mid-size reasoning vs nearest Llama to Scout | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | NVIDIA NIM | `meta/llama-3.2-11b-vision-instruct` | NVIDIA NIM | Both have worked. |
| 3 | Reasoning vs Qwen | `openai/gpt-oss-20b` | Groq | `qwen/qwen3.8-27b` | Groq | Both have worked. |

Notes:

- `meta/llama-3.2-11b-vision-instruct` is a vision-language model, but the pipeline sends text only. No images are used.
- `qwen/qwen3.8-27b` is a different version and size from the Qwen3.5 in the brief.
- Pair 1 has no replacement for model A yet. Every small instruct model tried on NIM returned 404.
- The stand-in pair configurations are `configs/standin_pair1.yaml`, `standin_pair2.yaml`, and `standin_pair3.yaml`.

### 4.5 Why the stand-ins had to be chosen this way

Model availability was checked with real calls, not just model lists.

NVIDIA NIM:

- The list returned 81 models.
- It contained no Llama 3.1 8B/70B, no Llama 4, no Qwen, and no Ministral.
- `meta/llama-3.1-8b-instruct` returned HTTP 410 and said it reached end of life on 2026-08-26.
- Of 11 NIM models tried, 2 worked, 8 returned HTTP 404, and 1 timed out.
- Models that were listed but returned 404 included `mistralai/mistral-7b-instruct-v0.3`, `nv-mistralai/mistral-nemo-12b-instruct`, `nvidia/mistral-nemo-minitron-8b-8k-instruct`, `ibm/granite-3.0-8b-instruct`, `nvidia/llama-3.1-nemotron-70b-instruct`, `google/gemma-3-12b-it`, `zyphra/zamba2-7b-instruct`, and `aisingapore/sea-lion-7b-instruct`.
- `meta/llama-3.2-90b-vision-instruct` was listed but timed out once at 120 s. It is unverified, not proven dead.
- `meta/llama-3.2-11b-vision-instruct` worked in a 10-item MMLU-Pro run, with some slow calls.
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` worked on a math question and returned thinking in `message.reasoning_content`.

Groq:

- The list returned 11 models.
- Chat-capable models included `allam-2-7b`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b`, and `qwen/qwen3.8-27b`.
- There was no Llama chat model and no Mistral model.
- `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` both worked.
- `openai/gpt-oss-20b` returns thinking in `message.reasoning`; its reasoning tokens are included in `completion_tokens`.
- `qwen/qwen3.8-27b` returned only `content`, despite Groq marking it reasoning-capable.

Other configured models:

- The older models (`meta/llama-3.1-70b-instruct`, `qwen/qwen-2.5-72b-instruct`, `mistralai/mixtral-8x7b-instruct`, `deepseek/deepseek-r1-distill-llama-70b`, `openai/gpt-4o-mini`) are still `provider: mock`. They are not part of the required pairs and their simulated numbers mean nothing about the real models.
- `huggingface` is not implemented as a provider. A self-hosted vLLM/OpenAI-compatible server is not supported by the existing `openai` provider, because that provider only accepts `https://api.openai.com`.

### 4.6 Current key availability

Checked on 26 October 2026 without printing values:

- `.env` contains `NVIDIA_API_KEY`.
- `.env` contains `GROQ_API_KEY`.
- `OPENROUTER_API_KEY` is **not** present.
- `OPENAI_API_KEY` is not required for the current free-tier work.

To check key names safely from the repo root:

```powershell
python -c "import os; from src.gateway.env_loader import load_env_file; load_env_file(); print([k for k in ['NVIDIA_API_KEY','GROQ_API_KEY','OPENROUTER_API_KEY','OPENAI_API_KEY'] if os.environ.get(k)])"
```

This prints only variable names, never values.

### 4.7 Which models the current smoke run used

The current live smoke run used only the four callable stand-ins:

- NIM: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`
- NIM: `meta/llama-3.2-11b-vision-instruct`
- Groq: `openai/gpt-oss-20b`
- Groq: `qwen/qwen3.8-27b`

It did **not** use any required model, any OpenRouter model, or Gemma.

---

## 5. Benchmarks and datasets

### 5.1 All benchmark files

Every benchmark reads exactly one file, `datasets/<benchmark_id>/sample.jsonl`, except HumanEval which uses its own schema fields. Paths are relative to the current directory, so run from the repository root.

Current row counts:

| Benchmark | Rows | Type | Answer/scoring |
|---|---:|---|---|
| `mmlu_pro` | 15 | Multiple choice, 10 options | One letter A-J |
| `gsm8k` | 15 | Free answer | Math answer extraction |
| `math_500` | 15 | Free answer | Math answer extraction |
| `gpqa` | 15 | Multiple choice, 4 options | One letter A-D |
| `humaneval` | 15 | Code | Runs generated Python against tests |
| `truthfulqa` | 15 | Multiple choice, 4 options | One letter A-D |
| `custom_byob` | 2 | Multiple choice, 4 options | One letter A-D; demo only |
| `kyc_india` | 100 | Multiple choice, 4 options | One letter A-D |
| `rbi_regulations` | 100 | Multiple choice, 4 options | One letter A-D |
| `image_processing` | 100 | Multiple choice, 4 options | One letter A-D |

The three pair job files use nine benchmarks: the three new datasets plus `mmlu_pro`, `gsm8k`, `math_500`, `gpqa`, `humaneval`, and `truthfulqa`. `custom_byob` is not in the pair jobs.

### 5.2 The three new datasets

#### `kyc_india`

- 100 four-option questions.
- Source: `Master Direction - Know Your Customer (KYC) Direction, 2016`, DBR.AML.BC.No.81/14.01.001/2015-16, updated 14 August 2025.
- Local text: `_sources/kyc_master_direction.txt`.
- Current validation:
  - 100 items.
  - A=25, B=25, C=25, D=25.
  - 22/100 `human_checked: true`.
  - 0/100 `review: true`.
  - 100/100 source excerpts found in the source text.
  - `--final`: PASS with 0 errors, 0 warnings.

#### `rbi_regulations`

- 100 four-option questions.
- Sources:
  - `Master Circular - Basel III Capital Regulations`, RBI/2025-26/08, DOR.CAP.REC.2/21.06.201/2025-26, 1 April 2025.
  - `Master Circular - Prudential norms on Income Recognition, Asset Classification and Provisioning pertaining to Advances`, RBI/2025-26/13, DOR.STR.REC.9/21.04.048/2025-26, 1 April 2025.
- Local texts: `_sources/basel_iii_master_circular.txt`, `_sources/irac_advances_master_circular.txt`.
- Current validation:
  - 100 items.
  - A=25, B=25, C=25, D=25.
  - 22/100 `human_checked: true`.
  - 3/100 `review: true` (these still need a human decision).
  - 100/100 source excerpts found in the source texts.
  - `--final`: PASS with 0 errors, 0 warnings.

#### `image_processing`

- 100 four-option questions.
- Source: `Fundamentals of Image Processing`, Young, Gerbrands and van Vliet, Version 2.3 (2007).
- Local text: `_sources/young_fundamentals_image_processing.txt`.
- Current validation:
  - 100 items.
  - A=25, B=25, C=25, D=25.
  - 20/100 `human_checked: true`.
  - 0/100 `review: true`.
  - 100/100 source excerpts found in the source text.
  - `--final`: PASS with 0 errors, 0 warnings.
- This is a **text multiple-choice benchmark about image-processing theory**. It is not a test of a vision model. The Llama vision stand-in was run text-only.
- Human review details:
  - A worksheet of 20 rows is at `_sources/image_human_check_worksheet.md`.
  - All 9 original `review: true` items were included.
  - Weak excerpts were strengthened for `ip_020`, `ip_067`, `ip_078`, and `ip_082`.
  - `ip_044` failed review because more than one option was supported; it was replaced with an unambiguous cutoff-frequency question.
  - `ip_078` cited page 62 but the source passage is on page 65; the page reference was corrected.
  - The 20 reviewed rows were accepted and marked `human_checked: true`.

### 5.3 Human checking and review flags

- `review: true` means "a person still needs to check this".
- `human_checked: true` means a person actually reviewed it and accepted it.
- Do not treat the two fields as the same.
- KYC has 22/100 human-checked, 0 still flagged.
- RBI has 22/100 human-checked, 3 still flagged.
- Image has 20/100 human-checked, 0 still flagged.
- Most keys across the three new datasets were written from the source but were not independently checked. Do not describe every key as verified.
- Spot-checking the first three KYC and RBI items against their excerpts showed the keys are consistent with the source. That is a spot check, not full verification.

### 5.4 Older dataset changes

- `gpqa` and `truthfulqa` were rebalanced from all answers being A to `A=4 B=3 C=4 D=4`.
- Their questions and keys are still unverified. Both have only 15 items.
- The rebalance tool is `src/tools/rebalance_answers.py`; it never edits in place. It writes a copy.
- `gsm_05` was ambiguous and was reworded. The new wording is:

  > Baker bakes 120 loaves and sells 3/4 of them by noon. She bakes 60 more loaves and sells half of those 60 new loaves. How many loaves are left?

  The key remains `60`.
- Old run files that used the old `gsm_05` wording keep their old verdicts. Do not reuse their GSM8K numbers.

### 5.5 Dataset validation commands

```powershell
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --final --source-text "_sources/kyc_master_direction.txt"

python src/tools/validate_dataset.py datasets/rbi_regulations/sample.jsonl --final --source-text "_sources/basel_iii_master_circular.txt" --source-text "_sources/irac_advances_master_circular.txt"

python src/tools/validate_dataset.py datasets/image_processing/sample.jsonl --final --source-text "_sources/young_fundamentals_image_processing.txt"
```

The validator checks format, required fields, answer balance, duplicate IDs/questions, source and excerpt rules, and whether excerpts appear in the supplied source text. It **cannot** tell whether a question is true or whether the answer key is correct.

### 5.6 Dataset rules for future edits

- 100 items for each of the three new datasets.
- Exactly four options for A-D benchmarks (ten for `mmlu_pro`).
- One correct answer only.
- No "all of the above", "none of the above", or "both A and B" options.
- Options must not contain their own "A." labels.
- `question_id` must be unique across all datasets, because comparison matches by `question_id` alone.
- Source must name the document and section/paragraph. Generic values such as `RBI`, `KYC`, or `RBI KYC` are rejected.
- `source_excerpt` must be 20-600 characters and copied from the source.
- Answer positions should be about 25 of each letter.
- `review: true` for anything uncertain.
- Write and validate in batches of 25.
- JSONL: one JSON object per line, UTF-8, no BOM, no trailing commas.
- Do not edit a dataset file in place with the rebalance script; it writes a copy.

### 5.7 Source files

The source texts used by the validator are in the tracked `_sources/` folder:

| File | Document | Used for |
|---|---|---|
| `_sources/kyc_master_direction.txt` | RBI KYC Master Direction, updated 14 August 2025 | `kyc_india` |
| `_sources/basel_iii_master_circular.txt` | RBI Basel III Master Circular, 1 April 2025 | `rbi_regulations` |
| `_sources/irac_advances_master_circular.txt` | RBI IRAC/advances Master Circular, 1 April 2025 | `rbi_regulations` |
| `_sources/young_fundamentals_image_processing.txt` | Young, Gerbrands, van Vliet, Fundamentals of Image Processing, 2007 | `image_processing` |
| `_sources/SOURCES.md` | Source URLs, extraction notes, validation commands | Reference |
| `_sources/image_human_check_worksheet.md` | 20-row image review worksheet | Human review |
| `_sources/answer_plan.json` | Pre-decided answer letters for balance | Dataset construction |

The source PDFs and text copies are tracked so dataset validation can be reproduced. The two `_sources/_backup_*.jsonl` files are older backups and remain local only. Never commit `.env` or API keys.

---

## 6. How the pipeline works

### 6.1 Main files

| Path | Purpose |
|---|---|
| `configs/models_config.yaml` | Model definitions, providers, routes, costs, limits |
| `configs/benchmarks_config.yaml` | Names and display metadata for benchmarks; its `sample_file` values are not read by code |
| `configs/eval_pair*.yaml` | Real pair jobs |
| `configs/standin_pair*.yaml` | Free-tier stand-in jobs |
| `src/cli/nel.py` | CLI: `list`, `eval run`, `compare`, `gate` |
| `src/engine/runner.py` | Main evaluation loop |
| `src/engine/solvers.py` | Builds prompts (`simple`, `cot`, `harbor`) |
| `src/engine/answer_scoring.py` | Scores one reply and flags truncation/no-answer/API errors |
| `src/engine/sandbox.py` | Runs HumanEval code in a separate process with a 3 s timeout |
| `src/gateway/model_client.py` | Provider selection, auto routes, key/host checks, retries, throttling |
| `src/gateway/mock_models.py` | Deterministic simulator |
| `src/gateway/env_loader.py` | Loads `.env` without printing values |
| `src/gateway/interceptor.py` | In-memory response cache |
| `src/benchmarks/base.py` | Benchmark classes and `get_benchmark()`; hardcodes dataset paths |
| `src/benchmarks/loader.py` | JSONL loader |
| `src/scoring/rule_scorers.py` | Multiple-choice extraction |
| `src/scoring/math_answers.py` | Math extraction, normalisation, comparison |
| `src/scoring/statistics.py` | Wilson interval, McNemar test, quality gate |
| `src/scoring/version.py` | `SCORER_VERSION` and changelog |
| `src/storage/schema.py` | Run artifact schema, reliability thresholds |
| `src/storage/run_store.py` | JSON run files and SQLite registry |
| `src/comparison/diff_engine.py` | Aligns two runs and finds flips |
| `src/reporting/report_generator.py` | Markdown/HTML reports |
| `src/dashboard/app.py` | Streamlit UI |
| `src/dashboard/jobs.py` | Background evaluation thread and job status |
| `src/tools/validate_dataset.py` | Dataset validator |
| `src/tools/rebalance_answers.py` | Answer-position rebalancer |
| `src/tools/rescore_run.py` | Offline re-scoring of stored replies |
| `tests/` | Test suite |
| `runs/` | Saved run JSON files and `runs_registry.db` |
| `_sources/` | Source PDFs, text copies, review worksheets |

### 6.2 One run, step by step

1. The CLI loads the job YAML and `configs/models_config.yaml`.
2. `--model`, `--bench`, and `--limit` override the job file.
3. A model ID not found in the config gets a made-up simulated config. A typo therefore produces a SIMULATED run, not an error.
4. The runner creates a `UnifiedModelClient`.
   - For a live provider, it raises if the key is missing, the provider is unsupported, or the endpoint is not the provider's own host.
   - For `auto`, it chooses the first route whose key is present: NIM, Groq, OpenRouter, OpenAI. It never falls back to the simulator.
5. For each benchmark, the loader reads `datasets/<id>/sample.jsonl` and takes the first `--limit` rows; `0` means all rows.
6. For each row, the solver builds the prompt, the client gets a reply, and the scorer extracts and scores the answer.
7. Each `ItemResult` stores the prompt, raw reply, reasoning text, scored text, extracted answer, correctness, tokens, cost, latency, flags, HTTP attempts, and attempt log.
8. After every finished benchmark except the last, the partial run is saved as `status: "incomplete"`. When the model finishes, the same file is overwritten as `completed` and registered.
9. `compare` loads two runs, matches by `question_id`, runs McNemar, and can write Markdown/HTML.
10. `gate` applies min accuracy and max regression. It returns PASS or FAIL.

### 6.3 Providers and hosts

| Provider | Key variable | Host | Notes |
|---|---|---|---|
| `mock` | none | none | Simulator; no API call |
| `nvidia_nim` | `NVIDIA_API_KEY` | `https://integrate.api.nvidia.com/v1` | Free tier, slow |
| `groq` | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` | Free tier, fast |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | Paid; never called yet |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` | Paid; not used by real pair routes |
| `ollama` | none | `api_base`, default `http://localhost:11434` | Local models |
| `auto` | each route's key | chosen route's host | First key present |
| `huggingface` | - | - | Not implemented; raises an error |

Security properties:

- A key is only sent to its own provider's host.
- HTTPS only for hosted providers.
- No redirects.
- No look-alike host or `user@host` tricks.
- A model cannot read another provider's key variable.
- Keys are never printed in error messages.
- A test sweeps configured models against providers, keys, and endpoints.

### 6.4 Retries, throttling, and early stop

- HTTP 429 or 5xx: retried up to 4 times, with waits of 2, 4, 8, 16 seconds, or a longer `Retry-After` up to 120 seconds.
- Read timeout: retried once after 2 seconds, on a separate budget.
- Other 4xx: never retried.
- Maximum requests per item: 1 + 4 + 1 = 6.
- Each model is throttled to at least `60 / requests_per_minute` seconds between requests. Default is 20 rpm.
- A failed item is scored incorrect and flagged; it is not skipped.
- Early stop: if the first 5 items of a model all fail with the same one of 401, 403, 404, or 410, the model is aborted and the run is marked FAILED. Other statuses, a mix, a success, or fewer than 5 items do not trigger it.

### 6.5 Reliability flags

- A run is UNRELIABLE if more than 10% of its items are `api_error`, `truncated`, or `no_answer_found`. Exactly 10% is fine.
- A benchmark is UNRELIABLE if more than 20% of its items are flagged. Exactly 20% is fine.
- The per-benchmark flag exists because failures can cluster in one dataset while the run stays under 10%.
- An UNRELIABLE score is not a fair measure of the model. Find the cause and rerun.

### 6.6 Scoring summary

- API errors, truncation, and no-extractable-answer are counted incorrect.
- Truncation is detected from `finish_reason` values `length`, `max_tokens`, or `max_output_tokens`.
- Reasoning blocks (`<think>`, `<thinking>`, `<reasoning>`) are stripped before extraction.
- Separate reasoning fields are stored as `reasoning_text` and never scored.
- Multiple choice: the last explicit marker wins (`Answer: C`, `The answer is C`, `\boxed{C}`, bold `C`, etc.).
- Math: extracts the last `\boxed{}`, then `####`, then labelled answers, then final sentences. It does not guess between two different values. Units are not converted.
- HumanEval: extracts Python and runs it against tests in a separate process with a 3 s timeout. This is a timeout, not a security sandbox.
- `SCORER_VERSION` is stored in every run. Current version is 2.
- Run files before the version field load as `unversioned`; do not compare their math scores with v2 runs.
- `src/tools/rescore_run.py <RUN_ID>` re-scores stored replies offline into a new file; it never modifies the original. HumanEval items are kept as stored.

### 6.7 Statistics and comparison

- 95% Wilson confidence interval for accuracy.
- Paired McNemar test with continuity correction.
- Flips: questions where A is correct and B wrong, or vice versa.
- No correction for multiple comparisons.
- With 2-15 items per benchmark, confidence intervals are very wide and p-values are unstable.
- Quality gate has only PASS and FAIL.

### 6.8 Run lifecycle and labels

Run states:

- `completed`
- `failed` (early stop)
- `incomplete` (crash/interruption; JSON only, not in leaderboard)

Labels:

- `SIMULATED`: provider `mock`.
- `LIVE`: any network-backed run, including stand-ins.
- `STAND-IN`: job name starts with `standin_`.
- `SUBSTITUTE`: auto route served a different model.
- `UNRELIABLE`: more than 10% of the run flagged; or per-benchmark more than 20%.
- `FAILED`: early stop.
- `INCOMPLETE`: partial run; `compare` and `gate` refuse it.

`compare` and `gate` refuse anything that is not `completed`.

---

## 7. Current run results (stand-in only)

These are **live stand-in smoke checks**, not full benchmark runs and not results for the required models.

### 7.1 What was run

| Pair | Job name | Rows per benchmark | Items per model | Models |
|---|---|---:|---:|---|
| 3 | `standin_pair3_current_situation` | 3 | 27 | gpt-oss-20b vs Qwen3.8-27B |
| 2 | `standin_pair2_current_situation` | 2 | 18 | Nemotron vs Llama-3.2-11B |

Run IDs:

- Pair 3 A: `run_20261026_084909_38826d`
- Pair 3 B: `run_20261026_085214_f87b53`
- Pair 2 A: `run_20261026_085713_000c7c`
- Pair 2 B: `run_20261026_090913_9cdb1e`

### 7.2 Overall results

| Pair | Model A | Model B | Overall accuracy | McNemar | Flips | Reliability |
|---|---|---|---|---|---|---|
| 3 | gpt-oss-20b: **77.8%** | Qwen3.8-27B: **88.9%** | +11.1 points for B | p=0.248, not significant | B wins 3, A wins 0 | A run ok (7%); B run ok (0%) |
| 2 | Nemotron: **83.3%** | Llama-3.2-11B: **61.1%** | -22.2 points for B | p=0.134, not significant | A wins 4, B wins 0 | Both runs ok (6%) |

Latency:

- Pair 3: B was about 0.99 s faster on average.
- Pair 2: B was about 31.31 s faster on average, but A's mean was inflated by a 240 s timeout.

### 7.3 Per-benchmark results

Pair 3, 3 items each:

| Benchmark | gpt-oss-20b | Qwen3.8-27B | Notes |
|---|---:|---:|---|
| KYC India | 1/3 (33.3%) | 2/3 (66.7%) | |
| RBI Regulations | 0/3 (0.0%) | 1/3 (33.3%) | gpt-oss: 2 truncated; gpt-oss RBI UNRELIABLE (67%) |
| Image Processing | 3/3 (100%) | 3/3 (100%) | Text MCQ only |
| MMLU-Pro | 2/3 (66.7%) | 3/3 (100%) | |
| GSM8K | 3/3 (100%) | 3/3 (100%) | |
| MATH-500 | 3/3 (100%) | 3/3 (100%) | |
| GPQA | 3/3 (100%) | 3/3 (100%) | |
| HumanEval | 3/3 (100%) | 3/3 (100%) | |
| TruthfulQA | 3/3 (100%) | 3/3 (100%) | |

Pair 2, 2 items each:

| Benchmark | Nemotron | Llama-3.2-11B | Notes |
|---|---:|---:|---|
| KYC India | 1/2 (50.0%) | 1/2 (50.0%) | |
| RBI Regulations | 0/2 (0.0%) | 0/2 (0.0%) | Nemotron: 1 API timeout; RBI UNRELIABLE (50%) |
| Image Processing | 2/2 (100%) | 2/2 (100%) | Text MCQ only |
| MMLU-Pro | 2/2 (100%) | 0/2 (0.0%) | |
| GSM8K | 2/2 (100%) | 2/2 (100%) | |
| MATH-500 | 2/2 (100%) | 2/2 (100%) | |
| GPQA | 2/2 (100%) | 1/2 (50.0%) | Llama: 1 no answer; GPQA UNRELIABLE (50%) |
| HumanEval | 2/2 (100%) | 1/2 (50.0%) | One Llama test assertion failed |
| TruthfulQA | 2/2 (100%) | 2/2 (100%) | |

### 7.4 What the failures were

- gpt-oss-20b on `rbi_002` and `rbi_003`: `finish_reason: length`, `completion_tokens: 4096`. The reasoning text used the whole token budget and the answer was never produced. This is a token-limit problem, not an extraction bug.
- Nemotron on `rbi_002`: attempt log `503, 503, read_timeout, read_timeout`; it ended with `api_error: read timed out after 4 attempts`. NIM free tier is slow and returned 503s.
- Llama-3.2-11B on `gpqa_01`: `no_answer_found`.
- Llama-3.2-11B on `HumanEval/1`: generated code failed the test with an `AssertionError`. This is a normal incorrect answer, not a reliability flag.

### 7.5 Interpretation

- The numbers are from 27 items (pair 3) and 18 items (pair 2) per model. They are smoke checks, not rankings.
- The McNemar p-values are not significant. Do not claim one model is better.
- RBI was difficult for all four stand-ins. Spot-checking `rbi_001`-`rbi_003` against their excerpts showed the keys are consistent with the source, so these are not obvious key errors. They are model errors, truncation, or timeout.
- `kyc_002` (scope: every RBI-regulated entity) was missed by all four models. KYC is not trivially easy.
- Image Processing was 100% for all four, but with only 2-3 items it says almost nothing.
- NIM was slow and unreliable; Groq was fast. gpt-oss still truncated two RBI answers at 4096 tokens.
- The Llama vision model was used text-only. No vision evaluation was performed.
- A larger run with 15 rows per benchmark would be more informative but still not a final result. A full 100-row run is needed before serious claims.

---

## 8. What was completed in the latest work cycle

### 8.1 Dataset work

- Rebuilt the three new datasets to 100 rows each.
- Added source and source excerpt fields and checked all excerpts against the source text.
- Balanced answer positions to 25 of each letter.
- Completed human review of 20 image rows from a worksheet.
- Fixed weak image excerpts: `ip_020`, `ip_067`, `ip_078`, `ip_082`.
- Replaced `ip_044` because it had multiple defensible answers.
- Corrected `ip_078` page reference from 62 to 65.
- Marked 20 image rows `human_checked: true`.
- KYC and RBI each have 22/100 human-checked; RBI still has 3 `review: true`.
- Rebalanced GPQA and TruthfulQA from all-A to `A=4 B=3 C=4 D=4`.
- Reworded ambiguous GSM8K item `gsm_05`.
- Ran `--final` validation for all three new datasets; all PASS.
- Updated the rebalance test to reflect the shipped balanced files and avoid rebalancing already-balanced data.

### 8.2 Configuration work

- Set all three real pair configs to `limit_per_benchmark: 100`.
- Kept stand-in configs at 15.
- Configured the six real models with `provider: "auto"` and OpenRouter routes.
- Recorded substitute flags and notes for routes that are not exact.
- Added free-tier stand-in model blocks with measured/provisional token and timeout settings.
- Added NIM/Groq `requests_per_minute` settings to stay within free-tier limits.

### 8.3 Platform work included in this repository

The current state was copied to the new repository and committed. It includes:

- Groq and OpenRouter providers.
- `provider: "auto"` route selection and substitute reporting.
- Per-benchmark reliability flags.
- Incremental saving and INCOMPLETE runs.
- Scorer versioning and offline re-scoring.
- Math scorer v2.
- Dashboard background jobs.
- Dashboard warnings for stand-in/substitute/served model.
- Report improvements.
- Test coverage for providers, env, stand-in labelling, dashboard jobs, and rebalancing.

### 8.4 Tests

- Last full suite run: **402 passed** with `pytest` and `unittest`.
- Targeted rebalance tests passed after the test update.
- No code changes were made by the smoke runs.
- Run `python -m pytest -q` or `python -m unittest discover -s tests` after any change.

### 8.5 Current live smoke runs

- Pair 3 and pair 2 were run with the callable free-tier stand-ins.
- The runs are saved in `runs/` and registered in `runs/runs_registry.db`.
- They are live, completed, and labelled as stand-ins.
- Their scores are recorded in section 7 of this document with caveats.

---

## 9. How to install and run

### 9.1 Install

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest
$env:PYTHONDONTWRITEBYTECODE = "1"
```

Python 3.10 or newer is required. The work was run on Python 3.14. Run commands from the repository root because dataset and config paths are relative.

### 9.2 Environment keys

Create `.env` in the repo root. It is git-ignored.

```text
NVIDIA_API_KEY=<your NVIDIA key>
GROQ_API_KEY=<your Groq key>
OPENROUTER_API_KEY=<needed for the real pair routes; paid>
OPENAI_API_KEY=<only if you use provider "openai">
```

The CLI and dashboard load `.env` automatically. A variable already set in the shell is not overridden. Values are never printed by the loader.

### 9.3 List models, benchmarks, and runs

```powershell
python src/cli/nel.py list
```

This shows configured models, benchmarks, saved runs with mode and job name, and a separate table of INCOMPLETE/FAILED records.

### 9.4 Run a quick stand-in check

Use Groq for speed:

```powershell
python src/cli/nel.py eval run --config configs/standin_pair3.yaml --limit 3 --job-name standin_pair3_quick
```

Use NIM when testing NIM behavior, but expect slow calls:

```powershell
python src/cli/nel.py eval run --config configs/standin_pair2.yaml --limit 2 --job-name standin_pair2_quick
```

Do not run `configs/standin_pair1.yaml`; model A returns HTTP 404 and the run will stop early as FAILED.

### 9.5 Run a larger stand-in check

15 rows per benchmark for pair 3:

```powershell
python src/cli/nel.py eval run --config configs/standin_pair3.yaml --limit 15 --job-name standin_pair3_15rows
```

This is 135 prompts per model, 270 prompts for the pair. On Groq it is roughly 36 minutes at the configured rates, plus retries. A full NIM pair can take hours.

### 9.6 Run only the three new datasets at 100 rows

```powershell
python src/cli/nel.py eval run --config configs/standin_pair3.yaml --bench kyc_india --bench rbi_regulations --bench image_processing --limit 100 --job-name standin_pair3_new_data_100
```

This is 300 prompts per model, 600 for the pair. `--limit 0` also means all rows.

### 9.7 Run a real pair (after adding `OPENROUTER_API_KEY`)

```powershell
python src/cli/nel.py eval run --config configs/eval_pair3_ministral8b_reasoning_vs_qwen35.yaml --limit 10 --job-name pair3_openrouter_probe
```

Before a full run:

1. Confirm the printed run says LIVE.
2. Check the served model and substitute warning.
3. Check truncated/no-answer/API-error counts.
4. Check reliability.
5. Only then run with `--limit 0`.

The six real pair configs use `provider: "auto"` and only have OpenRouter routes, so without `OPENROUTER_API_KEY` they cannot run. If you add the key, OpenRouter is paid. Several routes are substitutes; report results for the served model, not the exact requested model.

### 9.8 Compare two runs

```powershell
python src/cli/nel.py compare --run-a <RUN_A_ID> --run-b <RUN_B_ID>
python src/cli/nel.py compare --run-a <RUN_A_ID> --run-b <RUN_B_ID> --output-md comparison.md
python src/cli/nel.py compare --run-a <RUN_A_ID> --run-b <RUN_B_ID> --output-html comparison.html
```

Current smoke comparisons:

```powershell
python src/cli/nel.py compare --run-a run_20261026_084909_38826d --run-b run_20261026_085214_f87b53
python src/cli/nel.py compare --run-a run_20261026_085713_000c7c --run-b run_20261026_090913_9cdb1e
```

### 9.9 Quality gate

```powershell
python src/cli/nel.py gate --baseline <RUN_A_ID> --candidate <RUN_B_ID> --min-acc 0.6 --max-reg 0.05
```

Options:

- `--min-acc`: minimum overall accuracy.
- `--max-reg`: maximum allowed regression from baseline.

The gate returns PASS or FAIL. It refuses runs that are not completed.

### 9.10 Offline re-score

```powershell
python src/tools/rescore_run.py <RUN_ID>
```

This makes no model call. It writes a new `<RUN_ID>_rescored_v<N>.json` file and never modifies the original. It can compare/gate, but it is not in the leaderboard or `nel list`.

### 9.11 Launch the dashboard

From the repository root:

```powershell
streamlit run src/dashboard/app.py
```

Open `http://localhost:8501`. Streamlit 1.51.0 is installed. If the port is busy:

```powershell
streamlit run src/dashboard/app.py --server.port 8502
```

Dashboard tabs:

- **Leaderboard**: all completed runs, with mode, job name, served model, stand-in/substitute warnings, reliability, and API-error counts.
- **Radar & domains**: domain-level comparison for selected runs.
- **Model diff & flips**: choose baseline A and candidate B; see McNemar p-value and item-level flips.
- **Launch evaluation**: defaults to Live stand-ins, selects benchmarks, sample cap, solver, and runs in a background thread. One model at a time. Status polls every 2 seconds.
- **Reports**: generate and download Markdown/HTML comparison reports.

Notes:

- The dashboard listens to `runs/runs_registry.db`; current smoke runs should appear in the Leaderboard.
- If the page was already open, reload it.
- The dashboard's "Live stand-ins" default is correct for current work.
- "All configured models" includes the auto real models, which have no OpenRouter key and will error.
- Do not select Gemma 3 12B; it returns 404.
- A dashboard run continues in a background thread, but status is in-process only. Restarting Streamlit loses the live status panel; the saved run file remains.
- The dashboard can set samples per benchmark up to 100 or use every row.

---

## 10. Current status

### Working and tested

- Deterministic simulator with seed support.
- Live providers fail loudly without a key.
- NVIDIA NIM, Groq, OpenRouter, OpenAI provider plumbing; OpenRouter never called.
- Auto route selection and substitute labels.
- Answer extraction for MCQ, math, and code.
- Reasoning stripping and separate reasoning fields.
- Truncation/no-answer/API-error flags.
- UNRELIABLE run and per-benchmark marks.
- Retries, timeout retry, throttling, attempt logs.
- HumanEval separate-process execution.
- Wilson intervals and McNemar.
- Run storage, registry, incremental saving.
- Scorer versioning, offline re-score.
- Dataset validator.
- Rebalance script.
- CLI list/eval/compare/gate.
- Streamlit dashboard with background jobs.
- Test suite: 402 passed.

### Verified with real services

- NIM key accepted; Groq key accepted.
- NIM and Groq model lists work.
- Replies read from four stand-in models:
  - NIM Llama-3.2-11B: no thinking field.
  - NIM Nemotron: `message.reasoning_content`.
  - Groq gpt-oss-20b: `message.reasoning`.
  - Groq Qwen3.8-27B: no separate thinking field.
- Two full 8-benchmark stand-in runs finished earlier (pair 3 on Groq, pair 2 on NIM) as pipeline validation.
- Pair 3 and pair 2 smoke runs finished on 26 October 2026 and are saved.
- NIM 503s and read timeouts were observed.
- Groq was fast and clean except gpt-oss truncation.
- No real OpenRouter call has been made.
- The early stop has never fired on a real run.

### Not done / not verified

- **No result for any required model exists.**
- Real pair models cannot run on NIM/Groq.
- OpenRouter routes are unverified and paid.
- Several OpenRouter routes are substitutes.
- Pair 1 stand-in cannot run because Gemma returns 404.
- No full 100-row or 15-row pair run has been completed for the new data.
- Most dataset keys are not independently verified.
- KYC/RBI: 22/100 human-checked each; RBI still has 3 review flags.
- Image: 20/100 human-checked.
- GPQA/TruthfulQA: content unverified, 15 items each.
- `gsm_05` old run verdicts are obsolete.
- No vision-model benchmark exists. The `image_processing` dataset is text-only.
- No new provider for self-hosted vLLM/OpenAI-compatible servers.
- `llm_judge.py` is present but unused.
- The HumanEval runner is a timeout, not a security sandbox.
- Runs are sequential; no concurrency.
- Comparison matches by `question_id` alone.
- McNemar has no multiple-comparison correction.
- Quality gate is only PASS/FAIL.
- NIM is slow; full NIM runs can take hours.
- gpt-oss-20b can exhaust 4096 tokens on reasoning and truncate.
- The old documents under `docs/` are outdated.

---

## 11. Known gotchas

- **Run from the repository root.** Paths are relative.
- A mistyped `--config` path is silently treated as an empty config. Check the "Starting job" panel.
- An unknown `--model` ID gets a simulated config instead of an error.
- There is no `--provider` CLI option. Change the provider in `configs/models_config.yaml`.
- The `execution:` and `quality_gate:` blocks in job YAMLs are not read by code. Retries are fixed in the client; runs are sequential.
- `benchmarks_config.yaml` `sample_file` values are display-only. Actual paths are hardcoded in `src/benchmarks/base.py`.
- `--limit N` takes the first N rows, not a random sample. Smoke runs use the first rows.
- `--limit 0` means all rows.
- `OPENROUTER_API_KEY` is needed for the six real pair models. Without it, auto has no route and the model errors.
- Pair 1 stand-in model A returns 404; do not run it.
- The dashboard's "All configured models" includes models that cannot run with the current keys.
- `nel list` excludes INCOMPLETE and FAILED runs from the main list; they are in a separate table.
- `compare` and `gate` refuse INCOMPLETE/FAILED runs.
- `rescore_run.py` output is not in the leaderboard.
- Do not commit `.env` or API keys. The source PDFs and text copies used for dataset validation are tracked in `_sources/`; do not add unrelated large files.
- This repository contains the current working state. The older `nemo-evaluator` folder is no longer the source of truth.

---

## 12. Open decisions

1. **Host the models on a GPU or call them through an API?**
   This decides whether a new self-hosted OpenAI-compatible provider is needed and which model IDs apply.
2. **Will the team pay for OpenRouter and accept substitute served models?**
   Ministral 8B/14B and Qwen3.5-9B are not the exact requested variants.
3. **Which Qwen3.5 size is meant?**
   The brief does not say. The current config assumes 8B and routes to 9B.
4. **What should replace pair 1 model A?**
   Every small instruct model tried on NIM returned 404. Groq `allam-2-7b` was considered and not added.
5. **What does the vision-model part of the brief require?**
   No models, images, or tasks were specified. Nothing is implemented.
6. **What does "features" mean in the brief?**
   The working assumption is that each pair is compared on the same benchmarks and settings.
7. **Are GPQA and TruthfulQA kept?**
   They are 15 items each and their keys are unverified, though their answer-position bias is fixed.
8. **How many dataset keys will be independently checked?**
   Current human coverage is 22/100 KYC, 22/100 RBI, 20/100 image.
9. **What token limits and timeouts should the real reasoning models use?**
   Current settings are provisional and measured mostly on easy stand-in questions.

---

## 13. Recommended next steps

1. Decide the real hosting/API route.
2. If OpenRouter is acceptable, add `OPENROUTER_API_KEY`, run a 10-item probe for one real pair, inspect the served model/substitute flag, then run a full pair. Do not call the results exact requested-model results if the route is a substitute.
3. If GPU hosting is chosen, implement a provider for the OpenAI-compatible server, with a user-chosen HTTPS host and no vendor key.
4. Replace pair 1's non-callable stand-in model A or drop pair 1 from the current phase.
5. Increase gpt-oss `max_tokens` above 4096, or use a non-reasoning stand-in, before a larger run that includes RBI.
6. Run a larger stand-in check at 15 rows per benchmark on Groq to get more stable pipeline numbers.
7. Verify the 3 RBI `review: true` keys and then expand human checking beyond 22/100.
8. Run the full 100-row new datasets when the team is ready to spend the time.
9. Define the vision-model scope before writing any vision code.
10. Update this handover and `CONTEXT.md` after each significant change.

---

## 14. New-person onboarding checklist

1. Open the repository root.
2. Read this document, then `CONTEXT.md` sections 0, 1, 3, 9, 10, 11, and 12.
3. Activate the virtual environment and install requirements.
4. Check key names without printing values.
5. Run `python src/cli/nel.py list`.
6. Run `streamlit run src/dashboard/app.py` and open `http://localhost:8501`.
7. Find the current stand-in runs in the Leaderboard.
8. Run a 2-3 item Groq stand-in check.
9. Validate one new dataset with `validate_dataset.py`.
10. Read the model config and the three pair configs.
11. Do not run a full NIM pair or a real pair without checking time, cost, keys, and substitutes.
12. If you report anything, state the sample size, run mode, model/served model, and reliability.

---

## 15. Glossary

| Term | Meaning |
|---|---|
| Benchmark | A fixed question set with a scoring rule |
| Dataset | The JSONL file behind a benchmark |
| Item | One question in a run |
| Run | One model over one set of benchmarks |
| Simulated | Provider `mock`; no API call |
| Live | Any run that reaches a provider over the network |
| Stand-in | A cheaper look-alike model used for pipeline validation |
| Substitute route | A live route serving a different model than requested |
| Served model | The model ID the provider actually answered with |
| Solver | How the prompt is written (`simple`, `cot`, `harbor`) |
| Scorer | Code that extracts and judges the answer |
| Flip | A question where one model is right and the other is wrong |
| McNemar | Paired significance test on disagreements |
| Wilson interval | 95% confidence interval for accuracy |
| Reliability flag | Marks a run/benchmark with too many failed items |
| Quality gate | PASS/FAIL release decision |
| Scorer version | Stamped on a run; different versions are not directly comparable |
| `human_checked` | A person actually reviewed and accepted the item |
| `review` | The item still needs a person to check it |

---

## 16. Quick reference: current run IDs and commands

Current stand-in smoke runs:

```text
Pair 3 A: run_20261026_084909_38826d  openai/gpt-oss-20b
Pair 3 B: run_20261026_085214_f87b53  qwen/qwen3.8-27b
Pair 2 A: run_20261026_085713_000c7c  nvidia/nemotron-3-nano-omni-30b-a3b-reasoning
Pair 2 B: run_20261026_090913_9cdb1e  meta/llama-3.2-11b-vision-instruct
```

Compare:

```powershell
python src/cli/nel.py compare --run-a run_20261026_084909_38826d --run-b run_20261026_085214_f87b53
python src/cli/nel.py compare --run-a run_20261026_085713_000c7c --run-b run_20261026_090913_9cdb1e
```

Dashboard:

```powershell
streamlit run src/dashboard/app.py
```

Dataset validation:

```powershell
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --final --source-text "_sources/kyc_master_direction.txt"
python src/tools/validate_dataset.py datasets/rbi_regulations/sample.jsonl --final --source-text "_sources/basel_iii_master_circular.txt" --source-text "_sources/irac_advances_master_circular.txt"
python src/tools/validate_dataset.py datasets/image_processing/sample.jsonl --final --source-text "_sources/young_fundamentals_image_processing.txt"
```

Tests:

```powershell
python -m pytest -q
```

---

## 17. Final status statement

The platform and the three 100-question datasets are in a demonstrable state. The live pipeline works end to end with free-tier stand-ins. The required model comparison is **not done**, because the required models are not available on the free providers and OpenRouter has not been enabled. The current numbers are small-sample stand-in smoke results and must never be presented as results for the required models.
