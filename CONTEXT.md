# Project context: NeMo Evaluator benchmarking platform

This file is for anyone picking up the repository without the conversation that built it, including a person
working with their own AI tool. It was written from the code and from commands that were actually run.
Where something was **not** verified, it says so. If this file and the code disagree, the code wins; please fix this file.

## 0. Rules that must not be broken

1. **Never present simulated or stand-in results as results for the real models.** Simulated runs come from a
   built-in fake model. Stand-in runs come from cheaper look-alike models used only to test the pipeline.
   Neither says anything about the models the project is meant to compare.
   Note the distinction, because it is easy to get wrong: **a run is only *simulated* when no API call is
   made** (provider `mock`). A stand-in run **does** make real API calls, so it is a **live** run, and its
   scores *are* real results - but for the **stand-in model only**. They are not results for the required
   models, and must never be labelled or reported as such.
   **A substitute route is a third case.** A `provider: "auto"` model can be served by a different model ID
   than the one requested (for example the exact Ministral variant is not hosted). The run records both IDs
   and `model_substituted: true`. Its scores are results for the **served** model; do not report them as
   results for the exact requested model.
2. **Never write regulatory or factual benchmark content from memory.** Every question must be written from source
   text the author has in front of them, and must say where it came from (section 6).
3. **Never invent or hand-write benchmark scores.** Scores come from running the code, nothing else.
4. **Never put an API key in a file that is committed, or print it.** Keys live in `.env` (git-ignored) or in
   environment variables.
5. If something failed or was not verified, say so in whatever you write up.

## 1. What the project is

- A platform that gives language models the same set of questions (benchmarks), scores the answers automatically,
  and compares two models statistically.
- It is a university capstone project. The industry use case is choosing between LLMs for enterprise work, with
  India-specific banking/KYC compliance questions as a focus.
- Per model it reports accuracy with a 95% Wilson confidence interval, latency, tokens and cost. For two models it
  runs a paired McNemar test and lists the individual questions where they disagree ("flips").
- It can talk to NVIDIA NIM, Groq, OpenRouter, the OpenAI API and Ollama, and has a built-in simulator so everything
  can be tested with no API calls. `provider: "auto"` resolves to the first configured route whose key is set.
- Interfaces: a command-line tool (`python src/cli/nel.py`) and a Streamlit dashboard. The CLI is what this file
  documents in detail; the dashboard now runs evaluations in a background thread (the page stays responsive and
  polls the job every 2 seconds), shows run mode, job name, served model, substitute-route and stand-in warnings,
  and defaults to the live stand-ins. It launches one model at a time.

## 1a. What was asked for

This is the brief as the project owner stated it. Anything marked **(assumption)** was *not* stated and is only a
working assumption. Please confirm or correct those points; the rest of this file depends on them.

**The three model pairs, exactly as written in the request**

| pair | model A | model B |
|---|---|---|
| 1 | `mistralai/Ministral-3-8B-Instruct-2512` | `meta-llama/Llama-3.1-8B-Instruct` |
| 2 | `mistralai/Ministral-3-14B-Reasoning-2512` | `meta-llama/Llama-4-Scout-17B-16E` |
| 3 | `mistralai/Ministral-3-8B-Reasoning-2512` | `Qwen3.5` (**size not specified**) |

- These are **Hugging Face-style model IDs**. The repo's configs use different spellings, which are NIM-style and
  lower-case (for example `meta/llama-3.1-8b-instruct`, `mistralai/ministral-3-8b-instruct-2512`). Whichever way the
  models are run, each name above has to be mapped to the exact ID that the chosen provider serves.
- Two details in the repo are **not** from the request: pair 2's second model is spelled with an `-instruct` suffix in
  `configs/models_config.yaml`, and pair 3's second model is configured as `qwen/qwen3.5-8b`. The request gives no
  Qwen3.5 size, so "8B" is an assumption that came from the repo's config. Other sizes of Qwen3.5 may be what is meant.
- **(assumption)** It is **unconfirmed whether the team plans to host these models on a GPU or to call an API.**
  The repo today assumes an API. What the code can and cannot do (checked in `src/gateway/model_client.py`):
  - Supported live providers are `nvidia_nim`, `groq`, `openrouter`, `openai` and `ollama`. A `huggingface`
    provider is **not implemented** (it raises an error). `provider: "auto"` picks one of the keyed providers by
    which key is set (section 3a).
  - A self-hosted server that speaks the OpenAI protocol (for example vLLM serving a Hugging Face model) **cannot be
    used with the existing `openai` provider**, because that provider only accepts `https://api.openai.com`. Using
    such a server needs a code change (a new provider with a user-chosen host and no vendor key).
  - `ollama` accepts any host, but it serves models under Ollama's own names, not Hugging Face IDs.
  - So the answer to "GPU or API" decides what has to be built; it should be settled before choosing model IDs.

**The three new datasets**

- **KYC India**: 100 questions (repo id `kyc_india`).
- **RBI bank regulation, India**: 100 questions (repo id `rbi_regulations`).
- **Image processing** (repo id `image_processing`): the request gives no count. **(assumption)** 100, like the
  other two, which is what the project's working rules use.
- The rules for writing them are in sections 6 to 8.

**Vision models**

- Requested in addition to the text pairs. **No detail was given**: which models, which images, which tasks. Nothing
  is implemented: the repo is text-only (no image input in the client, the dataset format or the scorers), and the
  `image_processing` set is text multiple-choice *about* image-processing theory, not a test of a vision model.

**What "features" means**

- The word "features" in the request is **undefined**: it could mean capabilities, benchmark categories or model
  features. **(assumption)** The working assumption is that **each pair is compared on the same benchmarks, under the
  same settings, including the three new datasets** (`kyc_india`, `rbi_regulations`, `image_processing`). The three
  pair job files in `configs/` already list those three plus the six older benchmarks (`mmlu_pro`, `gsm8k`,
  `math_500`, `gpqa`, `humaneval`, `truthfulqa`).

## 2. Architecture map and how a run flows

```
configs/                    YAML: models, benchmarks, and "job" files (which models x benchmarks x how many items)
datasets/<id>/sample.jsonl  question files, one per benchmark
src/cli/nel.py              command line: list | eval run | compare | gate
src/engine/runner.py        EvaluationRunner: the main loop
src/engine/solvers.py       turns a dataset row into the prompt text (simple | cot | harbor)
src/engine/answer_scoring.py  scores one reply: strips reasoning text, flags truncated / no-answer
src/engine/sandbox.py       runs generated Python for HumanEval in a separate process (3 s timeout)
src/gateway/model_client.py UnifiedModelClient: picks the provider (or resolves `auto` routes), enforces key/host rules, optional model list call
src/gateway/mock_models.py  the simulator
src/gateway/env_loader.py   reads .env
src/gateway/interceptor.py  in-memory response cache (per process)
src/benchmarks/base.py      one class per benchmark + get_benchmark(); decides which file each benchmark reads
src/benchmarks/loader.py    reads a JSONL file
src/scoring/rule_scorers.py   multiple-choice answer extraction (its math methods call math_answers.py)
src/scoring/math_answers.py   math answer extraction, normalisation and comparison (section 9)
src/scoring/version.py      SCORER_VERSION, stamped on every run file (section 9a)
src/scoring/reasoning.py    strips <think>...</think> text
src/scoring/code_extraction.py  pulls runnable Python out of a reply
src/scoring/statistics.py   Wilson interval, McNemar test, quality gate
src/scoring/llm_judge.py    present but NOT used by any run
src/storage/                RunArtifact (pydantic) + RunStore (runs/<run_id>.json and runs/runs_registry.db)
src/comparison/diff_engine.py  aligns two runs by question id, finds flips, runs McNemar
src/reporting/report_generator.py  Markdown/HTML comparison report
src/dashboard/app.py        Streamlit app
src/dashboard/jobs.py       background run thread + in-memory job status for the dashboard
src/tools/validate_dataset.py  dataset checker (section 8)
src/tools/rebalance_answers.py  script that reshuffles answer positions into a copy; never edits a file in place (section 8)
src/tools/rescore_run.py    re-scores the stored replies of a finished run offline into a NEW labelled run file (section 9a)
tests/                      pytest/unittest suite
```

**One run, step by step** (`python src/cli/nel.py eval run --config <job.yaml> ...`):

1. The CLI loads the job YAML and `configs/models_config.yaml`. `--model`, `--bench` (repeatable) and `--limit`
   override the job file.
2. For each model ID it looks the model up in `models_config.yaml`. An ID that is not found gets a **made-up
   simulated config** (so a typo in `--model` gives a simulated run, labelled SIMULATED, not an error).
3. `EvaluationRunner.run_evaluation` creates a `UnifiedModelClient`. For a live provider this **raises an error**
   if the API key is missing, the provider is unsupported, or the endpoint is not the provider's own host.
   For `provider: "auto"` the client picks the first route whose key is set, in the order NIM, Groq, OpenRouter,
   OpenAI; it records the requested ID, the served ID, whether the route is a substitute and a route note.
4. For each benchmark, `get_benchmark(id)` loads `datasets/<id>/sample.jsonl` (the first `--limit` rows; 0 = all).
5. For each row: the solver builds the prompt, the client gets a reply (simulator, NIM/Groq/OpenRouter/OpenAI, or
   Ollama), then `score_generation` scores it (section 9) and an `ItemResult` is stored with the full prompt, the raw
   reply, the extracted answer, tokens, cost, latency and the flags.
6. Per-benchmark and overall accuracy (all items pooled) get Wilson 95% intervals. **After every finished benchmark
   except the last, the partial run is saved** to `runs/<run_id>.json` as an **INCOMPLETE** record (section 9c). When
   the model has finished, the same file is overwritten by the final record, and a summary row is added to
   `runs/runs_registry.db`. Every run file carries the `scorer_version` that produced its verdicts (section 9a).
7. `compare` loads two run files, matches questions **by `question_id`**, runs McNemar on the paired results,
   and can write a Markdown/HTML report whose header says SIMULATED or LIVE and shows the scorer version. It refuses a
   run that is not COMPLETED.
8. `gate` applies a pass/fail rule: the candidate must reach a minimum accuracy and not fall more than a set
   amount below the baseline. It has only PASS and FAIL. It also refuses a run that is not COMPLETED.

## 3. How to run it

### Install (PowerShell, from the repo root)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install pytest                      # only needed to run the tests; not listed in requirements.txt
$env:PYTHONDONTWRITEBYTECODE = "1"      # stops .pyc files appearing in git status
```

Python 3.10 or newer is required (`pyproject.toml`). The work described here was run on Python 3.14.
Run every command from the **repo root**: dataset and config paths are relative to the current directory.

### Commands

```powershell
python src/cli/nel.py list                                   # models, benchmarks, saved runs (with mode and job name), then a separate table of INCOMPLETE / FAILED records
python src/cli/nel.py eval run --model meta/llama-3.1-70b-instruct --bench mmlu_pro --limit 10
python src/cli/nel.py eval run --config configs/eval_pair1_ministral8b_vs_llama31.yaml --limit 10
python src/cli/nel.py eval run --config <job.yaml> --limit 0  # 0 = every row in each dataset
python src/cli/nel.py compare --run-a <RUN_ID_A> --run-b <RUN_ID_B> --output-md report.md
python src/cli/nel.py gate --baseline <RUN_ID_A> --candidate <RUN_ID_B>
python src/tools/rescore_run.py <RUN_ID> [<RUN_ID> ...]      # offline, no model call: new run file <RUN_ID>_rescored_v<N>.json (section 9a)
python -m pytest -q                                          # or: python -m unittest discover -s tests
```

`eval run` options: `--config`, `--model`, `--bench` (repeatable), `--limit`, `--solver simple|cot|harbor`,
`--job-name`. Run IDs are printed after each run and shown by `list`.

Careful: the three real pair job files now set `limit_per_benchmark: 100`, so a full run uses every row in each
dataset (the older 15-row benchmarks still contribute only 15). The stand-in job files still set 15 for quick
pipeline checks.

A mistyped `--config` path is silently treated as an empty config (the run falls back to defaults). Check the
"Starting job" panel the CLI prints.

### Simulated vs live

- Which one you get is decided by the `provider:` line of each model in `configs/models_config.yaml`.
  `mock` = simulator. `nvidia_nim`, `groq`, `openrouter`, `openai`, `ollama` = real calls.
  `auto` = a real call through the first configured route whose key is present (NIM, then Groq, then
  OpenRouter, then OpenAI). The six real pair models now use `auto`; the older non-pair models still say `mock`.
- Every run records `run_mode`: `simulated` or `live` (older run files that predate the field are shown as
  `simulated` if their provider was `mock`, otherwise `unknown`). `list`, `eval run`, `compare`, `gate` and the
  report header all show it.
- An auto run also records `served_model_id`, `model_substituted` and `route_note`. `model_id` is always the
  **requested** model, so a run can be traced back to the brief; the served model is what actually answered.
- There is **no** command-line option to change the provider; you edit the YAML (below). Setting a route key in
  `.env` is enough to make an auto model live.

### `.env` and key names

Create a file named `.env` in the repo root (it is listed in `.gitignore`; never commit it):

```
NVIDIA_API_KEY=<your NVIDIA key>
GROQ_API_KEY=<your Groq key>
OPENROUTER_API_KEY=<needed by the six real pair models' auto routes; paid>
OPENAI_API_KEY=<only if you use provider "openai">
```

The CLI and dashboard read it automatically. It never overrides a variable already set in your shell, and the
loader never prints values. You can set the variable in the shell instead: `$env:NVIDIA_API_KEY = "..."`.

| provider | key variable | only talks to | notes |
|---|---|---|---|
| `mock` | none | nothing (offline) | the simulator |
| `nvidia_nim` | `NVIDIA_API_KEY` | `https://integrate.api.nvidia.com/v1` | free tier |
| `groq` | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` | free tier |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` | paid per use; **never called for real yet** (section 3a) |
| `openai` | `OPENAI_API_KEY` | `https://api.openai.com/v1` | paid; not used by the real pair routes |
| `ollama` | none | `api_base`, default `http://localhost:11434` | local models |
| `auto` | each route's own key | the chosen route's own host | picks a route by key presence, free first (section 3a) |
| anything else (e.g. `huggingface`) | - | - | **not implemented**; raises an error |

A key is only ever sent to its own provider's host: the client refuses a model whose `api_base` is another host
(https only, no other port, no look-alike or `user@host` tricks), refuses to read another provider's key variable,
never follows HTTP redirects, and never puts the key in an error message. A test sweeps all configured models against
every provider, key variable and endpoint, and fails if the NVIDIA, OpenAI, Groq or OpenRouter key could reach any
other host.

### Rate limits, retries and throttling (live providers)

- **Retries.** An HTTP **429** or **5xx** reply is retried up to **4 times** (so at most 5 requests per item), waiting
  2, 4, 8 and 16 seconds, or the server's `Retry-After` if that is longer (a `Retry-After` above 120 s is not waited
  for; the item fails instead). **Other 4xx errors are never retried** (400, 401, 403, 404, 408, 410, 422, ...).
  A **read timeout is retried once** (after 2 s), on a separate budget from the 429/5xx retries, so an item makes at
  most 1 + 4 + 1 = 6 requests. Connect timeouts and other connection errors are not retried.
- **Throttle.** Each model keeps its requests at least `60 / requests_per_minute` seconds apart. The default is
  **20 requests per minute** (one request every 3 s); change it per model with `requests_per_minute:` in
  `models_config.yaml`. Local `ollama` is not throttled unless you set it. Retries are throttled too; cache hits are not.
- **A failed item is not skipped.** It is scored incorrect, flagged `api_error: ...`, counted, and the run goes on.
  The count of `api_error` items is shown per benchmark in the run table, in the `compare` panel and in the report;
  each item also records how many HTTP `attempts` it took, the HTTP status of the final failure (`http_status`) and an
  **`attempt_log`**: one entry per request, in order, so retries can be audited. An entry is the HTTP status as text
  (`"429"`, `"503"`, `"200"`), `"read_timeout"`, or the name of a network error class (`"ConnectionError"`). For example
  `["429", "503", "200"]` is an item that was rate limited, then got a server error, then succeeded. The log holds only
  these short strings, never the key or the reply text. Run files saved before this field existed have an empty log,
  and simulated items have an empty log too (no HTTP). Failed calls are not cached.
- **Early stop for an unavailable model.** If the **first 5 items** of a model all fail with the **same** one of
  **401, 403, 404 or 410**, the model is aborted with a message naming the HTTP status and the model, and the run is
  marked **FAILED** (not UNRELIABLE). Only these four statuses count: 429, 5xx, timeouts, every other 4xx, a mix of
  statuses, a success among the first five, or fewer than five items never trigger it. A failed run is saved as a JSON
  record, kept out of `nel list` and the leaderboard, and `compare` / `gate` refuse it. In a config with several models
  the other models still run, and the command exits with an error at the end.
- A long `latency` in a run reflects the last request only (waiting and backoff are not included).

### Running a real pair

The six real pair models are already configured with `provider: "auto"` and an OpenRouter route, so the only
missing piece is a key:

1. Put `OPENROUTER_API_KEY` in `.env` (OpenRouter is paid; add credit).
2. Try 10 items first: `python src/cli/nel.py eval run --config configs/eval_pairN_....yaml --limit 10`.
3. Check the printed table says LIVE, shows the served model and does not show a substitute warning unless the
   route is marked substitute, then check the "Truncated / No answer / API error" column.
4. Run with `--limit 0`, then `compare` the two run IDs.

**None of the three real pairs can run live on NIM or Groq**: their model IDs are not on those providers
(section 10). The OpenRouter route is configured but **no real OpenRouter call has been made**, because no
`OPENROUTER_API_KEY` was available while this was written (section 3a).

## 3a. Running the real pairs through OpenRouter (with your own key)

OpenRouter is a service that puts one OpenAI-style API in front of many hosted models and charges per use. The repo
has an `openrouter` provider with the same safeguards as the others: it only talks to
`https://openrouter.ai/api/v1`, only reads `OPENROUTER_API_KEY`, is https only, never follows redirects and never
prints the key. It is covered by tests with made-up replies. **No real OpenRouter call has ever been made with it.**

1. **Account and key.** Create an OpenRouter account and an API key (add credit if the models you need are not free;
   the model list shows prices). The key is yours: do not share it, do not commit it. Put it in `.env`:
   ```
   OPENROUTER_API_KEY=<your OpenRouter key>
   ```
2. **The routes are already in `configs/models_config.yaml`.** The six real pair models use `provider: "auto"`
   with an OpenRouter route. Auto tries NIM, then Groq, then OpenRouter, then OpenAI, and uses the first route whose
   key is set. Because the six models have no NIM or Groq route, the OpenRouter key is the one that makes them live.
3. **Try 10 items first** (`--limit 10`), then run for real. In the output check: the run says LIVE, it shows the
   served model and any substitute warning, there is no STAND-IN banner, the `api_error`, truncated and no-answer
   counts are small, and the run is not marked UNRELIABLE.
4. **Before calling anything a result:** all three new datasets pass `--final` with their source text, but human
   checks cover only 20/100 image rows and 22/100 rows in each of KYC and RBI, so most keys are not independently
   verified. `gpqa` and `truthfulqa` answer positions have been rebalanced, but their questions and keys remain
   unverified (section 12); and every sentence you write must say which provider served the model and whether it was the exact
   requested variant.

**Model IDs checked against OpenRouter's model list (October 2026).** The list has
`meta-llama/llama-3.1-8b-instruct`, `meta-llama/llama-4-scout`, `mistralai/ministral-8b-2512`,
`mistralai/ministral-14b-2512` and `qwen/qwen3.5-9b`. It has **no exact Ministral-3 variant** and **no Qwen3.5-8B**.
The configured substitutions, each flagged with `substitute: true` and a `note:` in the route:

| requested model | OpenRouter route | substitute? |
|---|---|---|
| `mistralai/ministral-3-8b-instruct-2512` | `mistralai/ministral-8b-2512` | yes |
| `mistralai/ministral-3-8b-reasoning-2512` | `mistralai/ministral-8b-2512` | yes (same served model as pair 1 A) |
| `mistralai/ministral-3-14b-reasoning-2512` | `mistralai/ministral-14b-2512` | yes |
| `qwen/qwen3.5-8b` | `qwen/qwen3.5-9b` | yes (9B stands in for the unspecified 8B) |
| `meta-llama/llama-3.1-8b-instruct` | `meta-llama/llama-3.1-8b-instruct` | no |
| `meta-llama/llama-4-scout` | `meta-llama/llama-4-scout` | no |

The route costs in `models_config.yaml` are OpenRouter's per-1M-token prices at the time of checking (Ministral 8B
0.15/0.15; Ministral 14B 0.20/0.20; Llama 3.1 8B 0.05/0.08; Llama 4 Scout 0.10/0.30; Qwen3.5-9B 0.17/0.25).
`max_tokens` is 4096 and `timeout_seconds` 240 for the reasoning routes; these settings have **not** been measured
on the served models.

**Things not verified:** whether each route can actually be called (the model list is not proof; section 10 shows
that); whether the served Ministral model is instruct or reasoning; OpenRouter's upstream provider choice (it can
route one model to different hosts with different precision, context length and limits); whether a separate
`reasoning` field arrives as assumed; and the request-rate limits of the account.

## 4. Datasets: layout and format

### Layout and how each benchmark finds its file

Each benchmark reads exactly one file, `datasets/<benchmark_id>/sample.jsonl`. The path is written in the
benchmark's class in `src/benchmarks/base.py` and is relative to the current directory, so run from the repo
root. The `sample_file:` entries in `configs/benchmarks_config.yaml` are **not read by any code** (that file is
only used for the names and descriptions shown by `list` and by the dashboard). `--limit N` takes the first N lines.

JSONL means one complete JSON object per line, UTF-8, **no byte-order mark** (Windows Notepad can add one; the
loader then fails on line 1), no trailing commas, no line breaks inside an object. Blank lines are skipped.
Windows (CRLF) line endings work; the shipped files use them.

| benchmark id | type | id field | `answer` holds | options |
|---|---|---|---|---|
| `mmlu_pro` | multiple choice | `question_id` | one letter A-J | exactly 10 |
| `gpqa`, `truthfulqa`, `custom_byob` | multiple choice | `question_id` | one letter A-D | exactly 4 |
| `kyc_india`, `rbi_regulations`, `image_processing` | multiple choice | `question_id` | one letter A-D | exactly 4 |
| `gsm8k`, `math_500` | free answer | `question_id` | the final answer as a string, e.g. `"72"`, `"1/6"` | none |
| `humaneval` | code | `task_id` | (none; `test` decides pass/fail) | none |

Four options and letters A-D are a hard limit for the A-D benchmarks: the scorer only accepts those letters, so a
fifth option or an answer `E` can never be scored correctly.

### Fields

Multiple choice (all nine MCQ benchmarks):

| field | required | meaning |
|---|---|---|
| `question_id` | yes | unique string. Must be unique **across all datasets**, because comparison pairs questions by this id alone |
| `question` | yes | the question text only (no lettered options inside it) |
| `options` | yes | list of exactly 4 (or 10) plain strings, **without** "A." prefixes: the prompt adds the letters |
| `answer` | yes | one uppercase letter = the position of the correct option (first option = `A`) |
| `topic` / `domain` | no | free label; the shipped files use `topic` (kyc, rbi, image) or `domain` (mmlu_pro, custom_byob) |
| `source`, `source_excerpt` | **required for the three new datasets** | section 6 |
| `review` | no | JSON boolean, section 6 |

Free answer (`gsm8k`, `math_500`): `question_id`, `question`, `answer` (string). `math_500` rows also have `subject`.
The rules for finding and comparing the final answer are in section 9.

**`gsm_05` was ambiguous and has been reworded.** The original question ("Baker bakes 120 loaves, sells 3/4 by
noon. Bakes 60 more, sells half. How many left?", key `60`) could mean half of everything the baker then has (45 left)
or half of the 60 baked later (60 left). In the stand-in runs every model that read it the first way was marked wrong.
It now reads "Baker bakes 120 loaves and sells 3/4 of them by noon. She bakes 60 more loaves and sells half of those
60 new loaves. How many loaves are left?" with key `60`. Old run files keep their old verdicts; do not reuse them for
GSM8K.

Code (`humaneval`): `task_id`, `prompt` (the function signature plus docstring), `canonical_solution`, `test`
(Python that defines `check(candidate)`), `entry_point` (the function name). The model is only shown `prompt`
plus an instruction line; `test` and `canonical_solution` are never shown to it. `canonical_solution` is only used
by the simulator.

**The loader and extra fields.** The loader (`BenchmarkLoader.load_jsonl`) returns each line as a plain dict, so
extra fields such as `source`, `source_excerpt` and `review` are accepted. This is covered by a
test. The solvers build the prompt only from `question`/`prompt` and `options`, so extra fields are never shown to
the model. Extra fields are also not copied into run files. (The pydantic `EvalItem` class in `schema.py` is not
used anywhere.) The validator warns about field names it does not know, to catch typos such as `sourse`.

### Format examples (placeholders only, not facts, do not copy)

> Every value in angle brackets below is a placeholder. These rows show the **shape** of a row and nothing else.
> They are deliberately not valid dataset content.

Multiple choice with 4 options, e.g. `kyc_india` (`rbi_regulations` uses `rbi_` ids, `image_processing` uses `ip_`):

```json
{"question_id": "<prefix>_<number>", "topic": "<short label>", "question": "<question written from the source passage>", "options": ["<option 1>", "<option 2>", "<option 3>", "<option 4>"], "answer": "<one of A, B, C, D>", "source": "<full document title and year; section or paragraph>", "source_excerpt": "<sentence copied word for word from the source>", "review": false}
```

Multiple choice with 4 options, no `source` fields (`gpqa`, `truthfulqa`):

```json
{"question_id": "<prefix>_<number>", "question": "<question>", "options": ["<1>", "<2>", "<3>", "<4>"], "answer": "<A-D>"}
```

`mmlu_pro` (10 options):

```json
{"question_id": "<prefix>_<number>", "domain": "<label>", "question": "<question>", "options": ["<1>", "<2>", "<3>", "<4>", "<5>", "<6>", "<7>", "<8>", "<9>", "<10>"], "answer": "<A-J>"}
```

`gsm8k` / `math_500`:

```json
{"question_id": "<prefix>_<number>", "subject": "<math_500 only>", "question": "<problem text>", "answer": "<final answer as a string>"}
```

`humaneval`:

```json
{"task_id": "<Name>/<number>", "prompt": "<signature and docstring>", "canonical_solution": "<indented function body>", "test": "<def check(candidate): ... asserts>", "entry_point": "<function name>"}
```

## 5. How to add or replace a dataset

### Replace the questions of an existing benchmark (what the three new datasets need)

1. Write the new rows to `datasets/<id>/sample.jsonl` (same file name; the benchmark class hardcodes it).
2. Use `question_id` values that exist in no other dataset (`kyc_001` ... `kyc_100`; the old 15-row files use
   `kyc_01` ...; they are replaced, not extended, so the ids do not clash).
3. Run the validator after every batch (section 8).
4. Nothing else has to change: the benchmark class, the mapping in `get_benchmark`, `benchmarks_config.yaml` and the
   simulator already know these ids.
5. Remember `--limit 0` (or a larger `limit_per_benchmark` in the job file) so all rows are used.

### Add a brand-new benchmark

1. `datasets/<new_id>/sample.jsonl` with the format of the closest existing type.
2. `src/benchmarks/base.py`: add a class (copy `KYCIndiaBenchmark` for a 4-option MCQ; the constructor argument that
   matters is the file path `datasets/<new_id>/sample.jsonl`) and add it to the `mapping` dict in `get_benchmark`.
3. `configs/benchmarks_config.yaml`: add an entry (needed for `list` and for the dashboard's benchmark picker;
   the `scorer:` and `sample_file:` values are descriptive only).
4. `src/gateway/mock_models.py`: add the id to the MCQ list in `generate_response`, otherwise simulated replies for
   it will not contain a recognisable answer and every item will show up as `no_answer_found`. Without an entry in a
   model's `accuracy_bias` the simulator uses 0.75 for every model.
5. `src/tools/validate_dataset.py`: add the id to `PROFILES` so it can be validated.
6. Add the id to a job YAML under `benchmarks:`.
7. Dashboard radar chart: the dashboard now builds its domains from the selected runs (with the old five listed
   first), so a new domain appears automatically once a run contains it.

## 6. Rules for `kyc_india`, `rbi_regulations`, `image_processing`

Each of these three datasets must end up with **100 questions**, with these rules:

- **`source` on every question**, naming the document **and** the section or paragraph it came from, e.g. the full
  title of the document, its year or circular number, then `Section 12`, `Para 38(a)`, `Chapter VI`, `Annex II`.
  `RBI`, `KYC`, `RBI Master Direction` or `RBI KYC` alone are rejected. For `kyc_india` and `rbi_regulations` the
  documents are the RBI Master Direction on KYC and related RBI circulars (for `rbi_regulations`, the relevant RBI
  Master Directions/circulars for the topic). For `image_processing` there is no RBI document: name the textbook,
  standard or paper and the chapter/section. (Interpretation: the instructions say all three datasets are
  "grounded"; the RBI wording only makes sense for the two regulatory ones.)
- **`source_excerpt` on every question**: one or two sentences copied word for word from the source that support the
  correct answer (20 to 600 characters). The loader accepts it and the model never sees it.
- **No facts from memory.** If you cannot point to text you can read that supports the answer, do not include the
  question.
- **Exactly one correct answer.** The three wrong options must be wrong according to the same source, not
  "arguably also right". Avoid "all of the above" / "none of the above" / "both A and B" options.
- **Balanced answer positions.** Roughly 25 of each letter in 100 questions (the validator tolerates +/-10
  percentage points per letter, so 15 to 35 each).
- **No duplicates**: no repeated `question_id`, no repeated or near-identical question, no repeated option within a
  question.
- **`review: true`** on any question the writer is less than certain about. This means "a person must check this".
- Write and validate in **batches of 25**, fixing problems before starting the next batch.
- The three new datasets have been rebuilt to 100 rows with `source` and `source_excerpt` and all pass `--final` with
  their source text. Human checks cover 20/100 image rows and 22/100 rows in each of KYC and RBI; the remaining keys
  were written from the source but are not independently verified. Do not describe an unchecked key as independently
  verified, and quote a question only together with its source.

## 7. How to create the datasets (for the person writing them, and for her AI tool)

**An AI writing regulatory questions from memory is not acceptable.** Language models state outdated or invented
rules with confidence, and a wrong answer key silently corrupts every score that follows. An AI tool may only
help by turning text that a person has supplied into question drafts. The facts must come from the text.

1. **Get the real source text first.** Download the current RBI Master Direction on KYC and the related RBI
   circulars from the RBI website (rbi.org.in). Use the consolidated, up-to-date version and note its title, date
   or "updated up to" line and the URL in a notes file. For `image_processing`, take a textbook chapter or standard
   you can read. Keep these files **outside the repo**.
2. **Make plain-text copies** (copy from the PDF into `.txt` files) of the parts you will use. The validator can
   read them (step 7).
3. **Plan coverage and answer positions before writing.** List the sections you will cover and how many questions
   each gets (about 100 in total, spread out, not 30 from one section). Decide the correct letter for each
   question in advance from a shuffled list with 25 of each letter, then write the options to fit.
4. **Write each question from a passage in front of you.** One passage, one question. The correct option restates
   what the text says. The wrong options are plausible but contradicted by (or not supported by) the same text.
   Prefer questions whose answer does not change between amendments; if it can, put the version/date in `source`.
5. **Fill `source`** (full document title, year or circular number, section/paragraph) and **`source_excerpt`**
   (the supporting sentence, copied word for word). Use `topic` for a short label.
6. **If you are unsure, set `review: true`. If you cannot find supporting text, drop the question.**
7. **Validate every batch of 25**, pointing the validator at your text copy so it can confirm each excerpt really
   appears in the source:
   ```powershell
   python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --source-text C:\path\to\kyc_master_direction.txt
   ```
8. **Release check** when all 100 rows exist:
   ```powershell
   python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --final --source-text C:\path\to\source.txt
   ```
   `--final` additionally demands exactly 100 items. Do not use a dataset in a real run until this passes.
9. Do not paste model answers, benchmark scores or anything from a previous run into the dataset.

**Instruction to give an AI tool (adapt as needed):**

```
You will be given one passage of source text at a time, with its document title and section number. Write ONE
four-option multiple-choice question that can be answered using ONLY that passage. Do not use any knowledge from
outside the passage. If the passage does not support a clear question with exactly one correct option, reply
"NO QUESTION" instead. Output one JSON object per line with the fields question_id, topic, question, options (four
plain strings, no letter prefixes), answer (A-D), source (the document title and section I gave you),
source_excerpt (a sentence copied exactly from the passage), review (true if you are less than certain the answer
is the only correct one). The correct answer must be the letter I tell you. Never invent a section number.
```

The person still reads and checks the output; the tool only drafts.

## 8. The dataset validator

`src/tools/validate_dataset.py` has no dependencies beyond the repo itself. It checks format and consistency; it
**cannot** tell whether a question is true, nor whether a source says what the question claims.

```powershell
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --source-text path\to\source.txt
python src/tools/validate_dataset.py datasets/kyc_india/sample.jsonl --final
python src/tools/validate_dataset.py datasets/gpqa/sample.jsonl datasets/mmlu_pro/sample.jsonl   # several files at once
```

The benchmark type is taken from the folder name (`--type` overrides). Exit code 0 = no errors, 1 = errors.

| check | result |
|---|---|
| each line is valid JSON, file is UTF-8, no BOM | error |
| required fields per type; MCQ `options` count (4, or 10 for `mmlu_pro`); `answer` is one uppercase valid letter | error |
| options carrying their own "A." label; duplicate options inside a question | error |
| `question_id`/`task_id` unique in the file **and** across the other `datasets/*/sample.jsonl` | error |
| duplicate question text | error; near-identical questions: warning |
| `source` missing, empty, generic (`RBI`, `KYC`, ...), a placeholder, with no section/paragraph number, or without a specific document name | error (required for the three new datasets; checked whenever present elsewhere) |
| `source_excerpt` missing or outside 20-600 characters | error (same scope as `source`) |
| `source_excerpt` not found in the `--source-text` file(s) (comparison ignores case, spacing and punctuation) | error |
| `review` not a real boolean | error |
| answer-position balance (20 or more items: each letter within +/-10 points of equal, `--balance-tolerance` adjusts) | error; under 20 items, one letter above 50%: warning |
| correct option is the longest in more than 45% of questions | warning |
| whitespace around an option, "all/none of the above", unknown field names | warning |
| the real dataset loader can read the file and returns the same number of items | error |
| `--expect-count N`; `--final` = 100 items | error |

It prints the number of items, the answer-letter counts, **how many items have `review: true`**, then every warning
and error with its line number, and `RESULT: PASS` or `FAIL`.

What it will tell you about the current files (checked): every dataset passes the format checks.
`kyc_india`, `rbi_regulations` and `image_processing` have 100 items each and 25 of each letter; KYC and RBI have
22/100 `human_checked` and pass `--final` with their source text; image has 0/100 `review: true` and 20/100
`human_checked` and passes `--final`. `gpqa` and `truthfulqa` now have
`A=4 B=3 C=4 D=4` (rebalanced from `A=15`). `custom_byob` still has 2 of 2 answers `A` (a demo warning), and one
`mmlu_pro` option refers to other options (warning).

### Rebalancing script (applied to gpqa and truthfulqa)

`src/tools/rebalance_answers.py` reshuffles the options of multiple-choice questions and updates the `answer` letter
so the correct answers are spread evenly. It was applied to `gpqa` and `truthfulqa`; both now have
`A=4 B=3 C=4 D=4`, and the copies passed the validator before replacing the originals. The questions and keys
themselves remain unverified.

```powershell
python src/tools/rebalance_answers.py datasets/gpqa/sample.jsonl datasets/truthfulqa/sample.jsonl            # dry run
python src/tools/rebalance_answers.py datasets/gpqa/sample.jsonl --output-dir rebalanced                      # writes a copy
```

- Default is a **dry run**: it prints, per question, the old and new answer letter, the correct option's text and the
  option order before and after, then writes nothing. There is no in-place mode: with `--output-dir` it writes a
  copy to `<dir>/<dataset folder>/<file>`, and it refuses to write over its input.
- Only the order of `options` and the `answer` letter change; the correct option's text, the set of options and all
  other fields are checked to be identical. Questions that are already in their target position, and lines it does
  not change, stay byte-for-byte as they were (same CRLF line endings).
- Questions whose options refer to each other ("all of the above", "both A and B") or whose text mentions a lettered
  option are left alone and listed. The shuffle is seeded (`--seed`, default 42), so the same command always gives
  the same result.
- What the applied run did (checked): each file went from `A=15 B=0 C=0 D=0` to `A=4 B=3 C=4 D=4`, with 11 of 15
  questions reordered and 4 unchanged, none left alone. Both replaced files pass the validator.
- Scores from before and after the rebalancing are **not comparable**, and it did nothing about whether the questions
  and keys themselves are right (they are unverified).

## 9. How answers are scored

Per reply, in this order (`src/engine/answer_scoring.py`):

1. **API error** (network failure, HTTP error, timeout, or a 429/5xx that survived all retries): scored incorrect with
   `error: api_error: ...`. Nothing is extracted from the error text.
2. **Truncated**: if the provider says it stopped because of the token limit (`finish_reason` of `length`,
   `max_tokens` or `max_output_tokens`), the reply is scored incorrect with `error: truncated` and the extractor is
   never run, **even if an answer happens to appear**.
3. **Reasoning text is stripped**: `<think>...</think>` (also `<thinking>` and `<reasoning>`) blocks are removed
   before any answer is looked for. A closing tag without an opening tag keeps only the text after it. An opening
   tag that is never closed means the model never finished thinking: no answer, `error: no_answer_found`. If a
   provider returns the thinking in a separate field it is stored as `reasoning_text` for audit and never scored.
   The full raw reply is always kept, and `scored_text` records what was actually scored.
4. **Extraction** (on the stripped text only):
   - Multiple choice: the **last** explicit marker wins: `Answer: C`, `The (correct) answer is C`, `Option C is
     correct`, `\boxed{C}`, `**C**`; otherwise a first or last line that is only an answer such as `C` or `C. text`.
     The letter must be uppercase and one of the benchmark's valid letters. Letters inside ordinary prose ("a", "I")
     are never taken as an answer.
   - Math (`gsm8k`, `math_500`; code in `src/scoring/math_answers.py`). **How the final answer is found; the first rule
     that finds a value wins:**
     1. the last `\boxed{...}` (nested braces are fine);
     2. the last `#### value`;
     3. the text after the **last** explicit label: `Answer:`, `Final answer`, `The answer is`, `Conclusion:` or a
        `### Conclusion` heading. Inside it: a value written right after the label; else a **bold** value; else the value
        after the last `=` or after a cue verb ("is", "has", "earns", "costs", "simplifies to", ...), whichever comes
        later in the sentence, except that an `=` inside a trailing side clause ("..., which occurs at x = 2") never
        overrides the main statement; else the first value (the last one if the sentence says "not 12; it is 13");
     4. with no label: the last three sentences of the last three paragraphs, with the same bold / `=` / cue rules, then
        the last value of the sentence. Closing chatter ("Hope this helps", "let me know if ...") is skipped;
     5. otherwise nothing: `no_answer_found`. **There is no "last number anywhere in the reply" fallback**, because a
        number from the working must not be mistaken for the answer.

     **Ambiguity is never guessed.** A hedge between two different values ("Answer: 3 or 4", "the answer is 6 or 7") and
     a sentence that states two different results ("Answer: 40 minus 1, which is 39") give `no_answer_found`, not the first
     value, so a hedge that happens to contain the key is not credited.

     **How two answers are compared.** LaTeX is flattened first (`\frac{1}{6}` becomes `1/6`, `9\sqrt{3}` and `9√3`
     become `9*sqrt(3)`; `\text{..}`, `\(..\)`, `\[..\]`, `$$` are removed). Currency signs, thousands commas (`1,680`,
     `\$1{,}680`), `%` and a unit written after the number ("45 pencils left") are dropped. Two answers are equal when
     their normalised text is equal or when both are the same number (integers, decimals, fractions `a/b`, and
     `k*sqrt(n)`; tolerance 1e-5). **Units are never converted**: 0.5 (hours) is not 30 (minutes), 25% is not 0.25, 0.15
     (km) is not 150 (m); a missing or extra sign, digit or decimal point is a different answer. The math scorer only
     understands the number forms listed; any other expression (for example `x^2+1`) is compared as plain text.

     **Known limitations of the math scorer (all pinned by tests):**
     - In a reply with no label at all, the last sentences are read, and a number in the final sentence is taken as the
       answer even if it is incidental ("The problem has 3 parts and I cannot solve all of them." gives 3). It only
       produces a wrong "right" if the key happens to equal that number. Making this stricter was tried and loses
       genuinely correct unlabelled replies, so it was kept.
     - The expected verdicts of the 118 stored replies in `tests/fixtures/live_math_replies.json` were hand-read for the
       replies whose verdict changed and for a random sample of the unchanged ones; the rest are the scorer's own
       output, so they only show that nothing changed unexpectedly, not that every one is right.
     - The 54 curated fixtures in `tests/fixtures/math_cases.json` include 32 "must stay wrong" guards (a correct
       intermediate number followed by a wrong final answer, the key appearing only in the working or in the question,
       a unit that changes the value, sign / digit / decimal-point errors, hedges, wrong fractions and radicals). A test
       also makes the scorer deliberately lenient in four ways and checks that these guards then fail.
   - HumanEval: Python is taken from a fenced code block (the last one that defines a function), or from the first
     code-looking line, or from a bare indented body (attached to the original `prompt`); imports from the prompt are
     added. The code then runs in a separate process against `test` with a 3-second limit. Note this is a timeout,
     not a security sandbox.
5. **Nothing extractable**: incorrect with `error: no_answer_found`.

Truncated, no-answer and api_error items are **counted as incorrect** (not dropped, so the paired McNemar test stays
aligned), flagged on the item, counted per benchmark and per run, and shown in the run table, the `compare` panel and
the report. A large truncated count means `max_tokens` is too small for that model, not that the model is wrong; a
large api_error count means the provider failed (rate limit, timeout, model not available), not that the model is wrong.

**UNRELIABLE runs.** A run is marked **UNRELIABLE** when *more than 10%* of its items are `api_error`, `truncated` or
`no_answer_found` (exactly 10% is still fine; the three kinds are added together). The mark is stored in the run file
(`unreliable`, `unreliable_fraction`), printed in the run summary after `eval run`, shown in the `compare` and `gate`
panels, and written to the report header with a warning banner. An UNRELIABLE run's accuracy is not a fair measure of
the model: find the cause (rate limit, too-small `max_tokens`, a model that is not available) and rerun.

**UNRELIABLE benchmarks (per-benchmark flag).** The run-level mark can hide failures that cluster in one benchmark (a
timeout storm on one dataset can leave the run under 10%). So each benchmark is judged on its own too: if *more than
20%* of **that benchmark's** items are `api_error`, `truncated` or `no_answer_found` (exactly 20% is fine; for 15 items,
3 is fine and 4 is flagged), the benchmark is marked **UNRELIABLE**, **even when the run as a whole is under 10%**. It is
shown in the `Reliability` column of the run table after `eval run` (with the percentage) plus an `UNRELIABLE
BENCHMARK(S): ...` line, in the `compare` panel and its table, and in the report (a banner and a `Reliability` column in
the benchmark breakdown). The `gate` panel shows the run-level mark only. The run file stores `unreliable` and
`unreliable_fraction` on each benchmark summary and `unreliable_benchmarks` on the run. The threshold is
`UNRELIABLE_BENCHMARK_THRESHOLD` in `src/storage/schema.py`. A failed or incomplete run reports no unreliable benchmarks
(it is reported as FAILED / INCOMPLETE instead). A flagged benchmark's accuracy is not a fair measure of the model.

## 9a. Scorer version and offline re-scoring

- Every run file has a `scorer_version` (`SCORER_VERSION` in `src/scoring/version.py`; a changelog is in that file). It is
  printed after `eval run`, shown in the `compare` / `gate` panels and in the report header. **Whoever changes how an
  answer is extracted or compared must bump it**, otherwise two runs scored differently look the same. Run files from
  before the field existed load as `unversioned`. `compare` and the report warn when the two runs have different
  versions ("scores are not directly comparable").
- `src/tools/rescore_run.py <RUN_ID> ...` re-scores the **stored replies** of a COMPLETED run with the current scorer. It
  makes **no model call** and **never modifies the original file**. It writes a new file `<RUN_ID>_rescored_v<N>.json`
  with `rescored_from` (the original id), `rescored_at` and the new `scorer_version`; the job name, mode, timestamp and
  all latencies/tokens are kept (so a stand-in run stays labelled STAND-IN), and the description says it was re-scored.
  The new file is JSON only: it is not in `nel list` or the leaderboard, but its id works with `compare` and `gate`, and
  the report header says RESCORED. Items that were `api_error`, `truncated` or `no_answer_found` go through the same code as
  in a live run, so their verdicts cannot change for those reasons; **HumanEval items are kept exactly as stored**
  (re-running code needs the original tests, which a run file does not hold). The tool refuses failed or incomplete runs
  and refuses to overwrite any file that is not itself a re-scored run. `--report-md` / `--report-json` write the
  before/after accuracy per benchmark and every item whose verdict or extracted answer changed.
- Re-scored copies of the four stand-in runs (pairs 2 and 3) were made with scorer v2. They are pipeline validation like
  the originals. Their scores are deliberately not recorded in this file.

## 9b. Incremental saving and INCOMPLETE runs

- After **every finished benchmark except the last**, the runner hands the partial run to the CLI, which writes it to
  `runs/<run_id>.json` with `status: "incomplete"` and a reason that names the benchmarks done and the ones missing. If the
  run then crashes or is interrupted (Ctrl-C included), the finished benchmarks are kept, and the CLI prints
  `RUN INTERRUPTED (...)` with the path of the INCOMPLETE record. A crash inside the *first* benchmark leaves nothing to
  keep. The save happens between benchmarks, not between items, so the benchmark that was running when the crash
  happened is lost.
- An INCOMPLETE record is JSON only: it is **not** in the run list or leaderboard. `nel list` shows it in a separate
  "INCOMPLETE or FAILED run records" table (with how many benchmarks of the plan were saved). **`compare` and `gate`
  refuse it** ("run ... is INCOMPLETE and cannot be used here"), and the report banner would say INCOMPLETE if it were
  ever passed in. When the run finishes normally, the final record has the same run id, overwrites the checkpoint, and is
  registered. A rerun after a crash is a new run with a new id; the old INCOMPLETE record stays until someone deletes it.
- Run states: `completed`, `failed` (stopped by the early stop, section 3) and `incomplete`. Only `completed` runs can be
  compared or gated.
- Not tested live: all of this was tested with mocked replies and a simulated crash, not with a real interrupted run.

## 10. Model configuration

Each model is a block in `configs/models_config.yaml`:

```yaml
vendor/model-name:
  id: "vendor/model-name"          # exact id the provider uses
  name: "Display Name"
  provider: "mock"                 # mock | nvidia_nim | groq | openrouter | openai | ollama | auto
  api_base: "https://integrate.api.nvidia.com/v1"   # must be the provider's own host (ollama: any)
  api_key_env: "NVIDIA_API_KEY"    # optional; defaults to the provider's own variable; may not name another provider's
  timeout_seconds: 120             # optional; default 120
  requests_per_minute: 20          # optional; default 20 for live providers (see "Rate limits" in section 3)
  params:
    temperature: 0.1
    max_tokens: 1024               # the reply cap; reasoning models need far more room
    top_p: 0.95
  cost_per_1m_input: 0.15          # USD, used for the cost estimate
  cost_per_1m_output: 0.15
  context_window: 131072           # display only
  strengths: ["..."]               # display only
```

An auto model names the requested ID at the top and gives one or more routes. The route's `id` is what the
provider is actually asked for; `substitute` and `note` record a difference from the requested model:

```yaml
vendor/requested-model:
  id: "vendor/requested-model"     # stays in the run as model_id
  name: "Display Name"
  provider: "auto"
  routes:
    openrouter:
      id: "vendor/served-model"    # recorded as served_model_id
      substitute: true             # recorded as model_substituted
      note: "The exact variant is not hosted."   # recorded as route_note
  params:
    max_tokens: 4096
  cost_per_1m_input: 0.15
  cost_per_1m_output: 0.15
```

Auto checks routes in the order NIM, Groq, OpenRouter, OpenAI and uses the first one whose key is present. It
never falls back to the simulator. The run artifact records the resolved provider (never `"auto"`), the
requested and served IDs, the substitute flag and the note.

To add a model: add a block, then list its id under `models:` in a job file. To make it simulated-only-capable you
may add a profile to `MODEL_PROFILES` in `src/gateway/mock_models.py`; **without one the simulator imitates the
Llama-3.1-70B profile**, so simulated numbers for it say nothing about that model.

`max_tokens`, `timeout_seconds` and `requests_per_minute` per model are used as written. The `execution:` blocks
inside the job files (`concurrency`, `timeout_seconds`, `retry_attempts`, `max_tokens`) and the `quality_gate:` blocks
are **not read by any code**: retries are fixed in the client (section 3) and the runs are sequential.

**Availability on NVIDIA NIM** (checked by calling the NIM model-list endpoint, 81 models returned, during this
project, October 2026):

| configured id | used by | status |
|---|---|---|
| `meta/llama-3.1-8b-instruct` | pair 1 | **unavailable: a real call returned HTTP 410 "reached its end of life on 2026-08-26"** |
| `mistralai/ministral-3-8b-instruct-2512` | pair 1 | not in the NIM model list |
| `mistralai/ministral-3-14b-reasoning-2512` | pair 2 | not in the NIM model list |
| `meta/llama-4-scout-17b-16e-instruct` | pair 2 | not in the NIM model list |
| `mistralai/ministral-3-8b-reasoning-2512` | pair 3 | not in the NIM model list |
| `qwen/qwen3.5-8b` | pair 3 | not in the NIM model list |
| `meta/llama-3.1-70b-instruct`, `qwen/qwen-2.5-72b-instruct`, `mistralai/mixtral-8x7b-instruct`, `deepseek/deepseek-r1-distill-llama-70b` | not in a pair | not in the NIM model list |
| `openai/gpt-4o-mini` | not in a pair | not in the NIM model list (it is an OpenAI model; it would use provider `openai`) |

Only the Llama-3.1-8B retirement was confirmed by an actual call; for the others, absence from the list is the
evidence. The list changes over time: re-check before any live run. `UnifiedModelClient.list_models()` does it
(it makes one GET call, so only run it when live calls are allowed). The list contained no Ministral and no Qwen
models at all. The auto routes in section 3a are the substitute plan for OpenRouter; on the free NIM/Groq tiers
there is still no callable equivalent for most of the real models (section 11). The request itself names Hugging
Face-style IDs (section 1a), which are a different naming scheme from the NIM IDs in this table.

### Being in a provider's model list does not mean the model can be called

Checked with real calls (stand-in work, October 2026):

| provider | model | result |
|---|---|---|
| NIM | `mistralai/mistral-7b-instruct-v0.3` | in the list, but the call returned **HTTP 404 "Function ... Not found for account ..."** |
| NIM | `nv-mistralai/mistral-nemo-12b-instruct` | in the list, same 404 |
| NIM | `nvidia/mistral-nemo-minitron-8b-8k-instruct` | in the list, same 404 |
| NIM | `ibm/granite-3.0-8b-instruct` | in the list, same 404 |
| NIM | `nvidia/llama-3.1-nemotron-70b-instruct` | in the list, same 404 |
| NIM | `google/gemma-3-12b-it` | in the list, same 404 |
| NIM | `zyphra/zamba2-7b-instruct` | in the list, same 404 |
| NIM | `aisingapore/sea-lion-7b-instruct` | in the list, same 404 |
| NIM | `meta/llama-3.2-90b-vision-instruct` | in the list; **one call timed out at 120 s**. Inconclusive: it is not a 404, but it was not shown to work |
| NIM | `meta/llama-3.2-11b-vision-instruct` | worked (10 `mmlu_pro` items). One call hit the 120 s read timeout |
| NIM | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | worked (1 math question: 371 completion tokens, 8.6 s) |
| Groq | `openai/gpt-oss-20b` | worked (10 `mmlu_pro` items, no errors) |
| Groq | `qwen/qwen3.8-27b` | worked (1 math question: 790 completion tokens, 1.9 s) |

So **every model has to be tried with one real call** before it goes into a config; the list only narrows the
candidates. Only these thirteen were tried. Of the 11 NIM models tried, **2 worked**, 8 returned 404 and 1 timed out;
both Groq models tried worked. The other listed models are unverified.

What each list contains (queried October 2026):
- **NIM**: 81 models (the same 81 as an earlier query). Each entry has only `id`, `object`, `created`, `owned_by`: no
  capability, size or modality fields. There is no Llama 3.1 8B/70B, no Llama 4, no Qwen and no Ministral.
- **Groq**: 11 models. Each entry also has `active`, `context_window`, `max_completion_tokens`, `input_modalities`,
  `supported_features` and often `hugging_face_id`. Chat-capable: `allam-2-7b`, `openai/gpt-oss-120b`,
  `openai/gpt-oss-20b`, `openai/gpt-oss-safeguard-20b` (a safety classifier) and `qwen/qwen3.8-27b` (text and image
  input). The rest are speech-to-text, text-to-speech and prompt-guard models. No Llama chat model, no Mistral.

**How to tell that a model is a reasoning model.** On **Groq** the list entry says so: `supported_features` contains
`"reasoning"` (true for `gpt-oss-120b`, `gpt-oss-20b`, `gpt-oss-safeguard-20b`, `qwen3.8-27b`). On **NIM** no field says
so; the only hint is the model name (for example `nemotron-3-nano-omni-30b-a3b-reasoning`), which is unconfirmed.
The reliable evidence is the reply itself.

**What a reasoning reply looks like** (observed on Groq `openai/gpt-oss-20b`): `choices[0].message.content` holds only
the final answer; the thinking is in a separate field, `choices[0].message.reasoning` (no inline `<think>` tags);
`finish_reason` was `stop`; `usage.completion_tokens` already **includes** the thinking tokens
(`usage.completion_tokens_details.reasoning_tokens`), so `max_tokens` has to cover thinking plus answer. The client
stores the separate field as `reasoning_text` and scores only `content`. **On NIM** (`nemotron-3-nano-omni-...-reasoning`)
the thinking came back in `message.reasoning_content` instead, again with the answer alone in `content`; its usage block gave no
`reasoning_tokens` count. **Groq `qwen/qwen3.8-27b`** returned only `content` (a worked solution, no
separate thinking field and no `<think>` tags), even though the Groq list flags it as reasoning-capable.

**Measured sizes used for the stand-in settings** (a very small sample, all easy questions; treat as provisional):
`gpt-oss-20b`: 42 to 472 completion tokens per item over 10 items (mean 132, thinking included, about 1,000 tokens/s);
`nemotron-3-nano-omni`: 371 tokens in 8.6 s (about 43 tokens/s end to end); `qwen3.8-27b`: 790 tokens in 1.9 s (about
416 tokens/s). Groq's token-per-minute limit is 8000 in the headers; a call with `max_tokens` 4096 left 7878 tokens in the
bucket, so `max_tokens` is **not** reserved against it.

**Live limits seen**: Groq's reply headers report limits of 1000 requests and 8000 tokens (the header does not say over
which period, so treat the periods as unverified). Groq answered in about 1 to 2 seconds per call. NIM's free tier was
slow and uneven: the 11B Llama took a median of about 20 s per successful call and up to about 115 s, and 9 of its 120
items in one run timed out twice at 120 s (5 of them on GPQA); the Nemotron reasoning model took a median of about 4 s
but got HTTP 503 on many calls. A full 120-item run took about 20 minutes for Nemotron and about 130 minutes for the
Llama, so budget hours, not minutes, for NIM runs.

## 11. Stand-in and simulated runs

**Live vs simulated - the one rule to remember.** A run is **simulated only when no API call is made**
(`provider: mock`). Any run that reaches a provider over the network is a **live** run, including a stand-in
run. So a stand-in's numbers are genuine live results **for the stand-in model**; they are simply not results
for the required models the project is meant to compare. "Pipeline validation" describes *what the numbers are
for*, not *how they were produced* - do not call a stand-in run "simulated".

- **Simulated** (`provider: mock`): a fake model that answers correctly with a fixed per-benchmark probability. It
  exists only to test the software. Its scores are not findings about anything. Since the simulator was fixed it is
  repeatable: the same seed and the same items give the same output in every process. Several models have
  no simulator profile and the three newest benchmarks have no per-model accuracy setting (see section 12).
- **Stand-in** runs use cheaper look-alike models, available on the provider, to test the live pipeline end to end
  (real calls, real reply formats, real truncation behaviour) when the real models cannot be called. They are
  **live** runs, and are **pipeline validation only**: they must **never** be described as results for the real
  models (they are results for the stand-in models themselves).
- **Current direction: free-tier stand-ins by default; a paid route only on request.** Stand-ins are free-tier
  NVIDIA NIM and Groq models. The real pair configs (`configs/eval_pair*.yaml`) are unchanged in which models they
  name, but those six models now use `provider: "auto"` with an OpenRouter route (section 3a); the route is paid and
  does nothing until `OPENROUTER_API_KEY` is set. Stand-in pairs are meant to mirror the *roles* of the real pairs
  (a small Mistral-family instruct model against a Llama of about 8B; a mid-size model against Llama 4 Scout or the
  nearest Llama; a reasoning model against a Qwen model). Where a provider has no equivalent for a role, the
  proposal says so and names the nearest model instead.
- **Substitute route** (only `provider: "auto"`): a live route whose served model is not the exact requested model.
  The run records both IDs and `model_substituted: true`; its scores are for the served model, not the requested one.
  `nel list`, the run line, the `compare` panel, the report header and the dashboard all mark it `[SUBSTITUTE]` or
  show the served model. It is a different case from a stand-in (a stand-in is a separate model block, not a
  route) and from simulated (a substitute route makes real API calls).
- **The three stand-in pairs** (configs `configs/standin_pair1.yaml`, `standin_pair2.yaml`, `standin_pair3.yaml`; each
  runs the same 9 benchmarks as the real pairs, 15 items each). Their model blocks are in `models_config.yaml`,
  marked stand-in only. **Pairs 2 and 3 have each been run once** (8 benchmarks, no HumanEval, as pipeline
  validation); **pair 1 has not been and cannot be run** (below).

  **Required models vs the stand-ins used now.** The required models (section 1a) cannot be called on the free
  providers, so each pair runs a stand-in that mirrors its role. The mapping is:

  | pair | required model A -> stand-in | required model B -> stand-in | why the real one is not used |
  |---|---|---|---|
  | 1 | `Ministral-3-8B-Instruct-2512` -> `google/gemma-3-12b-it` (NIM) | `Llama-3.1-8B-Instruct` -> `meta/llama-3.2-11b-vision-instruct` (NIM) | Ministral absent from NIM/Groq; Llama-3.1-8B retired (HTTP 410) |
  | 2 | `Ministral-3-14B-Reasoning-2512` -> `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (NIM) | `Llama-4-Scout-17B-16E` -> `meta/llama-3.2-11b-vision-instruct` (NIM) | both absent from NIM/Groq |
  | 3 | `Ministral-3-8B-Reasoning-2512` -> `openai/gpt-oss-20b` (Groq) | `Qwen3.5` (size unstated) -> `qwen/qwen3.8-27b` (Groq) | both absent from NIM/Groq; Qwen3.8 is a different version/size |

  The stand-in pair table (what actually runs today):

  | pair | role it mirrors | model A | model B | verification |
  |---|---|---|---|---|
  | 1 | small instruct vs Llama ~8B | `google/gemma-3-12b-it` (NIM) | `meta/llama-3.2-11b-vision-instruct` (NIM) | **NOT RUNNABLE, skipped by decision**: A returns 404 |
  | 2 | mid-size reasoning vs nearest Llama to Scout | `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` (NIM) | `meta/llama-3.2-11b-vision-instruct` (NIM) | both worked |
  | 3 | reasoning vs Qwen | `openai/gpt-oss-20b` (Groq) | `qwen/qwen3.8-27b` (Groq) | both worked |

  **Pair 1 is not runnable and has been skipped on purpose.** There is **no callable small Mistral-family instruct
  model on the free tiers**: the three Mistral-family ones in NVIDIA's list (`mistral-7b-instruct-v0.3`,
  `mistral-nemo-12b-instruct`, `mistral-nemo-minitron-8b-8k-instruct`) all returned HTTP 404, Groq has no Mistral
  model, and every substitute small instruct model tried on NIM (Granite 8B, Gemma 3 12B, Zamba2 7B, SEA-LION 7B)
  returned 404 too. A Groq 7B model (`allam-2-7b`) was considered and **deliberately not added**.
  `configs/standin_pair1.yaml` stays in the repo, marked in its header as not runnable; running it would stop after 5
  items with a FAILED run (the early stop) and would not produce a comparison.

  Other roles with **no equivalent** on the free providers: a Llama of about 8B, Llama 4 Scout (so pair 2 uses the same
  11B Llama as pair 1 would have; the 90B Llama timed out once and the 70B Nemotron fallback returned 404), and Qwen3.5
  (Qwen3.8 at 27B is a different version and size). The pairs that do run validate the pipeline; they do not reproduce
  the real comparisons.
- **Settings of the reasoning stand-ins**, from the measured sizes in section 10 (provisional, from one to ten easy
  items each): `max_tokens` **4096** for all three (4 x the largest measured reply, 790, rounded up to a multiple of
  1024). Timeouts: **240 s** for the NIM model (2.5 x the 95 s that 4096 tokens would take at 43 tokens/s, because
  NIM showed slow outliers), **60 s** for the two Groq models (the same 4096 tokens take about 10 s). To stay under
  Groq's 8000 tokens per minute: `requests_per_minute` **10** for `gpt-oss-20b` and **6** for `qwen3.8-27b` (about
  600 and 1000 tokens per request).
- Naming rules for stand-in runs:
  - Config files are named `configs/standin_*.yaml`. A test fails if a `standin_*` file has a `job_name` that
    does not start with `standin_`, or a `description` without the words "pipeline validation", or if a
    `configs/eval_*.yaml` file uses a `standin_` job name.
  - Template (format only):
    ```yaml
    job_name: "standin_<short_name>_pipeline_validation"
    description: "PIPELINE VALIDATION ONLY. Stand-in models, not the models under study. Not results for <real models>."
    seed: 42
    limit_per_benchmark: 10
    models:
      - "<stand-in model id>"
    benchmarks:
      - "mmlu_pro"
    ```
  - Give stand-in models their own blocks in `models_config.yaml` and a display `name:` that does not look like
    the real model.
- **How a stand-in run is labelled.** The run file records the `job_name`. `nel list`, the line printed when a run
  finishes, the `compare` panel and the report header all show it, and when a job name starts with `standin_` they
  also show a **STAND-IN RUN: pipeline validation only** banner (the report banner says it is not a result for the
  real models). `eval run` refuses to start if the config file is named `standin_*` but the job name does not start
  with `standin_` (so `--job-name` cannot strip the label).
- **Limits of that labelling:** it works from the job-name prefix alone. A stand-in run started from a config that
  is not named `standin_*`, with an ordinary job name, is **not** marked. The run mode still reads **LIVE** for a
  stand-in run made with real calls, because `run_mode` only distinguishes simulated from live. The dashboard now
  shows the mode, job name and a stand-in warning on its leaderboard and comparison tabs. Anyone sharing such a
  report must still describe it as pipeline validation in the text around it.

## 11a. Verified on live calls

What real calls have shown so far (October 2026). "Seen" means a real reply was received and read by the pipeline.
Only shapes and behaviours are listed; **scores are deliberately left out** (the stand-in runs are pipeline validation).

**Model lists**
- NIM `GET /v1/models`: each entry has only `id`, `object`, `created`, `owned_by`. Being listed does not mean callable.
- Groq `GET /openai/v1/models`: the same four fields plus `active`, `context_window`, `max_completion_tokens`,
  `input_modalities`, `supported_features` (a model is flagged reasoning-capable when it contains `"reasoning"`) and
  often `hugging_face_id`.

**Chat replies** (`POST .../chat/completions`; `finish_reason` was `stop` in every successful reply of every model below)

| provider / model | thinking | answer | other things seen |
|---|---|---|---|
| NIM `meta/llama-3.2-11b-vision-instruct` (not a reasoning model) | none: `message.reasoning_content` is present but always null | `message.content`, a single letter for multiple choice | message also has `refusal`, `annotations`, `audio`, `function_call`, `tool_calls`; top level has `prompt_logprobs`, `kv_transfer_params`; `usage` = prompt/completion/total tokens plus a null `prompt_tokens_details` |
| NIM `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` | `message.reasoning_content` (present in every reply) | `message.content` only (1 character for multiple choice, up to a few hundred for math) | message keys are just `content`, `role`, `reasoning_content`; top level adds `nvext`; `usage` has **no** reasoning-token count; replies reached 1651 completion tokens (thinking included) against a limit of 4096 |
| Groq `openai/gpt-oss-20b` | `message.reasoning` | `message.content` only | `usage.completion_tokens_details.reasoning_tokens` is present and included in `completion_tokens`; top level adds `usage_breakdown` and `x_groq`; response headers carry `x-ratelimit-limit/remaining/reset` for requests and tokens |
| Groq `qwen/qwen3.8-27b` | no separate field | `message.content`, a worked solution with markdown and LaTeX | message keys are just `content`, `role` |

- **No thinking leaked into scored text.** For the Nemotron run (116 replies) none had `<think>` tags in the answer,
  none had thinking text inside the answer, and nothing needed stripping: the pipeline scored `content` only and kept
  the separate thinking field as `reasoning_text`.
- **Not seen live:** `finish_reason` of `length` (nothing was truncated), inline `<think>` tags from a real model, any
  401 or 403, and anything from OpenRouter, OpenAI or Ollama (the `openrouter` provider has never been called).

**Error and failure shapes**
- NIM 404: JSON `{"status":404,"title":"Not Found","detail":"Function '<id>': Not found for account '<id>'"}`.
- NIM 410: JSON naming the model and its end-of-life date.
- NIM 503 (Service Unavailable) was frequent for the reasoning model, and read timeouts (no HTTP status) were frequent
  for the 11B Llama.
- **Behaviours of the client confirmed live:**
  - HTTP 503 retries on NIM: in one 120-item run, 54 items needed more than one request, 7 used all 5, and 4 ended as
    `api_error` (HTTP 503).
  - Retries on Groq: 7 items of one run needed 2 or 3 requests. The status of those intermediate failures was not
    recorded (the `attempt_log` field, section 3, was added afterwards), so they are presumed rate-limit replies. Runs
    made from now on record every status.
  - The single read-timeout retry on NIM: 25 of 120 items in one run needed more than one request, and 9 items timed
    out twice and ended as `api_error` (one successful item used 6 requests, the most seen).
  - Successful NIM calls for the 11B Llama took up to about 115 s (median about 20 s), close to the 120 s timeout.
  - The UNRELIABLE check ran on real runs without triggering (3.3% and 7.5% of items flagged; the limit is 10%).
- **Not triggered live:** the early stop for a dead model (it has only been tested with mocked replies; the 404 checks
  were single calls made outside a run).

## 12. Current status and known limitations

**Works and is tested** (402 tests passed at the time of writing, with both `pytest` and `unittest`; everything below was
tested with the simulator or mocked replies only; the newest parts are the math scorer v2 with its 54 curated fixtures and 118
stored real replies, the scorer version stamp, the per-request `attempt_log`, the per-benchmark UNRELIABLE flag, incremental
saving with INCOMPLETE runs refused by `compare` / `gate` (tested with a simulated crash), the offline re-score tool,
`provider: "auto"` route resolution with substitute labels, and the dashboard's background job thread): the simulator
is deterministic and honours the seed; live providers fail loudly without a key; run mode and job name are recorded
and shown, with a stand-in banner; reasoning text is stripped and truncation/no-answer/api_error is flagged and
counted; the UNRELIABLE mark (more than 10% flagged items) and the FAILED state with its early stop (first 5 items all
401/403/404/410); retries on 429/5xx with backoff and `Retry-After`, a single retry of a read timeout, and a per-model
throttle (tested with mocked replies and a fake clock); HumanEval code extraction; math answer extraction (see the
math limitations in section 9 and below); per-provider keys and host pinning for `nvidia_nim`, `groq`, `openrouter` and
`openai` (a sweep fails if any key could reach another host); auto routes pick the first key present and record the
served ID and substitution, with the cache keyed by the requested model; `.env` loading; the dataset validator; the
rebalancing script (dry run and copy writer; applied to `gpqa` and `truthfulqa`, never in place). The dashboard starts a run in a daemon thread,
shows a polling status panel every 2 seconds, keeps the finished result after a reload, and disables Start while a
job is running. This was checked with Streamlit's `AppTest` and a 1-item simulated run.

**Verified against the real services**: see section 11a for the response shapes and the client behaviours seen. In
short: the NIM and Groq keys are accepted; both model lists work; the pipeline read replies from four different models
(one with `reasoning_content`, one with `reasoning`, two with no thinking field); two complete 8-benchmark stand-in
runs (pair 3 on Groq and pair 2 on NIM) finished end to end, each reporting LIVE, the job name and the STAND-IN banner.
The `openrouter` provider has **never** been called, and the early stop has never fired on a real run.

**Not done / not verified:**
- **No result for any real model exists.** The stand-in runs of pairs 2 and 3 exist as run files in `runs/` and are
  pipeline validation only (their scores are deliberately not recorded in this file). All other saved runs are
  simulated, and the ones from before the simulator fix also came from the old non-repeatable simulator.
- **The math scorer used to under-score correct answers; scorer v2 fixes the causes that were found.** The stand-in runs
  showed that most GSM8K / MATH-500 items scored wrong had a correct final answer that the old extractor misread (a unit
  after the number, currency with thousands separators, a final line that is only bold text after a "Conclusion",
  LaTeX forms such as `\frac{1}{6}` against the key `1/6`). The rules in section 9 replace the old ones. **Run files saved
  before that (scorer `unversioned`) still hold the old verdicts for GSM8K and MATH-500: do not use their math numbers.**
  Re-score them (section 9a) or rerun. v2 was checked only on fixtures and on the stored stand-in replies, not on a new
  live run, and other reply styles (other models, harder problems) may still be misread. After re-scoring, the math
  items still marked wrong in the four stand-in runs were: `gsm_05` in four of them (the wording has since been fixed;
  those old run files keep the old verdicts), one genuinely wrong answer, and two failed API calls (`api_error`, not
  model errors); these were read by hand.
- **Failures cluster; the per-benchmark flag now shows it.** In one stand-in run a model timed out on a third of the
  items of one benchmark while the run as a whole stayed under the 10% mark. A benchmark with more than 20% flagged items is
  now marked UNRELIABLE on its own (section 9), but only runs made after that change carry the flag in the run file;
  older files get it when loaded (it is computed from the stored counts).
- **Dataset status.** `kyc_india` and `rbi_regulations` each have 100 items, 25 of each letter, and pass `--final`
  with their source text; 22/100 rows in each were actually human-checked. `image_processing` is rebuilt to 100 items
  with 25 of each letter and 100/100 excerpts matched, has 20/100 `human_checked`, and passes `--final` with its source
  text; most keys across the three new datasets were written from the source but are not independently verified. The
  older benchmarks still have 15 rows each.
- **`gpqa` and `truthfulqa` answer-position bias is fixed, but their content is still unverified.** Both files were
  rebalanced from `A=15 B=0 C=0 D=0` to `A=4 B=3 C=4 D=4`; questions and keys were not checked against any source, and
  each file has only 15 items. Do not present their scores as verified benchmark results; state these limitations next
  to the numbers. `rbi_regulations` and `image_processing` are now balanced at 25 of each letter, so the old lopsided
  15-row note no longer applies to them.
- **The real model pairs cannot run on NIM or Groq** because the models are not there (section 10). The six
  models now have OpenRouter routes via `provider: "auto"` (section 3a); OpenRouter's model list has the served IDs,
  but **no real call has been made**, so whether each route answers is unverified. Several routes are substitutes,
  not the exact requested variants. Stand-in pairs 2 and 3 have each been run once as 8-benchmark pipeline
  validation (their scores are deliberately not recorded here); pair 1 has not been run because its model A
  (`google/gemma-3-12b-it`) returned HTTP 404. The 90B Llama that would suit pair 2 better timed out once.
- **Reasoning replies seen so far**: thinking in `message.reasoning` (Groq gpt-oss), in `message.reasoning_content`
  (NIM Nemotron), or no separate field at all (Groq Qwen3.8). The inline `<think>` form is covered only by tests with
  made-up replies. The reasoning stand-ins' `max_tokens` (4096) and timeouts come from measurements on one to ten
  **easy** questions, so truncation or timeouts on harder items (competition math, code) are still possible. The real
  models' settings have not been measured at all.
- **Dead models are stopped early, but only for 401/403/404/410 on the first 5 items** (section 3). A model that fails
  in other ways (for example a model that works for a while and then starts returning errors) still runs to the end
  and is then marked UNRELIABLE. Make one real call to a model before a full run.
- Read timeouts are retried **once**; other network errors are not. On NIM's free tier a slow call can take over a
  minute, so a few `api_error` items from timeouts are possible on any run; check the api_error count and the
  UNRELIABLE mark before trusting a score.
- The simulator has profiles for six models only. Five newer models use the Llama-3.1-70B profile, and the three
  newest benchmarks have no per-model accuracy setting (0.75 for everyone), so simulated comparisons on them are
  noise.
- Other limits: runs are sequential (no concurrency); text-only questions (no vision, despite the
  "vision-ready" wording in some docs); the Streamlit dashboard shows mode, job name, served model, substitute and
  stand-in warnings and the truncated/no-answer/api-error counts, and its sample cap can be set to any number up to
  100 or to every row; a dashboard run continues in a background thread, and job status is held in that process only,
  so restarting Streamlit loses the live status panel (the saved run file remains). McNemar uses a continuity
  correction and there is no correction for many comparisons; with 10 to 15 items per benchmark the confidence
  intervals are very wide; the quality gate has only PASS/FAIL; `llm_judge.py` is unused; the HumanEval runner is a
  timeout, not a security sandbox; comparison pairs questions by `question_id` alone.
- **The older documents under `docs/` and the README predate these changes** and still describe the simulator as
  non-repeatable, say live providers silently fall back to the simulator, and quote results from the old simulator.
  Do not quote their numbers.

## 13. Open decisions

- **Whether the requested models will be hosted on a GPU or called through an API** (section 1a). This decides
  whether a new provider has to be written and which model IDs apply.
- Which models to compare, given that none of the three planned pairs is available on NIM; and which Qwen3.5 size
  is meant (the request does not say; the repo routes to `qwen/qwen3.5-9b`).
- The stand-in pairs (section 11): a replacement for pair 1's model A (every small instruct model tried on NIM
  returned 404), whether to retry `meta/llama-3.2-90b-vision-instruct` with a longer timeout for pair 2, and when
  pairs 1 and 2 may be run.
- Whether to pay for OpenRouter and run the real pairs through the auto routes (section 3a), and whether the
  substitute served models (Ministral 8B/14B, Qwen3.5-9B) are acceptable for the comparison.
- What the vision-model part of the request consists of (section 1a).
- What "features" means in the request (section 1a); the working assumption is the same benchmarks for each pair.
- Whether to keep `gpqa` and `truthfulqa` (15 items each, unverified keys) or drop them from the real comparisons;
  their answer-position bias has already been fixed.
- Whether to add a `--provider` option to `eval run` (today the only way to go live is editing the YAML).
- Token limits and timeouts for the *real* reasoning models, to be set from measured token counts after a small live probe.
