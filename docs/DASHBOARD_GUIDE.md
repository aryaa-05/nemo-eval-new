# Dashboard guide

This document explains every part of the Streamlit dashboard (`src/dashboard/app.py`)
and what each result means. It describes the dashboard only. The command-line tool is
documented in `CONTEXT.md`.

## 1. Starting the dashboard

Run from the repository root:

```powershell
python -m streamlit run src/dashboard/app.py
```

Open http://localhost:8501. If the port is busy, add `--server.port 8503`.
Only one Streamlit process should run on a port; two instances cause confusing results.

A run started from the page executes in a background thread, so the page stays
responsive and can be switched between tabs. The job status lives in that Streamlit
process only: if the process restarts, the live status panel is lost, but the saved run
file in `runs/` remains.

## 2. Run modes and labels

| Label | Meaning |
|---|---|
| `LIVE` | At least one real API call was made to a provider (NVIDIA NIM, Groq, OpenRouter, OpenAI or Ollama). |
| `SIMULATED` | The built-in mock provider answered; no API call was made. These numbers say nothing about a real model. |

Other things a run can record:

- **Served model / substitute route.** A model with `provider: auto` may be answered by a
  different model ID than the one requested. The run stores both: the requested
  `model_id` and the `served_model_id`. `model_substituted: true` means they differ.
  Scores are for the **served** model, not the exact requested variant.
- **Stand-in job names.** Job names that start with `standin_` mark stand-in runs:
  cheaper look-alike models used only to validate the pipeline. They are LIVE runs, but
  their scores are results for the stand-in model only and must never be reported as
  results for the models under study. The dashboard currently hides the stand-in banner
  warnings, but the job name still carries the prefix and the run files still record it.

## 3. Tabs

### 3.1 Leaderboard

Shows one row per `(model, provider, run mode)`. A simulated and a live run of the same
model are kept separate.

Top metrics: completed runs, live runs, simulated runs.

Filters:

- **Show run modes**: live, simulated.
- **Include stand-in runs**: include or hide runs whose job name starts with `standin_`.

Columns:

| Column | Meaning |
|---|---|
| Model requested | The model ID named in the configuration. |
| Provider | The provider that actually served the run (never `auto`; the resolved provider is stored). |
| Served / route | The model ID that answered, with `(SUBSTITUTE)` when it differs from the requested model. |
| Mode | LIVE or SIMULATED. |
| Latest job | Job name of the newest matching run. |
| Best accuracy | The highest overall accuracy among the matching runs. |
| 95% CI | Wilson 95% confidence interval of the run that achieved the best accuracy. |
| Mean latency (s) | Average of the per-run mean latency values. |
| Total tokens | Sum of prompt + completion tokens over the matching runs. |
| Est. cost (USD) | Sum of the cost estimates, using the per-1M-token prices in `configs/models_config.yaml`. |
| Runs | How many matching runs exist. |
| Latest run ID | The newest matching run. |

Charts:

- **Best accuracy by model and mode**: bar chart of the Best accuracy column.
- **Accuracy vs. latency**: scatter plot; bubble size is total tokens.

### 3.2 Radar & domains

Select up to several runs. Each axis is a benchmark **domain**; the value on an axis is
the average accuracy of the selected run's benchmarks that belong to that domain. A
domain with no selected benchmark shows 0. New datasets add their domain automatically
once they appear in a run.

**Benchmark detail** table (per run and benchmark):

| Column | Meaning |
|---|---|
| Accuracy | Correct items divided by all items in that benchmark for that run. |
| Items | Number of items in that benchmark for that run. |
| API errors | Items that failed with a network/HTTP/timeout error. Counted as incorrect. |
| Truncated | Items where the provider stopped at the token limit. Counted as incorrect. |
| No answer | Items where no answer could be extracted from the reply. Counted as incorrect. |

### 3.3 Model diff & flips

Compares two completed runs, matched **by `question_id`**. Select **Baseline - Run A**
and **Candidate - Run B**.

Top metrics:

| Metric | Meaning |
|---|---|
| Accuracy delta (B - A) | Candidate overall accuracy minus baseline overall accuracy, in percentage points. |
| Latency delta | Candidate mean latency minus baseline mean latency, in seconds. Lower is better (shown inverted). |
| B-only correct (flips) | Questions the candidate got right and the baseline got wrong. |
| A-only correct (flips) | Questions the baseline got right and the candidate got wrong. |

**McNemar paired test**: chi-squared statistic and p-value from the paired correct/incorrect
outcomes. `p < 0.05` is reported as statistically significant. The test uses a continuity
correction. There is no correction for comparing many benchmarks at once, so treat
per-benchmark p-values as exploratory.

**Benchmark breakdown**: accuracy for each side and the delta in percentage points.

**Flip inspector**: expand a question to see the full prompt, the ground-truth answer, and
both models' extracted answers and raw replies. Flips are the individual questions where
exactly one side was correct.

The comparison is refused unless both runs are `completed`.

### 3.4 Launch evaluation

| Control | Meaning |
|---|---|
| Model set | **Live stand-ins** (the free-tier stand-in models) or **All configured models** (everything in `configs/models_config.yaml`). |
| Model under test | The model block to run. The line below shows the provider, LIVE/SIMULATED, and whether the needed key is loaded. |
| Benchmarks | Which datasets to run. Each selected benchmark contributes up to the sample cap. |
| Use every row | Run every row of each selected dataset instead of a sample. |
| Samples per benchmark | How many rows to take from each dataset (1 to 100). |
| Prompting strategy | `simple` (question + options, answer directly) or `cot` (reason step by step, then `Answer: [OPTION]`). For HumanEval both use the same code instruction. |
| Job name | Written into the run file; automatically prefixed for stand-in models. |
| Start evaluation | Starts the run in a background thread. While a run is active the button reads **Run in progress...**. |

The status panel is rendered directly below the Start button. It polls every 2 seconds
and shows elapsed time, finished benchmarks, the run ID and, when finished, the overall
accuracy and saved file path. You can switch tabs while a run continues.

Notes:

- Runs are sequential; there is no concurrency.
- A missing provider key makes a live run fail. The Launch tab shows whether the key for
  the selected provider is loaded.
- A model that is not callable can stop early after 5 identical 401/403/404/410 errors;
  the run is then marked FAILED.
- HumanEval needs the code sandbox; each item has a 3-second execution timeout.

### 3.5 Reports

Select two runs and generate a Markdown or HTML comparison report. The report header
shows LIVE/SIMULATED, the requested and served model, substitute notes, job names, the
scorer version, and reliability banners. Download either format, or read the Markdown
preview on the page. Reports are refused unless both runs are completed.

## 4. What the numbers mean

### Accuracy and confidence interval

- **Accuracy** = correct items / all items. Incorrect answers, API errors, truncated
  replies and no-answer items all count as incorrect, so the denominator never changes.
- **95% CI (Wilson)** is the range in which the true accuracy would plausibly lie given
  the number of items. With 15-item benchmarks the interval is very wide; do not treat a
  few points of difference as real without the paired test.

### Latency, tokens and cost

- **Latency** is the time of the last successful HTTP request for an item. It does not
  include throttling waits or retry backoff. A provider timeout shows up as an API error,
  not as a large latency.
- **Tokens** are the provider's reported prompt + completion tokens. For reasoning models
  the completion count includes thinking tokens.
- **Cost** is an estimate from the configured per-1M-token prices, not a bill.

### Failure counts

| Count | Meaning | What to do |
|---|---|---|
| API errors | Network/HTTP/timeout failure after retries. | Check the provider status, rate limits and timeout setting; rerun. |
| Truncated | The reply hit the model's `max_tokens`. | Raise `max_tokens` for that model and rerun. |
| No answer | The reply contained no extractable answer. | Inspect the raw reply; the model may be refusing or reasoning without a final answer. |

### Reliability

- A **run** is UNRELIABLE when **more than 10%** of all its items are API errors,
  truncated or no-answer (exactly 10% is fine).
- A **benchmark** is UNRELIABLE when **more than 20%** of that benchmark's items are
  flagged (exactly 20% is fine).
- An UNRELIABLE accuracy is not a fair measure of the model. Find the cause and rerun.
  The dashboard does not display the reliability flags; use the CLI (`nel list`, `compare`,
  `gate`) or the generated report to see them.

### Run statuses

| Status | Meaning |
|---|---|
| `completed` | Finished normally. Usable for comparison and reports. |
| `failed` | Stopped by the early-stop rule for a dead model, or failed to start. |
| `incomplete` | Interrupted; finished benchmarks were saved as a checkpoint. Cannot be compared or used in a report. |

### Scorer version

Every run records the scorer version that produced its verdicts. If two compared runs
have different scorer versions, their scores were produced by different extraction rules
and are not directly comparable. The diff tab warns about this.

## 5. Benchmarks and datasets

| Benchmark | Type | Rows shipped | Notes |
|---|---|---|---|
| `mmlu_pro` | 10-option multiple choice | 15 | Small sample. |
| `gsm8k` | Free-answer math | 15 | Small sample. |
| `math_500` | Free-answer math | 15 | Small sample. |
| `gpqa` | 4-option multiple choice | 15 | Answer positions rebalanced; keys not independently verified. |
| `humaneval` | Code | 15 | Executed against hidden tests with a 3-second timeout. |
| `truthfulqa` | 4-option multiple choice | 15 | Answer positions rebalanced; keys not independently verified. |
| `custom_byob` | 4-option multiple choice | 2 | Demo only. |
| `kyc_india` | 4-option multiple choice | 100 | RBI-sourced; 22/100 rows human-checked. |
| `rbi_regulations` | 4-option multiple choice | 100 | RBI-sourced; 22/100 rows human-checked. |
| `image_processing` | 4-option multiple choice | 100 | Textbook-sourced; 20/100 rows human-checked. |

The older benchmarks have 15 rows each, so their confidence intervals are wide and their
keys are not independently verified. Most keys in the three 100-row datasets were written
from source text but only a subset was human-checked. State these limits next to any
numbers.

## 6. Presenting results honestly

- A SIMULATED run says nothing about a model; it only tests the software.
- A LIVE stand-in run (job name starting with `standin_`) is a real measurement of the
  **stand-in model**, not of the models the project is meant to compare.
- A substitute route measures the **served** model, not the exact requested variant.
- Do not put simulated, stand-in or substitute numbers in the same table as real-model
  results without labelling them.
- Do not describe unchecked answer keys as verified.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Start does nothing | The status panel appears below the button. If the button is disabled, a run is already in progress in this Streamlit process. |
| Page shows an error after clicking Start | A missing provider key or an invalid model configuration. The Launch tab states which key is needed. |
| Live run fails immediately | Check that the provider key is loaded and that the model is callable; being listed by a provider does not guarantee access. |
| Two dashboards on one port | Stop the extra Streamlit process; use one instance. |
| Live status disappears | The Streamlit process restarted. The run file remains in `runs/` and is listed after it completes. |
