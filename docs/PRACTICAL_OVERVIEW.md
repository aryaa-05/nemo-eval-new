# Practical Overview — What We Are Doing and How It Works

**Date:** 18 October 2026
**Language level:** Simple — no heavy jargon. Read this first, then `PROJECT_DOCUMENTATION.md` for deep technical detail.

---

## 1. What are we doing, practically?

We are building a system that **tests AI models like a student giving an exam**, then **compares their answer sheets statistically** to decide which model is actually better.

Real-world problem: a company wants to deploy an LLM (for chatbots, document processing, etc.). Which model should they pick? Vendor claims ("our model is 10% better!") are often noise or marketing. Our platform replaces guesswork with a **fair, repeatable exam + statistical proof**.

Concretely, the system:
1. **Asks a model questions** from standard exam papers (called "benchmarks") — maths, science, code, compliance, etc.
2. **Checks answers automatically** against the correct answer key (regex matching, code execution in a sandbox, etc.).
3. **Scores the model** — accuracy percentage with a 95% confidence interval (so we know how reliable the score is).
4. **Compares two models** on the exact same questions — question-by-question — and runs a **statistical significance test (McNemar's)** to answer: *"Is Model B really better, or did it just get lucky?"*
5. **Gives a PASS / FAIL verdict** for deployment ("Quality Gate").

---

## 2. How does it work? (Step by step, in plain words)

Imagine two students writing the same exam:

```
Question paper (benchmark dataset, e.g. KYC India questions)
        │
        ▼
   ┌─────────┐    prompt (question + options)    ┌──────────────┐
   │  Engine  │ ───────────────────────────────► │  LLM Model   │
   └─────────┘                                   │ (Ministral,  │
        │                                        │  Llama, Qwen)│
        │   raw text answer ("Answer: B")        └──────────────┘
        ▼
   Answer extractor (regex finds "B" from the response)
        ▼
   Compare with answer key → ✅ correct or ❌ wrong
        ▼
   Store every answer in SQLite (runs/ folder)
        ▼
   Compare Model A's sheet vs Model B's sheet:
     - both correct / both wrong / A-only / B-only ("flips")
     - McNemar test → is the difference real (p < 0.05)?
        ▼
   Report (markdown/HTML) + PASS/FAIL gate decision
```

**Key ideas in simple terms:**
- **Benchmark** = a fixed question paper (JSONL file with questions + answer key). We can't change questions between models — same paper, fair test.
- **Solver** = how we phrase the question. "Just answer" vs "think step-by-step" (Chain of Thought). Different phrasing can change scores.
- **Rule scorer** = an automatic answer-sheet checker (regex finds the final answer letter/number; no human needed).
- **Confidence interval (95% CI)** = "this model's true score is probably between X% and Y%". If two models' intervals overlap heavily, the difference may be fake.
- **McNemar's test** = a math test on the *disagreements* between two models. If Model B flipped 10 wrong→right answers and Model A only flipped 2, McNemar tells us if that's a real pattern or just luck.
- **Quality Gate** = the final verdict: candidate model must beat a minimum accuracy AND not regress vs the baseline → **PASS**. Otherwise **FAIL** or **INCONCLUSIVE**.

---

## 3. What was already done (built earlier)?

| Layer | What exists | Where |
|---|---|---|
| **6 standard benchmarks** | MMLU-Pro, GSM8K, MATH-500, GPQA, HumanEval, TruthfulQA — question datasets + auto-scorers | `datasets/`, `src/benchmarks/base.py` |
| **Model gateway** | One client to talk to NVIDIA NIM / OpenAI / HuggingFace / Ollama — plus a free **mock simulator** (no API cost) | `src/gateway/` |
| **Evaluation engine** | Runs every question through model → extracts answer → scores → stores in SQLite | `src/engine/runner.py` |
| **Statistics** | Wilson confidence intervals, McNemar significance test, PASS/FAIL quality gate | `src/scoring/statistics.py` |
| **Comparison** | Question-by-question diff between two runs, "flip" detection (which questions changed from right to wrong or vice versa) | `src/comparison/diff_engine.py` |
| **CLI (`nel`)** | `nel list`, `nel eval run`, `nel compare`, `nel gate`, `nel report` commands | `src/cli/nel.py` |
| **Web dashboard** | Streamlit UI: leaderboards, radar charts, flip inspector | `src/dashboard/app.py` |
| **Reports** | Auto-generated Markdown/HTML comparison reports | `src/reporting/` |
| **6 original models** | Llama-3.1 (8B/70B), Qwen-2.5-72B, Mixtral-8x7B, DeepSeek-R1-Distill, GPT-4o-mini (all mock by default) | `configs/models_config.yaml` |
| **Past experiment** | Llama-3.1-8B vs 70B comparison with McNemar results | `runs/llama_8b_vs_70b_comparison.md` |

---

## 4. What was added *just now* (18 October 2026)?

### 4.1 Six new models (3 comparison pairs)

| Pair | Model A (candidate) | Model B (baseline) | Purpose |
|---|---|---|---|
| **Pair 1** | `mistralai/ministral-3-8b-instruct-2512` | `meta/llama-3.1-8b-instruct` | Same-size instruct models (8B vs 8B) |
| **Pair 2** | `mistralai/ministral-3-14b-reasoning-2512` | `meta/llama-4-scout-17b-16e-instruct` | Reasoning model vs MoE frontier-small model |
| **Pair 3** | `mistralai/ministral-3-8b-reasoning-2512` | `qwen/qwen3.5-8b` | Compact reasoning models head-to-head |

> All six added to `configs/models_config.yaml`. Default provider is `mock` (free simulator) — switch to `nvidia_nim` for live testing. *Verify exact model IDs on integrate.api.nvidia.com before going live.*

### 4.2 Three new domain benchmarks (text-MCQ, 15 questions each — expandable to 100)

| Benchmark ID | Domain | Questions cover |
|---|---|---|
| **`kyc_india`** | Compliance & Regulation | Indian KYC norms — PAN, Aadhaar eKYC, V-CIP, PMLA, CKYC, beneficial ownership |
| **`rbi_regulations`** | Banking & Regulation | RBI rules — BASEL-III capital ratios, CRR/SLR, repo rate, priority sector lending, digital lending, NPAs |
| **`image_processing`** | Computer Vision | Image processing theory — Nyquist theorem, filters, DCT/JPEG, morphology, Otsu thresholding, color spaces |

- Dataset format: `datasets/<benchmark>/sample.jsonl` — same MCQ schema as existing benchmarks (`question_id`, `question`, `options[4]`, `answer` letter).
- Registered in `src/benchmarks/base.py` and `configs/benchmarks_config.yaml`.
- Currently **text-only** (as agreed) — image-input (vision) support is a planned next step; the benchmark IDs are already vision-ready so real images can be swapped in later without code changes.

### 4.3 Three ready-to-run comparison configs (15 items/benchmark)

| Config file | Runs this pair |
|---|---|
| `configs/eval_pair1_ministral8b_vs_llama31.yaml` | Ministral-3-8B-Instruct vs Llama-3.1-8B |
| `configs/eval_pair2_ministral14b_reasoning_vs_llama4scout.yaml` | Ministral-3-14B-Reasoning vs Llama-4-Scout |
| `configs/eval_pair3_ministral8b_reasoning_vs_qwen35.yaml` | Ministral-3-8B-Reasoning vs Qwen3.5-8B |

Each config: `limit_per_benchmark: 15` (fast/cheap runs; raise to 100 when full datasets are ready), all 9 benchmarks enabled, and a `quality_gate` section defining baseline vs candidate.

### 4.4 Verified end-to-end (smoke test, 18 Oct 2026)

Ran Pair 1 on the three new benchmarks using the mock simulator:

```
Ministral-3-8B-Instruct : KYC 14/15 (93.3%) | RBI 13/15 (86.7%) | Image-Proc 14/15 (93.3%)
Llama-3.1-8B-Instruct   : KYC 11/15 (73.3%) | RBI 10/15 (66.7%) | Image-Proc 12/15 (80.0%)
McNemar p-value: 0.043 → difference declared SIGNIFICANT
Report exported: runs/smoke_pair1_comparison.md
```

(Mock scores are simulated, not real model capabilities — they only prove the pipeline works.)

---

## 5. How to run everything (copy-paste commands)

```powershell
# See all models + benchmarks
python src/cli/nel.py list

# Run each pair (mock provider = free; switch to nvidia_nim for live)
python src/cli/nel.py eval run --config configs/eval_pair1_ministral8b_vs_llama31.yaml
python src/cli/nel.py eval run --config configs/eval_pair2_ministral14b_reasoning_vs_llama4scout.yaml
python src/cli/nel.py eval run --config configs/eval_pair3_ministral8b_reasoning_vs_qwen35.yaml

# Compare the two run IDs printed after each eval
python src/cli/nel.py compare --run-a <RUN_A_ID> --run-b <RUN_B_ID> --output-md comparison_pair1.md

# Deployment verdict
python src/cli/nel.py gate --baseline <BASELINE_RUN_ID> --candidate <CANDIDATE_RUN_ID>
```

---

## 6. What's next?

1. **Real 100-question datasets** — replace the 15-item placeholders in `datasets/kyc_india/`, `datasets/rbi_regulations/`, `datasets/image_processing/` with full JSONL files.
2. **Live model calls** — flip `provider` to `nvidia_nim` in `models_config.yaml` and set `$env:NVIDIA_API_KEY`; verify the exact Ministral-3 / Qwen3.5 model IDs on the NIM catalogue.
3. **Vision support** — add image input (base64) to the gateway client and JSONL schema for KYC document images / image-processing tasks; pick a vision model (e.g. Qwen2.5-VL or Llama-4 multimodal).
4. **Run all 3 pairs on live models** → export comparison reports for the Review submission.
