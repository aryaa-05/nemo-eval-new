# Form FF No. 180: Progress Review Report 2

**Vishwakarma Institute of Technology, Pune**
**Department of Computer Engineering | Academic Year 2026-27 | Semester VII**

- **Project Title**: NeMo Evaluator: Benchmarking and Comparing the Performance of Various Models
- **Industry Partner**: Tata Consultancy Services (TCS)
- **Group No.**: 16
- **Review Schedule**: End of Semester (Review 2 - Final Project Review)
- **Students**:
  - Ashish Kumar (Roll: 110, G.R.No: 12311937)
  - Aryaa Sharma (Roll: 149, G.R.No: 12311983)
- **Internal Guide**: Prof. (Dr.) Yogesh Sharma

---

## 1. Final deliverables accomplished

1. **Full benchmarking system**
   - Python package with `nel` CLI and Streamlit dashboard.
   - Benchmarks for MMLU-Pro, GSM8K, MATH-500, GPQA, HumanEval, TruthfulQA, BYOB, KYC India, RBI regulations and image-processing theory.
   - Three 100-row sourced datasets with `source`, `source_excerpt`, balanced answer positions and validator checks.

2. **Model gateway and run artefacts**
   - `mock`, NVIDIA NIM, Groq, OpenRouter, OpenAI, Ollama and `auto` route support.
   - LIVE/SIMULATED run mode, served-model and substitute-route recording.
   - Retries, throttling, early stop for dead models, incremental checkpoints, scorer version stamping and offline re-scoring.

3. **Scoring and comparison**
   - Wilson 95% confidence intervals, paired McNemar test, item-level flips, reliability flags and PASS/FAIL gate.
   - Markdown/HTML reports with mode, provider, job name, served model, substitute warnings, reliability and scorer version.

4. **Verification completed**
   - 396 tests pass with `pytest` and `unittest`.
   - 13 zero-cost NIM/Groq models completed one-item LIVE `mmlu_pro` probes.
   - LIVE free-tier pair jobs exercised the pipeline end to end; scores belong only to the exact served models.

## 2. Results that must not be claimed

- No result exists for any of the six required pair models.
- All six OpenRouter routes completed one-item LIVE probes on 20 November 2026, but full benchmark runs and comparisons are pending; probe scores are not results.
- SIMULATED runs only test software and say nothing about model capability.
- LIVE free-tier scores are results for those free-tier models only.
- SUBSTITUTE routes measure served models, not the exact requested variants.

## 3. Remaining work and blockers

- `OPENROUTER_API_KEY` is present; one-item route probes are complete and full required-pair runs are pending.
- The Qwen3.5 size and the vision-model part of the request are still unspecified.
- Most answer keys in the three new datasets were written from source text but were not independently human-checked.
- The older 15-row datasets have wide confidence intervals and unverified keys.
- Full runs and comparison reports for the required pairs are pending.
