# Form FF No. 180: Progress Review Report 1

**Vishwakarma Institute of Technology, Pune**
**Department of Computer Engineering | Academic Year 2026-27 | Semester VII**

- **Project Title**: NeMo Evaluator: Benchmarking and Comparing the Performance of Various Models
- **Industry Partner**: Tata Consultancy Services (TCS)
- **Group No.**: 16
- **Review Schedule**: Mid Semester (Review 1)
- **Students**:
  - Ashish Kumar (Roll: 110, G.R.No: 12311937)
  - Aryaa Sharma (Roll: 149, G.R.No: 12311983)
- **Internal Guide**: Prof. (Dr.) Yogesh Sharma

---

## 1. Work Completed in Review 1

1. **Literature survey and problem formulation**
   - Surveyed HELM, MMLU-Pro, MT-Bench/LLM-as-a-Judge, lm-evaluation-harness, and NVIDIA NeMo Evaluator.
   - Formulated the problem as reproducible, multi-dimensional LLM comparison for enterprise model selection.

2. **Platform architecture**
   - Designed layers for presentation, orchestration, evaluation, model gateway, scoring, storage, comparison, and reporting.

3. **Core scaffolding and model gateway**
   - Implemented `UnifiedModelClient` with `mock`, NVIDIA NIM, Groq, OpenRouter, OpenAI, Ollama and `auto` routes.
   - Added provider host/key pinning, retries, throttling, and response caching.

4. **Benchmark ingestion and scoring**
   - Added MMLU-Pro, GSM8K, MATH-500, GPQA, HumanEval, TruthfulQA and custom BYOB datasets.
   - Added multiple-choice, math and code scorers with reasoning-text stripping and failure flags.

## 2. Milestones for Review 2

- Add the three sourced 100-row India-focused datasets.
- Add statistical comparison with Wilson intervals, McNemar flips and quality gate.
- Deploy the Streamlit dashboard with a background evaluation thread.
- Validate the live pipeline on callable free-tier models and document limitations honestly.
