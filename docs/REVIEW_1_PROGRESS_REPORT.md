# Form FF No. 180: Progress Review Report 1
**Vishwakarma Institute of Technology, Pune**
**Department of Computer Engineering | Academic Year 2026?27 | Semester VII**

- **Project Title**: NEMO Evaluator Use Case: Benchmarking and Comparing the Performance of Various Models
- **Industry Partner**: Tata Consultancy Services (TCS)
- **Group No.**: 16
- **Review Schedule**: Mid Sem. Semester (Review 1)
- **Students**:
  - Ashish Kumar (Roll: 110, G.R.No: 12311937)
  - Aryaa Sharma (Roll: 149, G.R.No: 12311983)
- **Internal Guide**: Prof. (Dr.) Yogesh Sharma

---

## 1. Work Completed in Review 1
1. **Literature Survey & Problem Formulation**:
   - Surveyed fundamental literature: HELM (Liang et al., 2022), MMLU-Pro (Wang et al., NeurIPS 2024), MT-Bench & LLM-as-a-Judge (Zheng et al., NeurIPS 2023), lm-evaluation-harness (Gao et al.), and NVIDIA NeMo Evaluator (2026).
   - Formulated the problem statement addressing lack of reproducible multi-dimensional LLM benchmarking.
2. **Platform Architecture Design**:
   - Finalized 8-layer architecture spanning Presentation Layer, Orchestration & Control Plane, Core Evaluation Engine, Model Gateway, Scoring & Judgement, Results Store, Comparison Engine, and Infrastructure.
3. **Core Scaffolding & Model Gateway**:
   - Implemented `UnifiedModelClient` supporting NVIDIA NIM, OpenAI-compatible APIs, Ollama, and built-in Mock Simulator.
   - Built `InterceptorProxy` supporting response caching, rate limiting, and millisecond-level latency tracking.
4. **Standardized Benchmarks Ingestion**:
   - Curated test suites for MMLU-Pro, GSM8K, MATH-500, GPQA, HumanEval, TruthfulQA, and BYOB custom sets.
   - Built safe multiprocessing code execution sandbox (`CodeSandbox`) for HumanEval.

## 2. Milestones for Review 2
- Execute comprehensive cross-model benchmark runs across 6 LLMs.
- Implement statistical significance testing (McNemar paired test, Wilson confidence intervals).
- Deploy interactive Streamlit Dashboard with item-level flip inspector.
- Prepare Google Colab notebook for GPU cloud execution.
