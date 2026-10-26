# Form FF No. 180: Progress Review Report 2
**Vishwakarma Institute of Technology, Pune**
**Department of Computer Engineering | Academic Year 2026?27 | Semester VII**

- **Project Title**: NEMO Evaluator Use Case: Benchmarking and Comparing the Performance of Various Models
- **Industry Partner**: Tata Consultancy Services (TCS)
- **Group No.**: 16
- **Review Schedule**: End of Semester (Review 2 - Final Project Review)
- **Students**:
  - Ashish Kumar (Roll: 110, G.R.No: 12311937)
  - Aryaa Sharma (Roll: 149, G.R.No: 12311983)
- **Internal Guide**: Prof. (Dr.) Yogesh Sharma

---

## 1. Final Deliverables Accomplished
1. **Full-Stack Benchmarking System Implemented**:
   - Complete Python package with CLI tool `nel` and Streamlit web application.
   - Support for 6 flagship models: Llama-3.1-70B, Llama-3.1-8B, Qwen-2.5-72B, Mixtral-8x7B, DeepSeek-R1-Distill-70B, and GPT-4o-Mini.
2. **Experimental Evaluation Results**:
   - **Top Overall Accuracy**: Qwen-2.5-72B-Instruct (86.7%) and DeepSeek-R1-Distill-70B (81.7%).
   - **Math & Reasoning Leader**: DeepSeek-R1 achieved 100% on GSM8K and highest accuracy on GPQA (80.0%) via deep Chain-of-Thought reasoning.
   - **Code Generation Leader**: Qwen-2.5-72B and Llama-3.1-70B achieved 100% pass@1 on evaluated HumanEval problems.
   - **Efficiency Leader**: Llama-3.1-8B achieved lowest mean inference latency (0.28s) and lowest token cost ($0.0019/run).
3. **Statistical Significance & Flip Analysis (`nel compare`)**:
   - Paired McNemar tests demonstrated statistically significant accuracy gains for Llama-3.1-70B over Llama-3.1-8B (`p = 0.00052`, Chi2 = 12.04).
   - Flip Inspector revealed 21 items where 70B corrected 8B errors, primarily in multi-step math and coding.
4. **Sustainability & Resource Optimization (SDG 9 & SDG 12)**:
   - Evaluated models on Pareto accuracy-vs-latency frontier.
   - Created turnkey Google Colab notebook (`nemo_evaluator_colab.ipynb`) allowing zero-cost cloud GPU execution, eliminating the need for expensive dedicated on-premise infrastructure.
