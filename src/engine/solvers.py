from typing import Dict, Any

CODE_INSTRUCTION = (
    "Complete the following Python function. Reply with the complete function, including its signature "
    "and docstring, in a single ```python code block and no other text."
)

def _code_prompt(item: Dict[str, Any]) -> str:
    return f"{CODE_INSTRUCTION}\n\n{item.get('prompt') or item.get('question', '')}"

class BaseSolver:
    def format_prompt(self, item: Dict[str, Any], benchmark_id: str) -> str:
        raise NotImplementedError

class SimpleChatSolver(BaseSolver):
    def format_prompt(self, item: Dict[str, Any], benchmark_id: str) -> str:
        if benchmark_id == "humaneval":
            return _code_prompt(item)
        q = item.get("question") or item.get("prompt", "")
        options = item.get("options")
        if options:
            opt_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            return f"{q}\n\nOptions:\n{opt_str}\n\nPlease choose the correct option (e.g. A, B, C...) directly."
        return q

class ChainOfThoughtSolver(BaseSolver):
    def format_prompt(self, item: Dict[str, Any], benchmark_id: str) -> str:
        if benchmark_id == "humaneval":
            return _code_prompt(item)
        q = item.get("question") or item.get("prompt", "")
        options = item.get("options")
        if options:
            opt_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            return f"{q}\n\nOptions:\n{opt_str}\n\nThink through the problem step-by-step. In your final line, clearly state the correct option letter like: Answer: [OPTION]."
        return f"{q}\n\nSolve this step-by-step. Provide the final concise answer clearly on the last line."

class HarborAgenticSolver(BaseSolver):
    def format_prompt(self, item: Dict[str, Any], benchmark_id: str) -> str:
        if benchmark_id == "humaneval":
            return _code_prompt(item)
        q = item.get("question") or item.get("prompt", "")
        return f"[Harbor Agentic Workflow]\nTask: {q}\nExplore hypotheses, verify intermediate bounds, and provide verified final output."

def get_solver(solver_type: str = "simple") -> BaseSolver:
    if solver_type == "cot":
        return ChainOfThoughtSolver()
    elif solver_type == "harbor":
        return HarborAgenticSolver()
    return SimpleChatSolver()
