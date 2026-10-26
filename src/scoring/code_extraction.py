import re
from typing import List, Optional

_FENCE = re.compile(r"```[ \t]*(?:[A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
_OPEN_FENCE_TO_END = re.compile(r"```[ \t]*(?:[A-Za-z0-9_+-]*)[ \t]*\r?\n(.*)\Z", re.DOTALL)
_CODE_START = re.compile(r"^(def |async def |class |import |from |@)")


def _prompt_imports(prompt: str) -> List[str]:
    return [ln.rstrip() for ln in prompt.splitlines() if re.match(r"^(import|from)\s", ln)]


def extract_python_code(response: str, prompt: str = "") -> Optional[str]:
    """Turn a model reply to a HumanEval prompt into runnable Python, or None if it contains no code.

    Handles the usual reply shapes: prose around a fenced block, several fenced blocks (the last one
    that defines a function wins), an unfenced function after a line of prose, and a bare indented
    function body (which is attached to the original prompt). Imports from the prompt are always
    available, because replies often omit them.
    """
    if not response or not response.strip():
        return None

    blocks = [m.group(1) for m in _FENCE.finditer(response)]
    if not blocks:
        trailing = _OPEN_FENCE_TO_END.search(response)
        if trailing:
            blocks = [trailing.group(1)]

    code: Optional[str] = None
    if blocks:
        with_def = [b for b in blocks if re.search(r"^\s*(async\s+)?def\s", b, re.MULTILINE)]
        code = (with_def or blocks)[-1]
    else:
        lines = response.splitlines()
        start = next((i for i, ln in enumerate(lines) if _CODE_START.match(ln)), None)
        if start is not None:
            code = "\n".join(lines[start:])
        else:
            first = next((ln for ln in lines if ln.strip()), "")
            if first[:1] in (" ", "\t"):
                code = response  # a bare, indented function body
    if code is None or not code.strip():
        return None

    code = code.rstrip() + "\n"
    if not re.search(r"^\s*(async\s+)?def\s", code, re.MULTILINE) and prompt:
        code = prompt.rstrip("\n") + "\n" + code  # body only: complete the original stub

    present = {ln.strip() for ln in code.splitlines()}
    missing = [imp for imp in _prompt_imports(prompt) if imp.strip() not in present]
    if missing:
        code = "\n".join(missing) + "\n" + code
    return code
