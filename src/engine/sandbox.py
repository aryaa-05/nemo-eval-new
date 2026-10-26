import multiprocessing
import sys
from typing import Tuple, Optional

def _unsafe_execute(program: str, result_queue):
    try:
        # Create dedicated execution namespace
        exec_globals = {}
        exec(program, exec_globals)
        # Call the check function if defined
        if "check" in exec_globals and "candidate" in exec_globals:
            exec_globals["check"](exec_globals["candidate"])
        result_queue.put((True, None))
    except AssertionError as e:
        result_queue.put((False, f"AssertionError: {e}"))
    except Exception as e:
        result_queue.put((False, f"{type(e).__name__}: {e}"))

class CodeSandbox:
    @staticmethod
    def execute_code(code: str, test_code: str, entry_point: str, timeout_seconds: float = 3.0) -> Tuple[bool, Optional[str]]:
        full_program = f"""{code}

candidate = {entry_point}
{test_code}
check(candidate)
"""
        # Run execution in a separate process with timeout
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_unsafe_execute, args=(full_program, queue))
        proc.start()
        proc.join(timeout=timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join(1.0)
            return False, f"TimeoutExpired: Execution exceeded {timeout_seconds}s limit"

        if not queue.empty():
            passed, err = queue.get()
            return passed, err
        return False, "Unknown execution error (Process died unexpectedly)"
