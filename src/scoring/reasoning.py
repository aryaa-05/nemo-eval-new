import re
from typing import Tuple

_TAGS = "think|thinking|reasoning"
_BLOCK = re.compile(rf"<({_TAGS})>.*?</\1>", re.IGNORECASE | re.DOTALL)
_OPEN = re.compile(rf"<({_TAGS})>", re.IGNORECASE)
_CLOSE = re.compile(rf"</({_TAGS})>", re.IGNORECASE)

OK = "ok"
UNCLOSED = "unclosed_reasoning"


def strip_reasoning(text: str) -> Tuple[str, str]:
    """Remove a reasoning model's thinking text so only the final answer is scored.

    Returns (answer_text, status):
      - complete <think>...</think> blocks are removed
      - a closing tag with no opening tag (template pre-filled the opener) keeps only the text after the last closing tag
      - an opening tag that is never closed means the model never finished thinking: there is no answer,
        so ("", "unclosed_reasoning") is returned
    """
    if not text:
        return "", OK
    out = text
    previous = None
    while previous != out:
        previous = out
        out = _BLOCK.sub("", out)
    if _OPEN.search(out):
        return "", UNCLOSED
    closers = list(_CLOSE.finditer(out))
    if closers:
        out = out[closers[-1].end():]
    # Drop leading blank lines and trailing whitespace, but keep the first line's indentation (code bodies need it).
    return re.sub(r"\A(?:[ \t]*\r?\n)+", "", out).rstrip(), OK
