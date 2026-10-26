import os
from typing import List, Optional

DEFAULT_ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")


def _parse_line(line: str) -> Optional[tuple]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].lstrip()
    if "=" not in line:
        return None
    name, value = line.split("=", 1)
    name = name.strip()
    if not name or not (name[0].isalpha() or name[0] == "_") or not all(c.isalnum() or c == "_" for c in name):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    else:
        # unquoted value: an inline " #" starts a comment
        hash_at = value.find(" #")
        if hash_at != -1:
            value = value[:hash_at].rstrip()
    return name, value


def load_env_file(path: Optional[str] = None) -> List[str]:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Variables that are already set in the environment are never overridden, and values are never
    printed or logged. Returns the NAMES of the variables that were newly set. A missing file is not an error.
    """
    env_path = os.path.abspath(path or DEFAULT_ENV_PATH)
    if not os.path.isfile(env_path):
        return []
    newly_set: List[str] = []
    with open(env_path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            parsed = _parse_line(raw)
            if parsed is None:
                continue
            name, value = parsed
            if name in os.environ:
                continue
            os.environ[name] = value
            newly_set.append(name)
    return newly_set
