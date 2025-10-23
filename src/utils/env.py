import os

from typing import Optional

def parse_timeout_env(var: str) -> Optional[float]:
    """Parse float timeout from env var; return None if invalid or missing."""
    val = os.environ.get(var)
    if not val:
        return None
    try:
        return float(val)
    except ValueError:
        print(f"invalid {var}='{val}', fallback to default.")
        return None
