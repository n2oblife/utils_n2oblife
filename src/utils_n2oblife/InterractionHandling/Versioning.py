import sys
from typing import Optional


def in_venv() -> bool:
    """Check if running in a virtual environment."""
    base_prefix = (
        getattr(sys, "base_prefix", None) or 
        getattr(sys, "real_prefix", None) or 
        sys.prefix
    )
    return sys.prefix != base_prefix


def get_python_version() -> Optional[str]:
    """Get the Python version string."""
    try:
        return sys.version.split(' ')[0]
    except (AttributeError, IndexError):
        return None