"""
Add the project's src/ directory to sys.path so scripts can do
`import config`, `import evaluate`, etc.

Every script in scripts/ should `import _path  # noqa: F401` as its first
import.
"""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SRC  = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
