"""
pytest fixtures and path setup. Adds the project's src/ directory to
sys.path so test files can import config, evaluate, features, etc.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC  = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
else:
    # Fallback for the flat layout where modules live in the repo root.
    sys.path.insert(0, str(ROOT))
