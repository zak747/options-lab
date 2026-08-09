"""Make `src/` importable without an editable install.

pytest imports the root conftest.py before collecting anything, so this
puts optionslab on sys.path for the whole test session.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
