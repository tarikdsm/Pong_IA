from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
ENGINE_PATH = REPO_ROOT / "engine"
TRAINING_SRC_PATH = REPO_ROOT / "training" / "src"

if str(ENGINE_PATH) not in sys.path:
    sys.path.insert(0, str(ENGINE_PATH))

if str(TRAINING_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(TRAINING_SRC_PATH))
