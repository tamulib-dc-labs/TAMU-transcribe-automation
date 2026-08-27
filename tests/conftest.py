"""Put the repo root on sys.path so `src.asr` imports work from anywhere."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
