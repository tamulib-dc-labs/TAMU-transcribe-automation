"""ASR implementation: Parakeet-TDT for words, Sortformer for speaker turns.

Replaces the WhisperX + wav2vec2 stack. Both models are NeMo, share one
environment, and derive time from the same FastConformer frontend, so word
timestamps and speaker turns land on a common measured time base.

No module here imports torch or NeMo at module level - every model import lives
inside the ``load()`` that needs it, so the orchestrator can import this package
without a GPU stack installed. Keep it that way: put model imports inside the
function that needs them, never at the top of a module.
"""

from .fusion import SpeakerTurn, fuse
from .lines import LineConfig, build_lines
from .output import write_outputs
from .parakeet import DEFAULT_PARAKEET_MODEL, ParakeetBackend, ParakeetConfig
from .preprocess import find_audio_files, prepare_audio
from .scoring import score_transcript
from .sortformer import DEFAULT_SORTFORMER_MODEL, SortformerConfig, SortformerDiarizer
from .transcriber import Transcriber, TranscriberConfig, transcribe
from .types import Line, Segment, Transcript, Word

__version__ = "3.0.0"

__all__ = [
    "DEFAULT_PARAKEET_MODEL",
    "DEFAULT_SORTFORMER_MODEL",
    "Line",
    "LineConfig",
    "ParakeetBackend",
    "ParakeetConfig",
    "Segment",
    "SortformerConfig",
    "SortformerDiarizer",
    "SpeakerTurn",
    "Transcriber",
    "TranscriberConfig",
    "Transcript",
    "Word",
    "build_lines",
    "find_audio_files",
    "fuse",
    "prepare_audio",
    "score_transcript",
    "transcribe",
    "write_outputs",
]
