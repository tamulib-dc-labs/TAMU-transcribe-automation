"""Reference-based word accuracy (WER / CER).

"Word-level accuracy" means two different things depending on what you have:

* **No reference transcript** - the per-word ``confidence`` produced by the
  aligner is the only signal available. It is an acoustic posterior, not a
  measure of correctness.
* **With a reference transcript** - actual correctness can be measured. This
  module Levenshtein-aligns the hypothesis against the reference and tags every
  hypothesised word ``correct`` / ``substitution`` / ``insertion``, alongside
  corpus-level WER, CER and word accuracy.

Both are surfaced: ``confidence`` on every word always, ``status`` on every word
when a reference was supplied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .textutil import normalize_for_scoring, split_words
from .types import (
    STATUS_CORRECT,
    STATUS_INSERTION,
    STATUS_SUBSTITUTION,
    Transcript,
    Word,
)

#: Refuse to build an edit matrix bigger than this (rows x columns).
MAX_MATRIX_CELLS = 25_000_000

MATCH, SUB, DELETE, INSERT = "match", "substitution", "deletion", "insertion"


@dataclass(frozen=True)
class EditOp:
    kind: str  # match | substitution | deletion | insertion
    ref_index: Optional[int]
    hyp_index: Optional[int]


def levenshtein_ops(reference: list[str], hypothesis: list[str]) -> list[EditOp]:
    """Minimum-edit alignment of ``hypothesis`` onto ``reference``."""
    n, m = len(reference), len(hypothesis)
    if (n + 1) * (m + 1) > MAX_MATRIX_CELLS:
        raise ValueError(
            f"reference ({n} words) x hypothesis ({m} words) exceeds the "
            f"{MAX_MATRIX_CELLS:,}-cell scoring limit; score shorter excerpts"
        )

    # dist[i][j] = edit distance between reference[:i] and hypothesis[:j]
    dist = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = i
    for j in range(1, m + 1):
        dist[0][j] = j
    for i in range(1, n + 1):
        row, prev = dist[i], dist[i - 1]
        ref_word = reference[i - 1]
        for j in range(1, m + 1):
            substitute = prev[j - 1] + (ref_word != hypothesis[j - 1])
            row[j] = min(substitute, prev[j] + 1, row[j - 1] + 1)

    ops: list[EditOp] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = dist[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1])
            if dist[i][j] == cost:
                kind = MATCH if reference[i - 1] == hypothesis[j - 1] else SUB
                ops.append(EditOp(kind, i - 1, j - 1))
                i, j = i - 1, j - 1
                continue
        if i > 0 and dist[i][j] == dist[i - 1][j] + 1:
            ops.append(EditOp(DELETE, i - 1, None))
            i -= 1
            continue
        ops.append(EditOp(INSERT, None, j - 1))
        j -= 1
    ops.reverse()
    return ops


def _edit_distance(a: Iterable[str], b: Iterable[str]) -> int:
    """Plain distance (no backtrace), used for CER."""
    a, b = list(a), list(b)
    if not a:
        return len(b)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(previous[j - 1] + (ca != cb), previous[j] + 1, current[j - 1] + 1)
            )
        previous = current
    return previous[-1]


def score_transcript(
    transcript: Transcript,
    reference_text: str,
    tag_words: bool = True,
) -> dict[str, Any]:
    """Score ``transcript`` against ``reference_text`` and tag each word.

    Returns the metrics dict (also stored on ``transcript.accuracy``).
    """
    hyp_words = transcript.words
    ref_tokens = [
        norm
        for norm in (normalize_for_scoring(w) for w in split_words(reference_text))
        if norm
    ]
    hyp_pairs = [(w, normalize_for_scoring(w.text)) for w in hyp_words]
    hyp_tokens = [norm for _, norm in hyp_pairs if norm]
    hyp_objects = [word for word, norm in hyp_pairs if norm]

    if not ref_tokens:
        raise ValueError("reference transcript contains no scorable words")

    ops = levenshtein_ops(ref_tokens, hyp_tokens)
    hits = sum(1 for op in ops if op.kind == MATCH)
    substitutions = sum(1 for op in ops if op.kind == SUB)
    deletions = sum(1 for op in ops if op.kind == DELETE)
    insertions = sum(1 for op in ops if op.kind == INSERT)

    if tag_words:
        _tag(hyp_objects, ref_tokens, ops)

    n_ref = len(ref_tokens)
    ref_chars = list("".join(ref_tokens))
    hyp_chars = list("".join(hyp_tokens))
    metrics = {
        "reference_words": n_ref,
        "hypothesis_words": len(hyp_tokens),
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "wer": round((substitutions + deletions + insertions) / n_ref, 4),
        "word_accuracy": round(hits / n_ref, 4),
        "cer": round(_edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars)), 4),
        "mean_confidence": _mean([w.confidence for w in hyp_words]),
        "mean_confidence_correct": _mean(
            [w.confidence for w in hyp_objects if w.status == STATUS_CORRECT]
        ),
        "mean_confidence_incorrect": _mean(
            [w.confidence for w in hyp_objects if w.status == STATUS_SUBSTITUTION]
        ),
    }
    transcript.accuracy = metrics
    return metrics


def _tag(hyp_objects: list[Word], ref_tokens: list[str], ops: list[EditOp]) -> None:
    status = {
        MATCH: STATUS_CORRECT,
        SUB: STATUS_SUBSTITUTION,
        INSERT: STATUS_INSERTION,
    }
    for op in ops:
        if op.hyp_index is None:
            continue  # deletion: nothing in the hypothesis to tag
        word = hyp_objects[op.hyp_index]
        word.status = status[op.kind]
        word.reference = ref_tokens[op.ref_index] if op.ref_index is not None else None


def _mean(values: list[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return round(sum(present) / len(present), 4) if present else None


__all__ = ["EditOp", "levenshtein_ops", "score_transcript", "MAX_MATRIX_CELLS"]
