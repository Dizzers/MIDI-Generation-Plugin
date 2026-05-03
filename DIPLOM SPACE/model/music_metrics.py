"""Lightweight music-quality metrics computed from a list of token strings.
Used both during validation in train.py and offline in generate.py.

All metrics are scalar in [0..1].
"""
from __future__ import annotations


_MAJOR = (0, 2, 4, 5, 7, 9, 11)
_MINOR = (0, 2, 3, 5, 7, 8, 10)


def _best_scale_coverage(pitches: list[int]) -> float:
    if not pitches:
        return 0.0
    pcs = [p % 12 for p in pitches]
    best = 0.0
    for root in range(12):
        major_set = {(n + root) % 12 for n in _MAJOR}
        minor_set = {(n + root) % 12 for n in _MINOR}
        major_cov = sum(pc in major_set for pc in pcs) / len(pcs)
        minor_cov = sum(pc in minor_set for pc in pcs) / len(pcs)
        best = max(best, major_cov, minor_cov)
    return best


def sequence_metrics(tokens: list[str]) -> dict:
    """Return repeat_rate, unique_token_ratio, rhythm_diversity, scale_coverage."""
    if len(tokens) < 2:
        return {
            "repeat_rate": 0.0,
            "unique_token_ratio": 0.0,
            "rhythm_diversity": 0.0,
            "scale_coverage": 0.0,
        }

    repeat_rate = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1]) / (
        len(tokens) - 1
    )
    unique_token_ratio = len(set(tokens)) / len(tokens)

    pitches: list[int] = []
    time_steps: list[int] = []
    for tok in tokens:
        if tok.startswith("NOTE_ON_"):
            try:
                pitches.append(int(tok.rsplit("_", 1)[-1], 16))
            except ValueError:
                pass
        elif tok.startswith("TIME_SHIFT_"):
            try:
                time_steps.append(int(tok.rsplit("_", 1)[-1], 16))
            except ValueError:
                pass

    rhythm_diversity = len(set(time_steps)) / max(1, len(time_steps))
    scale_coverage = _best_scale_coverage(pitches)

    return {
        "repeat_rate": repeat_rate,
        "unique_token_ratio": unique_token_ratio,
        "rhythm_diversity": rhythm_diversity,
        "scale_coverage": scale_coverage,
    }


def aggregate_metrics(per_seq_metrics: list[dict]) -> dict:
    if not per_seq_metrics:
        return {
            "repeat_rate": 0.0,
            "unique_token_ratio": 0.0,
            "rhythm_diversity": 0.0,
            "scale_coverage": 0.0,
        }
    keys = ("repeat_rate", "unique_token_ratio", "rhythm_diversity", "scale_coverage")
    out: dict[str, float] = {}
    for key in keys:
        out[key] = sum(m[key] for m in per_seq_metrics) / len(per_seq_metrics)
    return out
