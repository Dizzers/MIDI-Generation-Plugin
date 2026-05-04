"""Music-quality metrics computed from a list of token strings + classification
metrics computed from logits/targets tensors.

Used both during validation in train.py and offline in generate.py.

The first half (`sequence_metrics`, `aggregate_metrics`) is pure Python and
operates on token-string lists. The second half (`TokenClassificationStats`)
is a torch-based accumulator that gives objective next-token classification
metrics (precision / recall / F1 macro-micro-weighted, top-1 / top-5 accuracy
and pitch-class IoU) over a whole eval pass.
"""
from __future__ import annotations

import torch


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


TOKEN_CATEGORIES = ("note_on", "note_off", "time_shift", "velocity", "special")


def _categorize_token(token: str) -> str:
    if token.startswith("NOTE_ON_"):
        return "note_on"
    if token.startswith("NOTE_OFF_"):
        return "note_off"
    if token.startswith("TIME_SHIFT_"):
        return "time_shift"
    if token.startswith("VELOCITY_"):
        return "velocity"
    return "special"


class TokenClassificationStats:
    """Accumulates objective classification metrics across an eval pass.

    Reports:
        top1_accuracy, top5_accuracy
        per-category precision / recall / F1 over 5 token categories
            (note_on, note_off, time_shift, velocity, special)
        category_macro_f1, category_micro_f1, category_weighted_f1
        pitch_class_iou: average per-sample IoU between the set of pitch
            classes present in predicted NOTE_ON tokens vs the set in target
            NOTE_ON tokens (a music-aware Jaccard score).
    """

    CATEGORIES = TOKEN_CATEGORIES

    def __init__(self, id2token) -> None:
        items: list[tuple[int, str]] = []
        for k, tok in id2token.items():
            items.append((int(k), tok))
        items.sort()
        vocab_size = items[-1][0] + 1 if items else 0
        cat_idx = torch.zeros(vocab_size, dtype=torch.long)
        pitch_class = torch.full((vocab_size,), -1, dtype=torch.long)
        for tid, tok in items:
            cat_idx[tid] = self.CATEGORIES.index(_categorize_token(tok))
            if tok.startswith("NOTE_ON_"):
                try:
                    pitch_class[tid] = int(tok.rsplit("_", 1)[-1], 16) % 12
                except ValueError:
                    pass
        self._cat_idx = cat_idx
        self._pitch_class = pitch_class
        self._num_categories = len(self.CATEGORIES)
        self.reset()

    def reset(self) -> None:
        K = self._num_categories
        self._tp = torch.zeros(K, dtype=torch.long)
        self._fp = torch.zeros(K, dtype=torch.long)
        self._fn = torch.zeros(K, dtype=torch.long)
        self._support = torch.zeros(K, dtype=torch.long)
        self._total_valid = 0
        self._top1_hits = 0
        self._top5_hits = 0
        self._iou_sum = 0.0
        self._iou_count = 0

    @torch.no_grad()
    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> None:
        device = logits.device
        cat_idx = self._cat_idx.to(device)
        pc = self._pitch_class.to(device)

        V = logits.size(-1)
        k = min(5, V)
        top5 = logits.topk(k, dim=-1).indices  # (B, T, k)
        top1 = top5[..., 0]                    # (B, T)

        f_top1 = top1[valid_mask]
        f_top5 = top5[valid_mask]
        f_target = targets[valid_mask]

        n = int(f_target.numel())
        if n == 0:
            return
        self._total_valid += n
        self._top1_hits += int((f_top1 == f_target).sum().item())
        self._top5_hits += int((f_top5 == f_target.unsqueeze(-1)).any(dim=-1).sum().item())

        cat_t = cat_idx[f_target]
        cat_p = cat_idx[f_top1]
        for c in range(self._num_categories):
            tgt_c = cat_t == c
            pred_c = cat_p == c
            self._support[c] += int(tgt_c.sum().item())
            self._tp[c] += int((tgt_c & pred_c).sum().item())
            self._fp[c] += int((~tgt_c & pred_c).sum().item())
            self._fn[c] += int((tgt_c & ~pred_c).sum().item())

        B = logits.size(0)
        for i in range(B):
            valid_i = valid_mask[i]
            if not valid_i.any():
                continue
            t_i = targets[i][valid_i]
            p_i = top1[i][valid_i]
            t_pc = pc[t_i]
            p_pc = pc[p_i]
            t_set = {int(x) for x in t_pc[t_pc >= 0].tolist()}
            p_set = {int(x) for x in p_pc[p_pc >= 0].tolist()}
            if not t_set and not p_set:
                continue
            inter = len(t_set & p_set)
            union = len(t_set | p_set)
            self._iou_sum += inter / max(1, union)
            self._iou_count += 1

    def compute(self) -> dict:
        eps = 1e-9
        tp = self._tp.float()
        fp = self._fp.float()
        fn = self._fn.float()
        support = self._support.float()
        total_support = float(support.sum().item())

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        per_cat: dict[str, dict[str, float]] = {}
        for i, name in enumerate(self.CATEGORIES):
            per_cat[name] = {
                "precision": float(precision[i].item()),
                "recall": float(recall[i].item()),
                "f1": float(f1[i].item()),
                "support": int(support[i].item()),
            }

        if total_support > 0:
            macro_f1 = float(f1.mean().item())
            macro_precision = float(precision.mean().item())
            macro_recall = float(recall.mean().item())
            weighted_f1 = float(((f1 * support).sum() / total_support).item())
            weighted_precision = float(((precision * support).sum() / total_support).item())
            weighted_recall = float(((recall * support).sum() / total_support).item())
            micro_f1 = float(tp.sum().item()) / total_support
        else:
            macro_f1 = macro_precision = macro_recall = 0.0
            weighted_f1 = weighted_precision = weighted_recall = 0.0
            micro_f1 = 0.0

        return {
            "top1_accuracy": self._top1_hits / max(1, self._total_valid),
            "top5_accuracy": self._top5_hits / max(1, self._total_valid),
            "category_micro_f1": micro_f1,
            "category_macro_f1": macro_f1,
            "category_weighted_f1": weighted_f1,
            "category_macro_precision": macro_precision,
            "category_weighted_precision": weighted_precision,
            "category_macro_recall": macro_recall,
            "category_weighted_recall": weighted_recall,
            "pitch_class_iou": self._iou_sum / max(1, self._iou_count),
            "per_category": per_cat,
        }
