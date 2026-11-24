"""Unified Dual-Protocol Enzyme EC Evaluation Metrics.

Combines TWO standard evaluation protocols for top-journal enzyme EC benchmarking:
1. CAFA / EC-Bench Multi-Label Protocol (threshold-free & protein-centric):
   - Protein-centric Fmax, Precision@Fmax, Recall@Fmax, best_threshold
   - Class-Macro AUPR, Micro AUPR
   - Macro-F1@L4 (at Fmax threshold), Recall@1
   - Hierarchical EC1 / EC2 / EC3 / EC4 Fmax
   - Rare (1-5 train count) Fmax
2. CLEAN / EnzymeHunter / GraphEC Enzyme Protocol (class-weighted & locked-threshold):
   - Weighted-F1, Weighted-Precision, Weighted-Recall (sklearn average="weighted" on multi-label indicator, matching CLEAN Science 2023 get_eval_metrics)
   - Micro-F1, Macro-F1 (at fixed/locked threshold tau)
   - Weighted-ROC-AUC (over classes with >=1 positive)
   - Exact-Match Accuracy (subset accuracy)

Supports both continuous probability/score matrices (N, C) and discrete/sparse predictions
(such as CLEAN Max-Separation outputs).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _ec_prefix(ec: str, level: int) -> str:
    parts = ec.split(".")
    return ".".join(parts[:level])


def compute_dual_ec_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    ec_list: list[str],
    train_counts: dict[str, int] | None = None,
    locked_threshold: float | None = None,
    thresholds: np.ndarray | None = None,
) -> dict:
    """Compute both CAFA/EC-Bench and CLEAN/EnzymeHunter enzyme EC metrics.

    Args:
        probs: (N, C) float32/float64 score/probability matrix in [0, 1] (or arbitrary scores).
        y_true: (N, C) binary int/float ground-truth indicator matrix.
        ec_list: length-C list of EC4 strings (e.g. "1.1.1.1").
        train_counts: optional dict {ec4: train_count} for Rare (1-5) slice.
        locked_threshold: if provided, CLEAN-protocol discrete metrics (Weighted-F1/P/R,
            Micro-F1, Macro-F1, ExactMatch) are computed at this exact fixed threshold
            (zero test-set threshold tuning!). If None, uses the val-optimal Fmax threshold.
    """
    probs = np.asarray(probs, dtype=np.float32)
    y_true = (np.asarray(y_true) > 0.5).astype(np.int8)
    N, C = probs.shape
    assert y_true.shape == (N, C) and len(ec_list) == C

    # Filter to samples with >= 1 ground-truth EC4
    has_truth = y_true.sum(axis=1) > 0
    if not np.all(has_truth):
        probs = probs[has_truth]
        y_true = y_true[has_truth]
        N = probs.shape[0]

    if thresholds is None:
        pmin, pmax = float(probs.min()), float(probs.max())
        if pmin >= 0.0 and pmax <= 1.0001:
            thresholds = np.linspace(0.02, 0.98, 49)
        else:
            thresholds = np.quantile(probs.max(axis=1), np.linspace(0.02, 0.98, 49))

    # --- 1. Protein-centric Fmax sweep (vectorized) ---
    true_counts = y_true.sum(axis=1).astype(np.float32)  # (N,)
    best_fmax = -1.0
    best_pr = 0.0
    best_rc = 0.0
    best_tau = 0.5

    for t in thresholds:
        pred = (probs >= t)
        pred_counts = pred.sum(axis=1)
        hits = (pred & (y_true > 0)).sum(axis=1).astype(np.float32)
        rc = float((hits / true_counts).mean())
        cov = pred_counts > 0
        pr = float((hits[cov] / pred_counts[cov]).mean()) if cov.any() else 0.0
        f = (2.0 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
        if f > best_fmax:
            best_fmax, best_pr, best_rc, best_tau = f, pr, rc, float(t)

    eval_tau = float(locked_threshold) if locked_threshold is not None else best_tau
    y_pred = (probs >= eval_tau).astype(np.int8)

    # --- 2. Recall@1 ---
    top1_idx = probs.argmax(axis=1)
    recall_at_1 = float(y_true[np.arange(N), top1_idx].mean())

    # --- 3. AUPR-Macro, AUPR-Micro, Weighted-ROC-AUC ---
    pos_cols = y_true.sum(axis=0) > 0
    neg_cols = y_true.sum(axis=0) < N
    valid_ap_cols = np.where(pos_cols)[0]
    valid_auc_cols = np.where(pos_cols & neg_cols)[0]

    aps = []
    for j in valid_ap_cols:
        aps.append(average_precision_score(y_true[:, j], probs[:, j]))
    aupr_macro = float(np.mean(aps)) if aps else 0.0
    aupr_micro = float(average_precision_score(y_true.ravel(), probs.ravel()))

    aucs = []
    auc_weights = []
    for j in valid_auc_cols:
        aucs.append(roc_auc_score(y_true[:, j], probs[:, j]))
        auc_weights.append(int(y_true[:, j].sum()))
    weighted_auc = float(np.average(aucs, weights=auc_weights)) if aucs else 0.0
    macro_auc = float(np.mean(aucs)) if aucs else 0.0

    # --- 4. CLEAN / EnzymeHunter Protocol (Weighted P/R/F1, Micro/Macro F1 at eval_tau) ---
    # Only evaluate class-weighted metrics over classes with >=1 true positive (CLEAN convention)
    yt_pos = y_true[:, valid_ap_cols]
    yp_pos = y_pred[:, valid_ap_cols]
    weighted_f1 = float(f1_score(yt_pos, yp_pos, average="weighted", zero_division=0))
    weighted_prec = float(precision_score(yt_pos, yp_pos, average="weighted", zero_division=0))
    weighted_rec = float(recall_score(yt_pos, yp_pos, average="weighted", zero_division=0))
    macro_f1_L4 = float(f1_score(yt_pos, yp_pos, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    exact_match = float(np.all(y_true == y_pred, axis=1).mean())

    # --- 5. Hierarchical EC1 / EC2 / EC3 Fmax (max-pooling child EC4 scores) ---
    hier_fmax = {}
    for lvl in (1, 2, 3):
        pref_map: dict[str, list[int]] = {}
        for j, ec in enumerate(ec_list):
            p = _ec_prefix(ec, lvl)
            pref_map.setdefault(p, []).append(j)
        prefs = list(pref_map.keys())
        P_lvl = np.zeros((N, len(prefs)), dtype=np.float32)
        Y_lvl = np.zeros((N, len(prefs)), dtype=np.int8)
        for k, p in enumerate(prefs):
            cols = pref_map[p]
            P_lvl[:, k] = probs[:, cols].max(axis=1)
            Y_lvl[:, k] = y_true[:, cols].max(axis=1)
        tc = Y_lvl.sum(axis=1).astype(np.float32)
        bf = 0.0
        for t in thresholds[::2]:
            pr_m = (P_lvl >= t)
            pc = pr_m.sum(axis=1)
            h = (pr_m & (Y_lvl > 0)).sum(axis=1).astype(np.float32)
            rc = float((h / tc).mean())
            cov = pc > 0
            pr = float((h[cov] / pc[cov]).mean()) if cov.any() else 0.0
            f = (2.0 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
            if f > bf:
                bf = f
        hier_fmax[f"ec{lvl}_fmax"] = round(bf, 4)

    # --- 6. Rare (1-5 train count) Fmax ---
    rare_fmax = None
    rare_n = 0
    if train_counts is not None:
        rare_cols = np.array([1 <= train_counts.get(ec, 0) <= 5 for ec in ec_list], dtype=bool)
        if rare_cols.any():
            rare_rows = y_true[:, rare_cols].sum(axis=1) > 0
            rare_n = int(rare_rows.sum())
            if rare_n > 0:
                P_r = probs[rare_rows]
                Y_r = y_true[rare_rows]
                tc_r = Y_r.sum(axis=1).astype(np.float32)
                bf = 0.0
                for t in thresholds:
                    pr_m = (P_r >= t)
                    pc = pr_m.sum(axis=1)
                    h = (pr_m & (Y_r > 0)).sum(axis=1).astype(np.float32)
                    rc = float((h / tc_r).mean())
                    cov = pc > 0
                    pr = float((h[cov] / pc[cov]).mean()) if cov.any() else 0.0
                    f = (2.0 * pr * rc / (pr + rc)) if (pr + rc) > 0 else 0.0
                    if f > bf:
                        bf = f
                rare_fmax = round(bf, 4)

    return {
        "n": N,
        "n_active_ec4": int(len(valid_ap_cols)),
        # CAFA / EC-Bench protocol
        "fmax": round(best_fmax, 4),
        "fmax_prec": round(best_pr, 4),
        "fmax_rec": round(best_rc, 4),
        "best_tau": round(best_tau, 4),
        "eval_tau": round(eval_tau, 4),
        "aupr_macro": round(aupr_macro, 4),
        "aupr_micro": round(aupr_micro, 4),
        "recall@1": round(recall_at_1, 4),
        **hier_fmax,
        "rare_fmax": rare_fmax,
        "rare_n": rare_n,
        # CLEAN / EnzymeHunter / GraphEC protocol
        "weighted_f1": round(weighted_f1, 4),
        "weighted_prec": round(weighted_prec, 4),
        "weighted_rec": round(weighted_rec, 4),
        "weighted_auc": round(weighted_auc, 4),
        "macro_auc": round(macro_auc, 4),
        "macro_f1_L4": round(macro_f1_L4, 4),
        "micro_f1": round(micro_f1, 4),
        "exact_match": round(exact_match, 4),
    }


def compute_clean_discrete_metrics(
    pred_sets: list[set[str]],
    true_sets: list[set[str]],
) -> dict:
    """Exact CLEAN-protocol discrete set metrics (for e.g. CLEAN Max-Separation outputs)."""
    vocab = sorted({ec for s in true_sets for ec in s})
    idx = {ec: j for j, ec in enumerate(vocab)}
    N, C = len(true_sets), len(vocab)
    Y = np.zeros((N, C), dtype=np.int8)
    P = np.zeros((N, C), dtype=np.int8)
    for i, (ps, ts) in enumerate(zip(pred_sets, true_sets)):
        for ec in ts:
            if ec in idx:
                Y[i, idx[ec]] = 1
        for ec in ps:
            if ec in idx:
                P[i, idx[ec]] = 1
    return {
        "n": N,
        "weighted_f1": round(float(f1_score(Y, P, average="weighted", zero_division=0)), 4),
        "weighted_prec": round(float(precision_score(Y, P, average="weighted", zero_division=0)), 4),
        "weighted_rec": round(float(recall_score(Y, P, average="weighted", zero_division=0)), 4),
        "macro_f1_L4": round(float(f1_score(Y, P, average="macro", zero_division=0)), 4),
        "micro_f1": round(float(f1_score(Y, P, average="micro", zero_division=0)), 4),
        "exact_match": round(float(np.all(Y == P, axis=1).mean()), 4),
    }
