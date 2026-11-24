"""CAFA 标准评测：Fmax / AUPR / 分层 Recall@EC level（方案 5.2）。"""
from __future__ import annotations

import numpy as np


def fmax(scores: list[dict[str, float]], labels: list[set[str]],
         thresholds: np.ndarray | None = None) -> tuple[float, float, float, float | None]:
    """**micro-F1**（全体 TP/FP/FN 汇总），不是 CAFA 蛋白中心 Fmax。

    保留供历史结果对照与单元测试；论文主表口径请用 fmax_protein_centric()。
    返回 (micro_f1, precision, recall, best_threshold)。
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 1.0, 100)
    best = (-1.0, 0.0, 0.0, None)
    for t in thresholds:
        tp = fp = fn = 0
        for s, lab in zip(scores, labels):
            pred = {ec for ec, p in s.items() if p >= t}
            tp += len(pred & lab)
            fp += len(pred - lab)
            fn += len(lab - pred)
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f = 2 * prec * rec / (prec + rec)
        if f > best[0]:
            best = (f, prec, rec, float(t))
    # 所有阈值都无 TP（如未训练模型）时返回 0，不能把哨兵值 -1 写进表格
    return best if best[0] >= 0 else (0.0, 0.0, 0.0, None)


def fmax_protein_centric(scores: list[dict[str, float]], labels: list[set[str]],
                         thresholds: np.ndarray | None = None
                         ) -> tuple[float, float, float, float | None]:
    """CAFA 蛋白中心 Fmax：先逐蛋白算 precision/recall，再在蛋白维平均。

    pr(t) = 1/m(t) * Σ_i |P_i∩T_i|/|P_i|，m(t) 为有 ≥1 个预测的蛋白数；
    rc(t) = 1/n   * Σ_i |P_i∩T_i|/|T_i|，n 为有真值的蛋白数；
    Fmax  = max_t 2*pr*rc/(pr+rc)。

    无真值样本（如仅有部分 EC 的 completion 样本）由调用方分任务排除，
    不应作为假阳性进入 exact-EC4 分母。
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 1.0, 100)
    with_truth = [(s, lab) for s, lab in zip(scores, labels) if lab]
    n = len(with_truth)
    if n == 0:
        return 0.0, 0.0, 0.0, None
    best = (-1.0, 0.0, 0.0, None)
    for t in thresholds:
        pr_sum, rc_sum, m = 0.0, 0.0, 0
        for s, lab in with_truth:
            pred = {ec for ec, p in s.items() if p >= t}
            hit = len(pred & lab)
            rc_sum += hit / len(lab)
            if pred:
                m += 1
                pr_sum += hit / len(pred)
        pr = pr_sum / m if m else 0.0
        rc = rc_sum / n
        f = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
        if f > best[0]:
            best = (f, pr, rc, float(t))
    return best if best[0] >= 0 else (0.0, 0.0, 0.0, None)


def aupr_macro_by_class(scores: list[dict[str, float]],
                        labels: list[set[str]]) -> float:
    """按 EC 类别计算 AP 后平均（真正的 class-macro AUPR）。

    词表 = 真值类别 ∪ 预测类别；未出现的类别分数按 0 计，
    避免只统计「真值 + 已保留预测」造成的人为抬高。
    """
    from sklearn.metrics import average_precision_score
    vocab = sorted({ec for lab in labels for ec in lab} |
                   {ec for s in scores for ec in s})
    if not vocab or not scores:
        return 0.0
    idx = {ec: j for j, ec in enumerate(vocab)}
    y_true = np.zeros((len(scores), len(vocab)), dtype=np.int8)
    y_score = np.zeros((len(scores), len(vocab)), dtype=np.float32)
    for i, (s, lab) in enumerate(zip(scores, labels)):
        for ec in lab:
            j = idx.get(ec)
            if j is not None:
                y_true[i, j] = 1
        for ec, p in s.items():
            j = idx.get(ec)
            if j is not None:
                y_score[i, j] = p
    aps = [average_precision_score(y_true[:, j], y_score[:, j])
           for j in range(len(vocab)) if y_true[:, j].sum() > 0]
    return float(np.mean(aps)) if aps else 0.0


def aupr(scores: list[dict[str, float]], labels: list[set[str]], micro: bool = True):
    """PR 曲线面积（micro 或 macro-average）。"""
    from sklearn.metrics import average_precision_score
    if micro:
        y_true, y_score = [], []
        vocab = sorted({ec for lab in labels for ec in lab})
        for s, lab in zip(scores, labels):
            y_true.append([1 if ec in lab else 0 for ec in vocab])
            y_score.append([s.get(ec, 0.0) for ec in vocab])
        return float(average_precision_score(np.array(y_true).ravel(),
                                             np.array(y_score).ravel()))
    aps = []
    for s, lab in zip(scores, labels):
        keys = list(lab) + [k for k in s if k not in lab]
        y = [1 if k in lab else 0 for k in keys]
        if sum(y) == 0:
            continue
        aps.append(average_precision_score(y, [s.get(k, 0.0) for k in keys]))
    return float(np.mean(aps)) if aps else 0.0


def recall_at_level(scores: list[dict[str, float]], labels: list[set[str]],
                    level: int = 1, top_k: int = 1) -> float:
    """按 EC 层级的 Recall@top-K（截断到指定层级比较，如 level=1 只看大类）。"""
    def trunc(ec: str) -> str:
        return ".".join(ec.split(".")[:level])

    hit = tot = 0
    for s, lab in zip(scores, labels):
        top = sorted(s.items(), key=lambda kv: -kv[1])[:top_k]
        pred = {trunc(ec) for ec, _ in top}
        true = {trunc(ec) for ec in lab}
        hit += len(pred & true)
        tot += len(true)
    return hit / max(tot, 1)


def zeroshot_recall(scores: list[dict[str, float]], labels: list[set[str]],
                    unseen_ecs: set[str], top_k: int = 10) -> float:
    """零样本召回：仅在 hold-out EC 集合上统计 Top-K 命中率。"""
    hit = tot = 0
    for s, lab in zip(scores, labels):
        target = lab & unseen_ecs
        if not target:
            continue
        top = {ec for ec, _ in sorted(s.items(), key=lambda kv: -kv[1])[:top_k]}
        hit += len(top & target)
        tot += len(target)
    return hit / max(tot, 1)


def hit_at_k(scores: list[dict[str, float]], labels: list[set[str]],
             top_k: int = 5) -> float:
    """蛋白级命中率：Top-K 预测中至少命中 1 个真实 EC 的样本占比。

    对标 CLEAN 的 promiscuous / multi-functional 基准口径（每个蛋白算一次）。
    """
    hit = 0
    for s, lab in zip(scores, labels):
        if not lab:
            continue
        top = {ec for ec, _ in sorted(s.items(), key=lambda kv: -kv[1])[:top_k]}
        hit += int(bool(top & lab))
    return hit / max(len(labels), 1)


def ec_multi_mask(labels: list[set[str]]) -> list[bool]:
    """多功能酶掩码（真实标签数 > 1）。"""
    return [len(lab) > 1 for lab in labels]


def subset_metrics(scores, labels, mask: list[bool], name: str,
                   top_ks: tuple[int, ...] = (1, 3, 5)) -> dict:
    """在子集（如单功能/多功能）上算一套指标。"""
    idx = [i for i, m in enumerate(mask) if m and labels[i]]
    if not idx:
        return {}
    sub_s = [scores[i] for i in idx]
    sub_l = [labels[i] for i in idx]
    out = {f"{name}/n": len(idx)}
    out[f"{name}/fmax"] = fmax_protein_centric(sub_s, sub_l)[0]
    out[f"{name}/fmax_micro"] = fmax(sub_s, sub_l)[0]
    out[f"{name}/aupr_macro"] = aupr_macro_by_class(sub_s, sub_l)
    out[f"{name}/aupr_perprotein"] = aupr(sub_s, sub_l, micro=False)
    out[f"{name}/aupr_micro"] = aupr(sub_s, sub_l, micro=True)
    out[f"{name}/n_labels"] = int(sum(len(l) for l in sub_l))
    for k in top_ks:
        out[f"{name}/hit@{k}"] = hit_at_k(sub_s, sub_l, top_k=k)
    return out


def _prefix_match(ec: str, prefixes: set[str]) -> bool:
    """完整 EC4 是否落在任一部分 EC 前缀（如 3.4.24）之下。"""
    parts = ec.split(".")
    return any(".".join(parts[:i]) in prefixes for i in (1, 2, 3, 4))


def completion_metrics(scores, partial_labels: list[set[str]],
                       top_ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """EC-Bench "EC number completion" 任务：部分 EC（x.x.x.-）补全第 4 位。

    真值只有前缀（如 3.4.24），只要 Top-K 预测中有任一完整 EC 落在该前缀下即算命中。
    partial_labels[i] 为空表示该蛋白不是 completion 样本，跳过。
    """
    idx = [i for i, p in enumerate(partial_labels) if p]
    out = {"completion/n": len(idx)}
    for k in top_ks:
        hit = 0
        for i in idx:
            top = [ec for ec, _ in sorted(scores[i].items(), key=lambda kv: -kv[1])[:k]]
            hit += int(any(_prefix_match(ec, partial_labels[i]) for ec in top))
        out[f"completion/hit@{k}"] = hit / max(len(idx), 1)
    return out


def recommendation_metrics(scores, labels,
                           top_ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """EC-Bench "EC number recommendation" 任务：Top-K 中给出候选 EC。

    蛋白级 hit@k（至少一个命中）与标签级 recall@k（命中真实 EC 的比例）。
    """
    out = {}
    for k in top_ks:
        hit = rec = n_lab = 0
        for s, lab in zip(scores, labels):
            if not lab:
                continue
            top = {ec for ec, _ in sorted(s.items(), key=lambda kv: -kv[1])[:k]}
            hit += int(bool(top & lab))
            rec += len(top & lab)
            n_lab += len(lab)
        out[f"recommendation/hit@{k}"] = hit / max(len(labels), 1)
        out[f"recommendation/recall@{k}"] = rec / max(n_lab, 1)
    return out


def evaluate_all(scores, labels, unseen_ecs: set[str] | None = None,
                 multi_ec: list[bool] | None = None, n_boot: int = 0,
                 partial_labels: list[set[str]] | None = None,
                 score_protocol: str = "unspecified",
                 score_truncation_k: int | None = None) -> dict:
    """exact-EC4 主指标 + completion 分任务（对齐 CAFA / EC-Bench 口径）。

    口径修订（2026-11-08 审计）：
      - `fmax` 改为 CAFA **蛋白中心 Fmax**；旧的全局 micro-F1 保留为 `fmax_micro`。
      - `aupr_macro` 改为按 EC **类别** 的 macro-AUPR；旧的逐蛋白 AP 平均
        保留为 `aupr_perprotein`，便于追溯历史结果。
      - 修正 `hierarchical_fmax` 的返回顺序错位（原 `hf` 实际写入的是 hR）。
      - 无完整 EC4 真值的样本（如仅部分 EC）不再进入 exact 指标的 FP 分母，
        改由 `completion/*` 单独报告；`n_exact` / `n_no_exact_label` 记录分任务计数。
      - `score_protocol` 显式声明预测分数是否覆盖完整词表：
        `full_scores` / `truncated_topk` / `unspecified`。截断分数会低估
        低阈值处的 FP、抬高 precision/Fmax，并使 AUPR 排序失真；只有
        `full_scores` 的结果标注为可用于论文主表。
    """
    # 只在完整 EC4（叶子）上评测，层级中间节点仅用于训练监督
    scores = leaf_only(scores)
    n_total = len(labels)
    exact_idx = [i for i, lab in enumerate(labels) if lab]
    ex_s = [scores[i] for i in exact_idx]
    ex_l = [labels[i] for i in exact_idx]
    warnings = []
    if score_protocol == "truncated_topk":
        warnings.append(
            "预测分数仅保留 top-K 类别：低阈值处的 FP 被低估、precision/Fmax 偏高；"
            "若生成侧又补回了真值类别分数，class-macro AUPR 会因负例分数被置 0 "
            "而人为抬高甚至接近 1。该结果仅作趋势对照，不可用于论文主表。")
    elif score_protocol != "full_scores":
        warnings.append(
            "调用方未声明分数完整性协议，按截断/不完整分数处理；不可用于论文主表。")
    out = {"metric_protocol": "cafa_protein_centric",
           "score_protocol": score_protocol,
           "score_truncation_k": score_truncation_k,
           "score_vocab_size": len({ec for s in scores for ec in s}),
           "metrics_publication_ready": score_protocol == "full_scores",
           "metric_warnings": warnings,
           "n_total": n_total, "n_exact": len(exact_idx),
           "n_no_exact_label": n_total - len(exact_idx)}
    if partial_labels and any(partial_labels):
        out.update(completion_metrics(scores, partial_labels))
    if not ex_s:
        out.update({"fmax": 0.0, "precision_at_fmax": 0.0, "recall_at_fmax": 0.0,
                    "threshold_at_fmax": 0.5, "fmax_micro": 0.0,
                    "aupr_micro": 0.0, "aupr_macro": 0.0, "aupr_perprotein": 0.0})
        return out

    f, p, r, t = fmax_protein_centric(ex_s, ex_l)
    t = t if t is not None else 0.5
    out.update({"fmax": f, "precision_at_fmax": p, "recall_at_fmax": r,
                "threshold_at_fmax": t,
                "fmax_micro": fmax(ex_s, ex_l)[0],
                "aupr_micro": aupr(ex_s, ex_l, micro=True),
                "aupr_macro": aupr_macro_by_class(ex_s, ex_l),
                "aupr_perprotein": aupr(ex_s, ex_l, micro=False)})
    for lv in (1, 2, 3, 4):
        out[f"recall@{lv}"] = recall_at_level(ex_s, ex_l, level=lv, top_k=1)
        out[f"acc@L{lv}"] = accuracy_at_level(ex_s, ex_l, level=lv, top_k=1)
        lf = level_f1(ex_s, ex_l, level=lv, threshold=t)
        out[f"f1_micro@L{lv}"] = lf["micro"]
        out[f"f1_macro@L{lv}"] = lf["macro"]
        out[f"f1_weighted@L{lv}"] = lf["weighted"]

    hf, hp, hr, _ = hierarchical_fmax(ex_s, ex_l)
    out.update({"hp": hp, "hr": hr, "hf": hf})

    cov, sim_avg, sim_w = similarity_metrics(ex_s, ex_l, threshold=t)
    out.update({"coverage": cov, "sim_avg": sim_avg, "sim_weighted": sim_w,
                "mean_pred_labels": mean_pred_labels(ex_s, threshold=t),
                "mrr": mrr(ex_s, ex_l)})
    if unseen_ecs:
        out["zeroshot_recall@10"] = zeroshot_recall(ex_s, ex_l, unseen_ecs, top_k=10)
    if n_boot:
        lo, hi = bootstrap_ci(lambda s, l: fmax_protein_centric(s, l)[0],
                              ex_s, ex_l, n_boot=n_boot)
        out["fmax_ci95"] = (lo, hi)

    out.update(recommendation_metrics(ex_s, ex_l))
    mask = ([multi_ec[i] for i in exact_idx] if multi_ec is not None
            else ec_multi_mask(ex_l))
    out.update(subset_metrics(ex_s, ex_l, mask, "multi_ec"))
    out.update(subset_metrics(ex_s, ex_l, [not m for m in mask], "single_ec"))
    return out


# ---------------------------------------------------------------- 指标补齐 ----
def _is_leaf(ec: str) -> bool:
    """完整四级 EC（可参与 exact EC4 评测）。"""
    return ec.count(".") == 3 and "-" not in ec


def leaf_only(scores: list[dict[str, float]]) -> list[dict[str, float]]:
    """过滤掉层级中间节点（如 3.4.24），只保留 EC4 叶子。"""
    return [{ec: p for ec, p in s.items() if _is_leaf(ec)} for s in scores]


def _ancestors(ec: str) -> set[str]:
    parts = ec.split(".")
    return {".".join(parts[:i]) for i in range(1, len(parts))}


def coverage(scores: list[dict[str, float]], threshold: float) -> float:
    """EC-Bench Coverage：至少给出 1 个 ≥threshold 预测的蛋白占比。"""
    n = sum(1 for s in scores if any(p >= threshold for p in s.values()))
    return n / max(len(scores), 1)


def mean_pred_labels(scores: list[dict[str, float]], threshold: float) -> float:
    """平均预测标签数（压多标签退化解的诊断指标）。"""
    return float(np.mean([sum(1 for p in s.values() if p >= threshold) for s in scores])) \
        if scores else 0.0


def level_f1(scores, labels, level: int, threshold: float) -> dict:
    """按层级截断后的多标签 micro / macro / weighted F1（HIT-EC、EC-Bench 口径）。"""
    def trunc(ec: str) -> str:
        return ".".join(ec.split(".")[:level])

    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}
    TP = FP = FN = 0
    for s, lab in zip(scores, labels):
        pred = {trunc(ec) for ec, p in s.items() if p >= threshold}
        true = {trunc(ec) for ec in lab}
        for c in pred & true:
            tp[c] = tp.get(c, 0) + 1
        for c in pred - true:
            fp[c] = fp.get(c, 0) + 1
        for c in true - pred:
            fn[c] = fn.get(c, 0) + 1
        TP += len(pred & true); FP += len(pred - true); FN += len(true - pred)

    micro = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) else 0.0
    f1s, supports = [], []
    for c in set(tp) | set(fn):
        ti, fi = tp.get(c, 0), fn.get(c, 0)
        pr = tp.get(c, 0) / (tp.get(c, 0) + fp.get(c, 0)) if (tp.get(c, 0) + fp.get(c, 0)) else 0.0
        rc = ti / (ti + fi) if (ti + fi) else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if (pr + rc) else 0.0)
        supports.append(ti + fi)
    macro = float(np.mean(f1s)) if f1s else 0.0
    tot = sum(supports) or 1
    weighted = float(sum(f * s_ for f, s_ in zip(f1s, supports)) / tot)
    return {"micro": micro, "macro": macro, "weighted": weighted}


def accuracy_at_level(scores, labels, level: int, top_k: int = 1) -> float:
    """蛋白级 Accuracy@K：Top-K 截断预测中至少命中一个该层真实类的样本占比。"""
    def trunc(ec: str) -> str:
        return ".".join(ec.split(".")[:level])

    hit = 0
    for s, lab in zip(scores, labels):
        top = [trunc(ec) for ec, _ in sorted(s.items(), key=lambda kv: -kv[1])[:top_k]]
        true = {trunc(ec) for ec in lab}
        hit += int(bool(set(top) & true))
    return hit / max(len(labels), 1)


def hierarchical_fmax(scores, labels, thresholds: np.ndarray | None = None):
    """CAFA 式层级指标：预测/真值均扩展到祖先闭包（EC 前缀即祖先）。

    TP = |pred_exp ∩ true_exp|；hp = TP/|pred_exp|；hr = TP/|true_exp|。
    返回 (hf, hp, hr, best_threshold)。
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 1.0, 100)
    exp_lab = [ {ec for ec in lab} | set().union(*(_ancestors(ec) for ec in lab))
                if lab else set() for lab in labels]
    best = (-1.0, 0.0, 0.0, None)
    for t in thresholds:
        tp = p_den = r_den = 0
        for s, lab in zip(scores, exp_lab):
            pred_exp = {ec for ec, p in s.items() if p >= t}
            pred_exp |= set().union(*(_ancestors(ec) for ec in pred_exp)) if pred_exp else set()
            tp += len(pred_exp & lab)
            p_den += len(pred_exp)
            r_den += len(lab)
        if tp == 0 or p_den == 0:
            continue
        hp, hr = tp / p_den, tp / r_den
        hf = 2 * hp * hr / (hp + hr)
        if hf > best[0]:
            best = (hf, hp, hr, float(t))
    return best if best[0] >= 0 else (0.0, 0.0, 0.0, None)


def ec_similarity(e1: str, e2: str) -> float:
    """EC-Bench 风格层级相似度：公共前缀层数 / 4（完全相同 = 1.0）。"""
    n = 0
    for x, y in zip(e1.split("."), e2.split(".")):
        if x != y:
            break
        n += 1
    return n / 4


def similarity_metrics(scores, labels, threshold: float) -> tuple[float, float, float]:
    """返回 (coverage, average_similarity, weighted_similarity)。

    每个蛋白取 pred×true 的最大相似度；无预测的蛋白计 0（等价于
    EC-Bench 的 average×coverage 加权形式）。
    """
    cov = coverage(scores, threshold)
    sims = []
    for s, lab in zip(scores, labels):
        pred = [ec for ec, p in s.items() if p >= threshold]
        sims.append(max((ec_similarity(a, b) for a in pred for b in lab), default=0.0))
    sim_avg = float(np.mean(sims)) if sims else 0.0
    return cov, sim_avg, sim_avg * cov


def mrr(scores, labels) -> float:
    """平均倒数排名：最佳真实 EC 在排序中的 1/rank。"""
    rr = 0.0
    for s, lab in zip(scores, labels):
        for i, ec in enumerate(sorted(s, key=lambda k: -s[k])):
            if ec in lab:
                rr += 1 / (i + 1)
                break
    return rr / max(len(labels), 1)


def bootstrap_ci(metric_fn, scores, labels, n_boot: int = 200,
                 seed: int = 0) -> tuple[float, float]:
    """对样本重采样得到指标的 95% 置信区间（HIT-EC 式统计可靠性）。"""
    rng = np.random.default_rng(seed)
    n = len(scores)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn([scores[i] for i in idx], [labels[i] for i in idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def main_table_row(result: dict, method: str) -> str:
    """论文主表一行（Markdown）：全量 / 单功能 / 多功能 三列。"""
    def g(k: str, d: int = 3) -> str:
        v = result.get(k)
        return "—" if v is None else f"{v:.{d}f}"

    return (f"| {method} | {g('fmax')} | {g('aupr_macro')} | {g('recall@1')} | "
            f"{g('single_ec/fmax')} | {g('multi_ec/hit@5')} | {g('multi_ec/fmax')} |")

