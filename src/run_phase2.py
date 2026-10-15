"""Phase2: 近傍法（アイテム型・ユーザー型）。観測テストと本当の好みの両方で、人気順・平均評価順と比べる。

E1 近傍法の4通り（型 × 入力）を、10世界で比べる
E2 ハイパーパラメータ（α・収縮・近傍数）の効きと、「観測テストで選ぶ」と「本当の好みで選ぶ」の違い
E3 ユーザーの活発さ（評価の数）ごとの成績（個人化は、データが少ないと効かない）

使い方: .venv/bin/python src/run_phase2.py
出力:   outputs/phase2_*.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys_eval import (beyond_accuracy, evaluate_observed, evaluate_true, exclusion_mask, item_mean_scores,  # noqa: E402
                         oracle_scores, popularity_scores, random_scores, topk_items)
from recsys_models import item_knn_scores, user_knn_scores  # noqa: E402
from recsys_synth import make_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]
DEFAULT = dict(alpha=0.5, shrink=5.0, k=None)        # 近傍法の既定（通常のコサイン類似度・少し収縮・近傍は全部）


def _ctx(d):
    w = d.world
    tr, val = d.split("train"), d.split("val")
    return w, tr, val, w.n_users, w.n_items, exclusion_mask(w.n_users, w.n_items, tr)


def knn_methods(d, **kw):
    w, tr, val, U, I, ex = _ctx(d)
    p = {**DEFAULT, **kw}
    return {"アイテム型・評価した": item_knn_scores(tr, U, I, "interactions", **p), "アイテム型・好き(4以上)": item_knn_scores(tr, U, I, "liked", **p),
            "ユーザー型・評価した": user_knn_scores(tr, U, I, "interactions", **p), "ユーザー型・好き(4以上)": user_knn_scores(tr, U, I, "liked", **p)}


def _eval(d, S):
    w, tr, val, U, I, ex = _ctx(d)
    tk = topk_items(S, ex, 10)
    ob, tt = evaluate_observed(S, tk, val, U, I), evaluate_true(tk, w.true_rating(), ex)
    ba = beyond_accuracy(tk, I, tr.groupby("item").size().reindex(range(I), fill_value=0).to_numpy(), w.item_genre)
    return ob["ndcg@10"], tt["precision@10"], ba["カバレッジ"]


def e1(seeds=SEEDS):
    acc = {}
    for sd in seeds:
        d = make_dataset(sd)
        w, tr, val, U, I, ex = _ctx(d)
        ms = {"ランダム": random_scores(U, I), "人気順": popularity_scores(tr, U, I), "平均評価順": item_mean_scores(tr, U, I), **knn_methods(d),
              "上限（本当の好み）": oracle_scores(w.true_rating())}
        for name, S in ms.items():
            acc.setdefault(name, []).append(_eval(d, S))
    rows = []
    for name, v in acc.items():
        a = np.array(v)
        se = a.std(axis=0) / np.sqrt(len(a))
        rows.append({"手法": name, "観測NDCG@10": f"{a[:, 0].mean():.4f} ±{se[0]:.4f}", "本当のprecision@10": f"{a[:, 1].mean():.3f} ±{se[1]:.3f}",
                     "カバレッジ": f"{a[:, 2].mean():.2f}"})
    return pd.DataFrame(rows)


GRID = list(itertools.product((0.0, 0.25, 0.5, 0.75, 1.0), (0.0, 5.0, 20.0), (10, 30, 100, None)))


def e2(select_seed=SEEDS[0], eval_seeds=SEEDS[1:]):
    """格子を、1つの世界（開発用）で探索 → 「観測で選んだ設定」と「本当の好みで選んだ設定」を、別の世界で評価する"""
    d = make_dataset(select_seed)
    w, tr, val, U, I, ex = _ctx(d)
    rows = []
    for (kind, mode), (alpha, shrink, k) in itertools.product([("アイテム型", "interactions"), ("アイテム型", "liked"), ("ユーザー型", "interactions"), ("ユーザー型", "liked")], GRID):
        fn = item_knn_scores if kind == "アイテム型" else user_knn_scores
        o, t, _ = _eval(d, fn(tr, U, I, mode, alpha, shrink, k))
        rows.append(dict(型=kind, 入力=mode, alpha=alpha, shrink=shrink, k=k, 観測=o, 真=t))
    grid = pd.DataFrame(rows)
    out = []
    for (kind, mode), g in grid.groupby(["型", "入力"], sort=False):
        for crit, col in (("観測テストで選ぶ", "観測"), ("本当の好みで選ぶ", "真")):
            r = g.loc[g[col].idxmax()]
            k_ = None if pd.isna(r.k) else int(r.k)
            fn = item_knn_scores if kind == "アイテム型" else user_knn_scores
            res = []
            for sd in eval_seeds:
                dd = make_dataset(sd)
                ww, tr2, v2, U2, I2, ex2 = _ctx(dd)
                res.append(_eval(dd, fn(tr2, U2, I2, mode, r.alpha, r.shrink, k_))[:2])
            a = np.array(res).mean(axis=0)
            out.append({"型": kind, "入力": "好き(4以上)" if mode == "liked" else "評価した", "選び方": crit,
                        "選ばれた設定": f"α={r.alpha:g}・収縮={r.shrink:g}・近傍={'全部' if k_ is None else k_}",
                        "別の世界での観測NDCG": round(a[0], 4), "別の世界での本当のprecision": round(a[1], 3)})
    return grid, pd.DataFrame(out)


def e3(seeds=SEEDS):
    bins = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 10000)]
    acc = {b: {} for b in bins}
    for sd in seeds:
        d = make_dataset(sd)
        w, tr, val, U, I, ex = _ctx(d)
        act = tr.groupby("user").size().reindex(range(U), fill_value=0).to_numpy()
        rel = (w.true_rating() >= 4) & ~ex
        ms = {"人気順": popularity_scores(tr, U, I), "平均評価順": item_mean_scores(tr, U, I), **knn_methods(d)}
        for name, S in ms.items():
            hits = np.take_along_axis(rel, topk_items(S, ex, 10), axis=1).mean(axis=1)
            for b in bins:
                m = (act >= b[0]) & (act < b[1])
                acc[b].setdefault(name, []).append(float(hits[m].mean()))
        for b in bins:
            m = (act >= b[0]) & (act < b[1])
            acc[b].setdefault("ユーザー数", []).append(int(m.sum()))
    rows = []
    for b in bins:
        r = {"学習期間の評価数": f"{b[0]}〜{b[1] - 1}" if b[1] < 10000 else f"{b[0]}以上", "ユーザー数(平均)": int(np.mean(acc[b]["ユーザー数"]))}
        r.update({n: round(float(np.mean(v)), 3) for n, v in acc[b].items() if n != "ユーザー数"})
        rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20)
    df = e1(); print("== E1 =="); print(df.to_string(index=False)); df.to_csv(ROOT / "outputs" / "phase2_E1.csv", index=False)
    grid, sel = e2(); print("\n== E2（選び方の違い）=="); print(sel.to_string(index=False))
    grid.to_csv(ROOT / "outputs" / "phase2_E2_grid.csv", index=False); sel.to_csv(ROOT / "outputs" / "phase2_E2.csv", index=False)
    df = e3(); print("\n== E3 =="); print(df.to_string(index=False)); df.to_csv(ROOT / "outputs" / "phase2_E3.csv", index=False)
