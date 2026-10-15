"""Phase3: 行列分解（ALS・BPR）。評価の点数を使う個人化が、効くかを確かめる。

E1 ALS の因子数 k と正則化 reg の効き（RMSE・観測NDCG・本当のprecision）
E2 手法の比較（世界10回）: 基準・近傍法・ALS・BPR（外し役の選び方2通り）
E3 学習した因子は、真の因子に近いか。データ（評価の数）が増えると、どうなるか
E4 ハイパーパラメータを、何で選ぶか（RMSE／観測NDCG／本当）

使い方: .venv/bin/python src/run_phase3.py
出力:   outputs/phase3_*.csv
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys_eval import (beyond_accuracy, evaluate_observed, evaluate_true, exclusion_mask, item_mean_scores,  # noqa: E402
                         oracle_scores, popularity_scores, rating_errors, rating_rmse_all, topk_items)
from recsys_models import als_fit, bpr_fit, item_knn_scores  # noqa: E402
from recsys_synth import make_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]
ALS_DEFAULT = dict(k=4, reg=5.0)                       # E1・E4 で選んだ設定
BPR_DEFAULT = dict(k=16, lr=0.05, reg=0.01, epochs=40)


def _ctx(d):
    w = d.world
    tr, val = d.split("train"), d.split("val")
    return w, tr, val, w.n_users, w.n_items, exclusion_mask(w.n_users, w.n_items, tr)


def _eval(d, S, rmse_pred=None):
    w, tr, val, U, I, ex = _ctx(d)
    tk = topk_items(S, ex, 10)
    ba = beyond_accuracy(tk, I, tr.groupby("item").size().reindex(range(I), fill_value=0).to_numpy(), w.item_genre)
    return dict(obs=evaluate_observed(S, tk, val, U, I)["ndcg@10"], true=evaluate_true(tk, w.true_rating(), ex)["precision@10"],
                cov=ba["カバレッジ"], rmse=rating_errors(S, val)["RMSE"] if rmse_pred else np.nan)


def e1(seed=SEEDS[0], ks=(1, 2, 4, 8, 16, 32), regs=(0.5, 1, 2, 5, 10, 20)):
    d = make_dataset(seed)
    w, tr, val, U, I, ex = _ctx(d)
    rows = []
    for k, reg in itertools.product(ks, regs):
        P = als_fit(tr, U, I, k=k, reg=reg).predict()
        r = _eval(d, P, rmse_pred=True)
        rows.append(dict(k=k, reg=reg, RMSE=r["rmse"], 観測NDCG=r["obs"], 本当のprecision=r["true"]))
    return pd.DataFrame(rows)


def e2(seeds=SEEDS):
    acc = {}
    for sd in seeds:
        d = make_dataset(sd)
        w, tr, val, U, I, ex = _ctx(d)
        ms = {"人気順": (popularity_scores(tr, U, I), False), "平均評価順": (item_mean_scores(tr, U, I), False),
              "近傍法（アイテム型・好き）": (item_knn_scores(tr, U, I, "liked", 0.5, 5.0), False),
              "ALS（評価の点数・k=4）": (als_fit(tr, U, I, **ALS_DEFAULT).predict(), True),
              "BPR（外し役：一様）": (bpr_fit(tr, U, I, **BPR_DEFAULT, neg_pop_power=0.0, seed=sd).predict(), False),
              "BPR（外し役：人気に比例）": (bpr_fit(tr, U, I, **BPR_DEFAULT, neg_pop_power=1.0, seed=sd).predict(), False),
              "上限（本当の好み）": (oracle_scores(w.true_rating()), False)}
        for name, (S, rm) in ms.items():
            acc.setdefault(name, []).append(_eval(d, S, rm))
    rows = []
    for name, v in acc.items():
        a = pd.DataFrame(v)
        se = a.std() / np.sqrt(len(a))
        rows.append({"手法": name, "観測NDCG@10": f"{a.obs.mean():.4f} ±{se.obs:.4f}", "本当のprecision@10": f"{a.true.mean():.3f} ±{se.true:.3f}",
                     "カバレッジ": f"{a['cov'].mean():.2f}", "RMSE(val)": "" if np.isnan(a.rmse.mean()) else f"{a.rmse.mean():.3f}"})
    return pd.DataFrame(rows)


def subspace_overlap(L: np.ndarray, V: np.ndarray) -> float:
    """学習した因子の張る空間 L（n×k）と、真の因子の空間 V（n×r）の重なり（0〜1。主角の cos² の平均）"""
    r = V.shape[1]
    Lo, _ = np.linalg.qr(L - L.mean(0))
    sv = np.linalg.svd(V.T @ Lo, compute_uv=False)[:r]
    return float(np.mean(sv ** 2))


def e3(events=(10, 25, 50, 100), seed=SEEDS[0]):
    rows = []
    for me in events:
        d = make_dataset(seed, mean_events=me)
        w, tr, val, U, I, ex = _ctx(d)
        T = w.true_score
        C = T - T.mean(1, keepdims=True) - T.mean(0, keepdims=True) + T.mean()
        Uu, S, Vt = np.linalg.svd(C, full_matrices=False)
        m = als_fit(tr, U, I, **ALS_DEFAULT)
        r = _eval(d, m.predict(), rmse_pred=True)
        base = _eval(d, item_mean_scores(tr, U, I))
        rows.append({"ユーザーあたりの評価数（目安）": me, "学習期間の評価の総数": len(tr), "ALSの本当のprecision": round(r["true"], 3),
                     "平均評価順の本当のprecision": round(base["true"], 3), "RMSE(val)": round(r["rmse"], 3),
                     "RMSE（全部の組・本当の好み）": round(rating_rmse_all(m.predict(), w.true_rating()), 3),
                     "商品因子の重なり(上位4次元)": round(subspace_overlap(m.Q, Vt.T[:, :4]), 2),
                     "ユーザー因子の重なり(上位4次元)": round(subspace_overlap(m.P, Uu[:, :4]), 2)})
    return pd.DataFrame(rows)


def e4(select_seed=SEEDS[0], eval_seeds=SEEDS[1:], ks=(1, 2, 4, 8, 16, 32), regs=(0.5, 1, 2, 5, 10, 20)):
    g = e1(select_seed, ks, regs)
    out = []
    for crit, col, best in (("検証のRMSEで選ぶ", "RMSE", "min"), ("観測NDCGで選ぶ", "観測NDCG", "max"), ("本当のprecisionで選ぶ", "本当のprecision", "max")):
        r = g.loc[g[col].idxmin() if best == "min" else g[col].idxmax()]
        res = []
        for sd in eval_seeds:
            d = make_dataset(sd)
            w, tr, val, U, I, ex = _ctx(d)
            res.append(_eval(d, als_fit(tr, U, I, k=int(r.k), reg=float(r.reg)).predict(), True))
        a = pd.DataFrame(res).mean()
        out.append({"選び方": crit, "選ばれた設定": f"k={int(r.k)}・reg={r.reg:g}", "別の世界での観測NDCG": round(a.obs, 4),
                    "別の世界での本当のprecision": round(a.true, 3), "別の世界でのRMSE": round(a.rmse, 3)})
    return pd.DataFrame(out)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20)
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4)]:
        df = fn()
        print(f"\n== {name} ==")
        print(df.to_string(index=False))
        df.to_csv(ROOT / "outputs" / f"phase3_{name}.csv", index=False)
