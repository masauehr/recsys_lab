"""Phase1: 評価の落とし穴。

E1 評価点の予測誤差（RMSE）と、順位の良さは、別物
E2 候補の選び方（全商品／サンプリング）で、数字も差も変わる
E3 偏りの強さで、「観測テストの順位」が「本当の良さの順位」とずれる度合い
E4 世界（シード）を変えた10回の平均とばらつき（基準の比較）

使い方: .venv/bin/python src/run_phase1.py
出力:   outputs/phase1_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys_eval import (bias_pred, evaluate_observed, evaluate_observed_sampled, evaluate_true, exclusion_mask,  # noqa: E402
                         global_mean_pred, item_mean_pred, item_mean_scores, oracle_scores, popularity_scores,
                         random_scores, rating_errors, rating_rmse_all, topk_items, user_mean_pred)
from recsys_synth import make_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]


def _ctx(d):
    w = d.world
    tr, val = d.split("train"), d.split("val")
    return w, tr, val, w.n_users, w.n_items, exclusion_mask(w.n_users, w.n_items, tr)


def method_scores(d):
    """比べる手法（スコア行列）。観測データだけを使う。oracle は上限（手法ではない）"""
    w, tr, val, U, I, ex = _ctx(d)
    return {"ランダム": random_scores(U, I), "人気順": popularity_scores(tr, U, I), "人気順（高評価の回数）": popularity_scores(tr, U, I, True),
            "平均評価順": item_mean_scores(tr, U, I), "ユーザー+商品のバイアス": bias_pred(tr, U, I),
            "全体平均": global_mean_pred(tr, U, I), "ユーザー平均": user_mean_pred(tr, U, I),
            "上限（本当の好み）": oracle_scores(w.true_rating())}


def e1(seed=SEEDS[0]):
    d = make_dataset(seed)
    w, tr, val, U, I, ex = _ctx(d)
    rows = []
    for name, P in [("全体平均", global_mean_pred(tr, U, I)), ("ユーザー平均（甘辛だけ）", user_mean_pred(tr, U, I)),
                    ("商品平均", item_mean_pred(tr, U, I)), ("ユーザー+商品のバイアス", bias_pred(tr, U, I))]:
        tk = topk_items(P, ex, 10)
        rows.append({"手法": name, "RMSE（観測テスト）": round(rating_errors(P, val)["RMSE"], 3), "RMSE（全部の組・本当の好み）": round(rating_rmse_all(P, w.true_rating()), 3),
                     "観測NDCG@10": round(evaluate_observed(P, tk, val, U, I)["ndcg@10"], 4),
                     "本当の好みprecision@10": round(evaluate_true(tk, w.true_rating(), ex)["precision@10"], 3)})
    S = popularity_scores(tr, U, I); tk = topk_items(S, ex, 10)
    rows.append({"手法": "人気順（点の予測ではない）", "RMSE（観測テスト）": np.nan, "RMSE（全部の組・本当の好み）": np.nan,
                 "観測NDCG@10": round(evaluate_observed(S, tk, val, U, I)["ndcg@10"], 4),
                 "本当の好みprecision@10": round(evaluate_true(tk, w.true_rating(), ex)["precision@10"], 3)})
    return pd.DataFrame(rows)


def e2(seed=SEEDS[0]):
    d = make_dataset(seed)
    w, tr, val, U, I, ex = _ctx(d)
    pop = tr.groupby("item").size().reindex(range(I), fill_value=0).to_numpy(dtype=float)
    rows = []
    for name, S in [("ランダム", random_scores(U, I)), ("人気順", popularity_scores(tr, U, I)), ("平均評価順", item_mean_scores(tr, U, I)),
                    ("上限（本当の好み）", oracle_scores(w.true_rating()))]:
        full = evaluate_observed(S, topk_items(S, ex, 10), val, U, I)
        s1 = evaluate_observed_sampled(S, val, ex, 100)
        s2 = evaluate_observed_sampled(S, val, ex, 100, neg_by_pop=pop)
        rows.append({"手法": name, "全商品 NDCG": round(full["ndcg@10"], 4), "全商品 hit": round(full["hit@10"], 3),
                     "サンプリング(一様) NDCG": round(s1["ndcg@10"], 4), "サンプリング(一様) hit": round(s1["hit@10"], 3),
                     "サンプリング(人気比例) NDCG": round(s2["ndcg@10"], 4), "サンプリング(人気比例) hit": round(s2["hit@10"], 3)})
    return pd.DataFrame(rows)


def e3(seeds=SEEDS[:5], powers=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5)):
    """偏りの強さ pop_power ごとに、手法の順位（観測NDCG）と、本当の良さの順位（本当のprecision）の、順位相関"""
    rows = []
    for pp in powers:
        corr, pop_vs_orc = [], []
        for sd in seeds:
            d = make_dataset(sd, pop_power=pp)
            w, tr, val, U, I, ex = _ctx(d)
            obs, tru = {}, {}
            for name, S in method_scores(d).items():
                tk = topk_items(S, ex, 10)
                obs[name] = evaluate_observed(S, tk, val, U, I)["ndcg@10"]
                tru[name] = evaluate_true(tk, w.true_rating(), ex)["precision@10"]
            names = [n for n in obs if n != "上限（本当の好み）"]
            corr.append(spearmanr([obs[n] for n in names], [tru[n] for n in names])[0])
            pop_vs_orc.append(obs["人気順"] / obs["上限（本当の好み）"])
        rows.append({"人気の効き pop_power": pp, "順位相関（観測 と 本当）": round(float(np.mean(corr)), 2),
                     "観測NDCG: 人気順÷上限": round(float(np.mean(pop_vs_orc)), 2)})
    return pd.DataFrame(rows)


def e4(seeds=SEEDS):
    rows = {}
    for sd in seeds:
        d = make_dataset(sd)
        w, tr, val, U, I, ex = _ctx(d)
        for name, S in method_scores(d).items():
            tk = topk_items(S, ex, 10)
            ob, tt = evaluate_observed(S, tk, val, U, I), evaluate_true(tk, w.true_rating(), ex)
            rows.setdefault(name, []).append((ob["ndcg@10"], ob["precision@10"], tt["precision@10"], ob["評価したユーザー数"]))
    out = []
    for name, v in rows.items():
        a = np.array(v)
        se = a.std(axis=0) / np.sqrt(len(a))
        out.append({"手法": name, "観測NDCG@10": f"{a[:, 0].mean():.4f} ±{se[0]:.4f}", "観測precision@10": f"{a[:, 1].mean():.4f} ±{se[1]:.4f}",
                    "本当のprecision@10": f"{a[:, 2].mean():.3f} ±{se[2]:.3f}", "評価したユーザー数(平均)": int(a[:, 3].mean())})
    return pd.DataFrame(out)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4)]:
        df = fn()
        print(f"\n== {name} ==")
        print(df.to_string(index=False))
        df.to_csv(ROOT / "outputs" / f"phase1_{name}.csv", index=False)
