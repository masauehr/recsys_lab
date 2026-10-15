"""Phase5: 偏りの補正とフィードバックループ。

E1 偏りを補正した評価（IPS）は、本当の順位に近づくか（手法の並びの一致＝スピアマン相関）
E2 偏りを補正した学習（IPW-ALS）は、効くか（γ=重みの強さ）
E3 推薦を繰り返すシミュレーション: 薦めたものが、よく評価される → 偏りは増幅するか（人気順／ALS／ALS＋探索）
E4 人気を抑える並べ替え（多様性）と、公平性（評価が少ないユーザー対多いユーザー）

使い方: .venv/bin/python src/run_phase5.py
出力:   outputs/phase5_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys_eval import (beyond_accuracy, evaluate_observed, evaluate_true, exclusion_mask, gini, ipw_weights,  # noqa: E402
                         item_mean_scores, item_propensity, popularity_scores, random_scores, rating_rmse_all,
                         recall_observed, topk_items)
from recsys_models import als_fit, bpr_fit, item_knn_scores  # noqa: E402
from recsys_synth import make_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]
ALS = dict(k=4, reg=5.0)
BPR = dict(k=16, lr=0.05, reg=0.01, epochs=40)


def _true_recall(tk, w, ex):
    return evaluate_true(tk, w.true_rating(), ex)["recall@10"]


def e1(seeds=SEEDS, pop_powers=(0.5, 1.0, 1.5)):
    rows, corr = [], []
    for pp in pop_powers:
        per_method, rhos = {}, []
        for sd in seeds:
            d = make_dataset(sd, pop_power=pp)
            w, tr, val = d.world, d.train(), d.split("val")
            U, I = w.n_users, w.n_items
            ex = exclusion_mask(U, I, tr)
            wt = 1.0 / item_propensity(tr, U, I)
            ms = {"ランダム": random_scores(U, I, sd), "人気順": popularity_scores(tr, U, I), "平均評価順": item_mean_scores(tr, U, I),
                  "近傍法": item_knn_scores(tr, U, I, "liked", 0.5, 5.0), "ALS": als_fit(tr, U, I, **ALS).predict(),
                  "BPR（一様）": bpr_fit(tr, U, I, **BPR, neg_pop_power=0.0, seed=sd).predict(),
                  "BPR（人気比例）": bpr_fit(tr, U, I, **BPR, neg_pop_power=1.0, seed=sd).predict()}
            o, ip, tru = [], [], []
            for name, S in ms.items():
                tk = topk_items(S, ex, 10)
                o.append(recall_observed(tk, val, U, I)); ip.append(recall_observed(tk, val, U, I, wt)); tru.append(_true_recall(tk, w, ex))
                per_method.setdefault(name, []).append((o[-1], ip[-1], tru[-1]))
            rhos.append((spearmanr(o, tru)[0], spearmanr(ip, tru)[0]))
        rh = np.array(rhos)
        corr.append({"偏り（pop_power）": pp, "観測と本当の順位相関": round(rh[:, 0].mean(), 2), "IPSと本当の順位相関": round(rh[:, 1].mean(), 2)})
        if pp == 1.0:
            for name, v in per_method.items():
                a = np.array(v).mean(axis=0)
                rows.append({"手法": name, "観測recall@10": round(a[0], 3), "IPS補正recall@10": round(a[1], 3), "本当のrecall@10": round(a[2], 3)})
    return pd.DataFrame(rows), pd.DataFrame(corr)


def e2(seeds=SEEDS, gammas=(0.0, 0.25, 0.5, 1.0)):
    rows = []
    for g in gammas:
        acc = []
        for sd in seeds:
            d = make_dataset(sd)
            w, tr = d.world, d.train()
            U, I = w.n_users, w.n_items
            ex = exclusion_mask(U, I, tr)
            wt = ipw_weights(tr, item_propensity(tr, U, I), g)
            S = als_fit(tr, U, I, **ALS, sample_weight=wt).predict()
            tk = topk_items(S, ex, 10)
            ba = beyond_accuracy(tk, I, np.bincount(tr["item"], minlength=I), w.item_genre)
            acc.append(dict(true=evaluate_true(tk, w.true_rating(), ex)["precision@10"], rmse=rating_rmse_all(S, w.true_rating()),
                            cover=ba["カバレッジ"], gini=ba["露出のジニ係数"]))
        a = pd.DataFrame(acc)
        rows.append({"重みの強さγ": g, "本当のprecision": round(a.true.mean(), 3), "標準誤差": round(a.true.std(ddof=1) / np.sqrt(len(a)), 3),
                     "全部の組のRMSE": round(a.rmse.mean(), 3), "カバレッジ": round(a.cover.mean(), 2), "露出のジニ": round(a.gini.mean(), 2)})
    return pd.DataFrame(rows)


def _consume(w, recs, rng, organic=3):
    """薦められた商品を、ユーザーが評価する。好みが高いほど評価しやすい。ほかに、自然な閲覧（人気×好み）も少しある"""
    U, I = w.n_users, w.n_items
    tru = w.true_rating()
    u = np.repeat(np.arange(U), recs.shape[1]); i = recs.ravel()
    take = rng.random(len(u)) < 1.0 / (1.0 + np.exp(-2.0 * (tru[u, i] - 3.5)))
    us, its = [u[take]], [i[take]]
    for x in range(U):                                    # 自然な閲覧
        p = w.item_pop * np.exp(0.5 * (tru[x] - 3.5)); p /= p.sum()
        it = rng.choice(I, organic, replace=False, p=p)
        us.append(np.full(organic, x)); its.append(it)
    u2, i2 = np.concatenate(us), np.concatenate(its)
    r = np.clip(np.round(w.true_score[u2, i2] + rng.normal(0, 0.5, len(u2))), 1, 5).astype(int)
    return pd.DataFrame({"user": u2, "item": i2, "rating": r})


def e3(seeds=SEEDS[:5], rounds=8, explore=0.2):
    rows = []
    for policy in ("人気順", "ALS", "ALS＋探索20%"):
        for sd in seeds:
            d = make_dataset(sd)
            w = d.world
            U, I = w.n_users, w.n_items
            data = d.train()[["user", "item", "rating"]].copy()
            rng = np.random.default_rng(sd + 31)
            ex0 = exclusion_mask(U, I, data)                  # 第0回の時点で知っていた商品。固定した候補で、モデルの良さを測る（消費で候補が減る影響を除く）
            rel0 = (w.true_rating() >= 4) & ~ex0
            for t in range(rounds + 1):
                ex = exclusion_mask(U, I, data)
                S = popularity_scores(data, U, I) if policy == "人気順" else als_fit(data, U, I, **ALS).predict()
                tk = topk_items(S, ex, 10)
                fixed = np.take_along_axis(rel0, topk_items(S, ex0, 10), axis=1).mean()
                if policy == "ALS＋探索20%":                       # リストの2件を、知らない商品からランダムに入れ替える
                    for x in range(U):
                        cand = np.flatnonzero(~ex[x])
                        tk[x, -2:] = rng.choice(cand, 2, replace=False)
                cnt = np.bincount(data["item"], minlength=I)
                top20 = cnt >= np.quantile(cnt, 0.8)
                rows.append(dict(方針=policy, seed=sd, 回=t,
                                 薦めたものの本当のprecision=evaluate_true(tk, w.true_rating(), ex)["precision@10"],
                                 固定候補での本当のprecision=float(fixed),
                                 薦めた商品のカバレッジ=float((np.bincount(tk.ravel(), minlength=I) > 0).mean()),
                                 薦めた回数のジニ=gini(np.bincount(tk.ravel(), minlength=I)),
                                 データの評価件数のジニ=gini(cnt),
                                 人気上位20pct商品の評価の割合=float(top20[data["item"].to_numpy()].mean())))
                if t < rounds:
                    data = pd.concat([data, _consume(w, tk, rng)], ignore_index=True)
    df = pd.DataFrame(rows)
    return df.groupby(["方針", "回"]).mean(numeric_only=True).drop(columns="seed").round(3).reset_index()


def e4(seeds=SEEDS, lams=(0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.4)):
    rows = []
    for lam in lams:
        acc = []
        for sd in seeds:
            d = make_dataset(sd)
            w, tr = d.world, d.train()
            U, I = w.n_users, w.n_items
            ex = exclusion_mask(U, I, tr)
            cnt = np.bincount(tr["item"], minlength=I)
            S = als_fit(tr, U, I, **ALS).predict() - lam * np.log1p(cnt)[None, :]            # 人気の高い商品を、対数の件数に応じて下げる
            tk = topk_items(S, ex, 10)
            rel = (w.true_rating() >= 4) & ~ex
            hits = np.take_along_axis(rel, tk, axis=1).sum(axis=1) / 10.0
            nu = np.bincount(tr["user"], minlength=U)
            light, heavy = nu <= np.quantile(nu, 0.3), nu >= np.quantile(nu, 0.7)
            ba = beyond_accuracy(tk, I, cnt, w.item_genre)
            acc.append(dict(prec=hits.mean(), light=hits[light].mean(), heavy=hits[heavy].mean(), cover=ba["カバレッジ"], gini=ba["露出のジニ係数"],
                            novel=ba["新奇性"], diverse=ba["ジャンル数/リスト"]))
        a = pd.DataFrame(acc).mean()
        rows.append({"人気を下げる強さλ": lam, "本当のprecision": round(a.prec, 3), "評価が少ないユーザー": round(a.light, 3), "評価が多いユーザー": round(a.heavy, 3),
                     "カバレッジ": round(a.cover, 2), "露出のジニ": round(a.gini, 2), "新奇性": round(a.novel, 2), "ジャンル数/リスト": round(a.diverse, 2)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20)
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4)]:
        out = fn()
        for j, df in enumerate(out if isinstance(out, tuple) else (out,)):
            print(f"\n== {name}-{j} ==")
            print(df.to_string(index=False))
            df.to_csv(ROOT / "outputs" / f"phase5_{name}_{j}.csv", index=False)
