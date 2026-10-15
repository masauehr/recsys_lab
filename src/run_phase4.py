"""Phase4: コールドスタート（新しい商品・新しいユーザー）と、商品の特徴（ジャンル）の併用。

E1 新しい商品（評価0件）: 人気順・ALS・ジャンル好み・ジャンル事前分布つきALS
E2 新しい商品に、評価が少しずつ集まると: 評価数 m = 0〜50 で、ジャンル事前分布の価値はどう変わるか
E3 新しいユーザーに、評価が少しずつ集まると: 評価数 n = 0〜50 で、フォールドインのALSと基準の比較
E4 ジャンルの情報が不正確なとき（ジャンル札を一部ランダムに付け替える）、ジャンル事前分布は害になるか

評価: 本当の好みでの precision@10。候補は、新しい商品（E1・E2・E4）／全商品（E3。新ユーザーだけ集計）。
使い方: .venv/bin/python src/run_phase4.py
出力:   outputs/phase4_*.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys_eval import evaluate_true, exclusion_mask, item_mean_scores, oracle_scores, popularity_scores, topk_items  # noqa: E402
from recsys_models import (als_fit, fold_in_items, fold_in_users, genre_prior, genre_profile_scores)  # noqa: E402
from recsys_synth import make_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEEDS = [1_000_000 + i for i in range(10)]
ALS = dict(k=4, reg=5.0)
COLD_FRAC = 0.2                  # 新しい商品（ユーザー）の割合
MS = (0, 1, 2, 5, 10, 20, 50)    # 新しい商品／ユーザーに集まった評価の数


def _split_items(d, seed):
    """商品の20%を「新しい商品」にする。戻り値: (新しい商品, 既存の商品)"""
    rng = np.random.default_rng(seed + 77)
    cold = np.sort(rng.choice(d.world.n_items, int(d.world.n_items * COLD_FRAC), replace=False))
    return cold, np.setdiff1d(np.arange(d.world.n_items), cold)


def _true_prec(d, S, extra_exclude):
    """本当の好みでの precision@10。extra_exclude（候補から外す組）を除く。評価できるユーザーだけの平均"""
    tk = topk_items(S, extra_exclude, 10)
    return evaluate_true(tk, d.world.true_rating(), extra_exclude)["precision@10"]


def _cold_item_setup(d, seed, m):
    """既存商品だけで学習。新しい商品には、m 件だけ評価（学習期間のうち、ランダムに m 件）を与える。
    戻り値: (ALSモデル, 新しい商品, 既存商品, 新しい商品の評価(m件ずつ), 除外マスク)"""
    w = d.world
    tr = d.split("train")
    cold, warm = _split_items(d, seed)
    is_cold = np.isin(tr["item"].to_numpy(), cold)
    warm_tr, cold_tr = tr[~is_cold], tr[is_cold]
    model = als_fit(warm_tr, w.n_users, w.n_items, **ALS)
    rng = np.random.default_rng(seed + 88)
    shuffled = cold_tr.sample(frac=1.0, random_state=int(rng.integers(1 << 30)))
    kept = shuffled.groupby("item").head(m) if m > 0 else cold_tr.iloc[:0]
    ex = exclusion_mask(w.n_users, w.n_items, kept) if len(kept) else np.zeros((w.n_users, w.n_items), bool)
    ex[:, warm] = True                                       # 候補は新しい商品だけ
    return model, cold, warm, kept, ex


def e1(seeds=SEEDS):
    acc = {}
    for sd in seeds:
        d = make_dataset(sd)
        w = d.world
        model, cold, warm, kept, ex = _cold_item_setup(d, sd, 0)
        tr = d.split("train"); tr_warm = tr[~np.isin(tr["item"].to_numpy(), cold)]
        prior = genre_prior(model, w.item_genre, warm, w.n_genres)[w.item_genre[cold]]
        true_genre = np.stack([np.bincount(w.item_genre, weights=w.true_rating()[u], minlength=w.n_genres) /
                               np.maximum(np.bincount(w.item_genre, minlength=w.n_genres), 1) for u in range(w.n_users)])[:, w.item_genre]
        ms = {"ランダム": np.random.default_rng(sd).random((w.n_users, w.n_items)),
              "人気順（評価0件は全部同点）": popularity_scores(tr_warm, w.n_users, w.n_items),
              "平均評価順（同上）": item_mean_scores(tr_warm, w.n_users, w.n_items),
              "ALS（新しい商品は因子0）": fold_in_items(model, kept, cold, None),
              "ジャンル好み（コンテンツ）": genre_profile_scores(tr_warm, w.item_genre, w.n_users, w.n_genres),
              "ALS＋ジャンル事前分布（ハイブリッド）": fold_in_items(model, kept, cold, prior),
              "上限（ジャンルだけで分かる最良）": true_genre,
              "上限（本当の好み）": oracle_scores(w.true_rating())}
        for name, S in ms.items():
            acc.setdefault(name, []).append(_true_prec(d, S, ex))
    return pd.DataFrame([dict(手法=n, **{"本当のprecision@10": round(np.mean(v), 3), "標準誤差": round(np.std(v, ddof=1) / np.sqrt(len(v)), 3)})
                         for n, v in acc.items()])


def e2(seeds=SEEDS, ms=MS):
    rows = []
    for m in ms:
        acc = {}
        for sd in seeds:
            d = make_dataset(sd)
            w = d.world
            model, cold, warm, kept, ex = _cold_item_setup(d, sd, m)
            tr = d.split("train"); tr_warm = tr[~np.isin(tr["item"].to_numpy(), cold)]
            prior = genre_prior(model, w.item_genre, warm, w.n_genres)[w.item_genre[cold]]
            ms_ = {"ALS（因子0から）": fold_in_items(model, kept, cold, None),
                   "ALS＋ジャンル事前分布": fold_in_items(model, kept, cold, prior),
                   "ジャンル好みだけ": genre_profile_scores(tr_warm, w.item_genre, w.n_users, w.n_genres)}
            for name, S in ms_.items():
                acc.setdefault(name, []).append(_true_prec(d, S, ex))
        row = {"新しい商品の評価数": m}
        for name, v in acc.items():
            row[name] = round(float(np.mean(v)), 3)
        diff = np.array(acc["ALS＋ジャンル事前分布"]) - np.array(acc["ALS（因子0から）"])
        row["ハイブリッド−ALS（差）"] = round(float(diff.mean()), 3)
        row["差の標準誤差"] = round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def e3(seeds=SEEDS, ns=MS):
    """新しいユーザー（20%）。既存ユーザーは全評価で学習。新ユーザーには、学習期間の最初の n 件（時間順）だけを与える"""
    rows = []
    for n in ns:
        acc = {}
        for sd in seeds:
            d = make_dataset(sd)
            w = d.world
            rng = np.random.default_rng(sd + 99)
            cold = np.sort(rng.choice(w.n_users, int(w.n_users * COLD_FRAC), replace=False))
            tr = d.split("train")
            is_cold = np.isin(tr["user"].to_numpy(), cold)
            warm_tr = tr[~is_cold]
            kept = tr[is_cold].sort_values("time").groupby("user").head(n) if n > 0 else tr[is_cold].iloc[:0]
            model = als_fit(warm_tr, w.n_users, w.n_items, **ALS)
            ex = exclusion_mask(w.n_users, w.n_items, warm_tr, kept)
            ex[np.setdiff1d(np.arange(w.n_users), cold)] = True      # 新しいユーザーだけ評価する
            ms_ = {"人気順": popularity_scores(warm_tr, w.n_users, w.n_items),
                   "平均評価順": item_mean_scores(warm_tr, w.n_users, w.n_items),
                   "ALS（フォールドイン）": fold_in_users(model, kept, cold),
                   "ジャンル好み": genre_profile_scores(kept, w.item_genre, w.n_users, w.n_genres)}
            if n == 0:
                ms_["ジャンル好み"] = np.zeros((w.n_users, w.n_items))      # 評価0件では、ジャンルの手がかりが無い
            for name, S in ms_.items():
                acc.setdefault(name, []).append(_true_prec_cold(d, S, ex, cold))
        row = {"新しいユーザーの評価数": n}
        for name, v in acc.items():
            row[name] = round(float(np.mean(v)), 3)
        diff = np.array(acc["ALS（フォールドイン）"]) - np.array(acc["平均評価順"])
        row["ALS−平均評価順（差）"] = round(float(diff.mean()), 3)
        row["差の標準誤差"] = round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _true_prec_cold(d, S, ex, cold):
    tk = topk_items(S, ex, 10)
    rel = (d.world.true_rating() >= 4) & ~ex
    hits = np.take_along_axis(rel, tk, axis=1).sum(axis=1) / 10.0
    return float(hits[cold].mean())


def e4(seeds=SEEDS, noises=(0.0, 0.25, 0.5, 0.75, 1.0), m_list=(0, 5, 20)):
    """新しい商品のジャンル札を、noise の割合だけランダムに付け替える（札が信用できない状況）"""
    rows = []
    for noise in noises:
        for m in m_list:
            a, b, c = [], [], []
            for sd in seeds:
                d = make_dataset(sd)
                w = d.world
                model, cold, warm, kept, ex = _cold_item_setup(d, sd, m)
                tr = d.split("train"); tr_warm = tr[~np.isin(tr["item"].to_numpy(), cold)]
                rng = np.random.default_rng(sd + 123)
                g = w.item_genre.copy()
                flip = rng.random(len(cold)) < noise
                g[cold[flip]] = rng.integers(0, w.n_genres, flip.sum())       # 新しい商品のジャンル札だけを、付け替える
                prior = genre_prior(model, w.item_genre, warm, w.n_genres)[g[cold]]
                a.append(_true_prec(d, fold_in_items(model, kept, cold, None), ex))
                b.append(_true_prec(d, fold_in_items(model, kept, cold, prior), ex))
                c.append(_true_prec(d, genre_profile_scores(tr_warm, g, w.n_users, w.n_genres), ex))
            diff = np.array(b) - np.array(a)
            rows.append({"札の付け替え割合": noise, "新しい商品の評価数": m, "ALS（因子0から）": round(float(np.mean(a)), 3),
                         "ALS＋ジャンル事前分布": round(float(np.mean(b)), 3), "ジャンル好みだけ": round(float(np.mean(c)), 3),
                         "ハイブリッド−ALS": round(float(diff.mean()), 3), "差の標準誤差": round(float(diff.std(ddof=1) / np.sqrt(len(diff))), 3)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    (ROOT / "outputs").mkdir(exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20)
    for name, fn in [("E1", e1), ("E2", e2), ("E3", e3), ("E4", e4)]:
        df = fn()
        print(f"\n== {name} ==")
        print(df.to_string(index=False))
        df.to_csv(ROOT / "outputs" / f"phase4_{name}.csv", index=False)
