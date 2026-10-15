"""Phase6: 最終評価。設定は開発用の世界（1.0M番台）で選んで固定した（Phase1〜5）。test の世界（3.0M番台）は1回だけ使う。

評価する問い（結果を見る前に決めた予想）:
  H1 本当のprecision: ALS ＞ 平均評価順 ＞ 近傍法 ＞ 人気順 ＞ ランダム
  H2 観測NDCG: 人気順は ALS・平均評価順より高い（観測と本当で順位が逆転する）
  H3 新しい商品（評価0件・5件）: ジャンル事前分布つきALS（ハイブリッド）＞ ALS（因子0から）
  H4 新しいユーザー（評価5件）: ALS（フォールドイン）＞ 平均評価順
  H5 人気を抑える並べ替え（λ=0.05）は、本当のprecisionをほぼ下げず、カバレッジを上げる

学習は train＋val（時刻 0.85 まで）、観測テストは test（時刻 0.85 以降）。候補から、学習で知っている商品を除く。
シード（世界）: 予行演習 val=2.2M番台、test=3.0M番台。開発用（1.0M・2.0M番台）とは別。結果を見て設定を変えない。

使い方:
  .venv/bin/python src/final_eval.py --split val    # 予行演習（何度でも可）
  .venv/bin/python src/final_eval.py --split test   # 本番（1回だけ。使用済みだと拒否する）
出力: outputs/final/final_{split}_*.csv（test は TEST_USED.json も作る）
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys_eval import (beyond_accuracy, evaluate_observed, evaluate_true, exclusion_mask, ipw_weights, item_mean_scores,  # noqa: E402
                         item_propensity, oracle_scores, popularity_scores, random_scores, topk_items)
from recsys_models import (als_fit, bpr_fit, fold_in_items, fold_in_users, genre_prior, genre_profile_scores,  # noqa: E402
                           item_knn_scores)
from recsys_synth import make_dataset  # noqa: E402
from run_phase4 import _cold_item_setup, _true_prec  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "final"

# Phase1〜5 で、開発用の世界（1.0M番台）を使って選んだ設定。最終評価の前に固定した
FINAL_CONFIG = dict(als=dict(k=4, reg=5.0), bpr=dict(k=16, lr=0.05, reg=0.01, epochs=40, neg_pop_power=1.0),
                    knn=dict(mode="liked", alpha=0.5, shrink=5.0), penalty_lambda=0.05, new_user_n=5, new_item_m=(0, 5))
SPLITS = {"val": [2_200_000 + i for i in range(10)], "test": [3_000_000 + i for i in range(10)]}


def _stat(v):
    v = np.asarray(v, dtype=float)
    return round(float(v.mean()), 3), round(float(v.std(ddof=1) / np.sqrt(len(v))), 3)


def part1(seeds, cfg=FINAL_CONFIG):
    """全体ランキング（train＋valで学習、testで観測評価）"""
    acc = {}
    for sd in seeds:
        d = make_dataset(sd)
        w = d.world
        U, I = w.n_users, w.n_items
        fit = pd.concat([d.split("train"), d.split("val")])
        test = d.split("test")
        ex = exclusion_mask(U, I, fit)
        cnt = np.bincount(fit["item"], minlength=I)
        als = als_fit(fit, U, I, **cfg["als"])
        wt = ipw_weights(fit, item_propensity(fit, U, I), 0.25)
        ms = {"ランダム": random_scores(U, I, sd), "人気順": popularity_scores(fit, U, I), "平均評価順": item_mean_scores(fit, U, I),
              "近傍法": item_knn_scores(fit, U, I, cfg["knn"]["mode"], cfg["knn"]["alpha"], cfg["knn"]["shrink"]),
              "BPR（人気比例）": bpr_fit(fit, U, I, **{k: v for k, v in cfg["bpr"].items()}, seed=sd).predict(),
              "ALS": als.predict(),
              "ALS＋IPW(γ=0.25)": als_fit(fit, U, I, **cfg["als"], sample_weight=wt).predict(),
              "ALS＋人気抑制(λ=0.05)": als.predict() - cfg["penalty_lambda"] * np.log1p(cnt)[None, :],
              "上限（本当の好み）": oracle_scores(w.true_rating())}
        for name, S in ms.items():
            tk = topk_items(S, ex, 10)
            ba = beyond_accuracy(tk, I, cnt, w.item_genre)
            acc.setdefault(name, []).append(dict(
                obs=evaluate_observed(S, tk, test, U, I)["ndcg@10"], true=evaluate_true(tk, w.true_rating(), ex)["precision@10"],
                cov=ba["カバレッジ"], gini=ba["露出のジニ係数"], nov=ba["新奇性"]))
    rows = []
    base = {k: np.array([r[k] for r in acc["平均評価順"]]) for k in ("true",)}
    for name, v in acc.items():
        a = pd.DataFrame(v)
        diff = a["true"].to_numpy() - base["true"]
        m, se = _stat(a["true"]); dm, dse = _stat(diff)
        rows.append({"手法": name, "観測NDCG": _stat(a.obs)[0], "本当のprecision": m, "標準誤差": se,
                     "平均評価順との差": dm, "差の標準誤差": dse, "カバレッジ": round(a["cov"].mean(), 2),
                     "露出のジニ": round(a.gini.mean(), 2), "新奇性": round(a.nov.mean(), 2)})
    return pd.DataFrame(rows)


def part2(seeds, cfg=FINAL_CONFIG):
    """新しい商品（評価m件）。候補は新しい商品だけ"""
    rows = []
    for m in cfg["new_item_m"]:
        acc = {}
        for sd in seeds:
            d = make_dataset(sd)
            w = d.world
            model, cold, warm, kept, ex = _cold_item_setup(d, sd, m)
            tr = d.split("train"); tr_warm = tr[~np.isin(tr["item"].to_numpy(), cold)]
            prior = genre_prior(model, w.item_genre, warm, w.n_genres)[w.item_genre[cold]]
            ms = {"ランダム": np.random.default_rng(sd).random((w.n_users, w.n_items)),
                  "ALS（因子0から）": fold_in_items(model, kept, cold, None),
                  "ジャンル好みだけ": genre_profile_scores(tr_warm, w.item_genre, w.n_users, w.n_genres),
                  "ハイブリッド（ALS＋ジャンル事前分布）": fold_in_items(model, kept, cold, prior)}
            for name, S in ms.items():
                acc.setdefault(name, []).append(_true_prec(d, S, ex))
        b = np.array(acc["ALS（因子0から）"])
        for name, v in acc.items():
            m_, se = _stat(v); dm, dse = _stat(np.array(v) - b)
            rows.append({"新しい商品の評価数": m, "手法": name, "本当のprecision": m_, "標準誤差": se, "ALSとの差": dm, "差の標準誤差": dse})
    return pd.DataFrame(rows)


def part3(seeds, cfg=FINAL_CONFIG):
    """新しいユーザー（学習期間の最初のn件）。新ユーザーだけを集計"""
    n = cfg["new_user_n"]
    acc = {}
    for sd in seeds:
        d = make_dataset(sd)
        w = d.world
        rng = np.random.default_rng(sd + 99)
        cold = np.sort(rng.choice(w.n_users, int(w.n_users * 0.2), replace=False))
        tr = d.split("train")
        is_cold = np.isin(tr["user"].to_numpy(), cold)
        warm_tr = tr[~is_cold]
        kept = tr[is_cold].sort_values("time").groupby("user").head(n)
        model = als_fit(warm_tr, w.n_users, w.n_items, **cfg["als"])
        ex = exclusion_mask(w.n_users, w.n_items, warm_tr, kept)
        ex[np.setdiff1d(np.arange(w.n_users), cold)] = True
        rel = (w.true_rating() >= 4) & ~ex
        for name, S in {"人気順": popularity_scores(warm_tr, w.n_users, w.n_items), "平均評価順": item_mean_scores(warm_tr, w.n_users, w.n_items),
                        "ALS（フォールドイン）": fold_in_users(model, kept, cold)}.items():
            hits = np.take_along_axis(rel, topk_items(S, ex, 10), axis=1).sum(axis=1) / 10.0
            acc.setdefault(name, []).append(float(hits[cold].mean()))
    b = np.array(acc["平均評価順"])
    rows = []
    for name, v in acc.items():
        m_, se = _stat(v); dm, dse = _stat(np.array(v) - b)
        rows.append({"新しいユーザーの評価数": n, "手法": name, "本当のprecision": m_, "標準誤差": se, "平均評価順との差": dm, "差の標準誤差": dse})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()
    lock = OUT / "TEST_USED.json"
    if a.split == "test" and lock.exists():
        sys.exit(f"testは使用済みです（{lock}）。再評価が必要なら、新しいシード（4M番台など）を使ってください。")
    seeds = SPLITS[a.split]
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 20)
    print(f"== {a.split}（世界のシード {seeds[0]}〜、{len(seeds)}個）==")
    for name, fn in (("ranking", part1), ("new_items", part2), ("new_users", part3)):
        df = fn(seeds)
        df.to_csv(OUT / f"final_{a.split}_{name}.csv", index=False)
        print(f"\n-- {name} --")
        print(df.to_string(index=False))
    if a.split == "test":
        lock.write_text(json.dumps(dict(used_at=datetime.now().isoformat(timespec="seconds"), config=FINAL_CONFIG,
                                        seeds=seeds[:1], note="testは最終評価で使用済み。設定は開発用の世界で選んで固定した"), ensure_ascii=False, indent=2, default=list))
        print("\nTEST_USED.json を作成しました（以後 --split test は拒否されます）")


if __name__ == "__main__":
    main()
