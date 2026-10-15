"""コールドスタート（評価が無い・少ない）への対応。商品の特徴（ジャンル）を使う。

  ジャンル好み（コンテンツベース）: ユーザーが各ジャンルに付けた評価の平均で、新しい商品を並べる
  ジャンル事前分布: 新しい商品の（バイアス・因子）の初期値を、同じジャンルの既存商品の平均にする
  フォールドイン: 学習済みの因子を固定して、新しいユーザー（商品）1人分だけを、少ない評価から解く
"""
import numpy as np
import pandas as pd

from .factorization import MF


def genre_profile_scores(train: pd.DataFrame, item_genre: np.ndarray, n_users: int, n_genres: int, weight: float = 3.0) -> np.ndarray:
    """ユーザー × 商品。ユーザーの「そのジャンルの平均評価」を、商品のスコアにする。
    評価が少ないジャンルは、ユーザー全体の平均に寄せる（weight 件ぶんの重み）"""
    g = item_genre[train["item"].to_numpy()]
    u, r = train["user"].to_numpy(), train["rating"].to_numpy(dtype=float)
    s = np.zeros((n_users, n_genres)); c = np.zeros((n_users, n_genres))
    np.add.at(s, (u, g), r); np.add.at(c, (u, g), 1.0)
    cnt = c.sum(axis=1); mean_u = np.where(cnt > 0, s.sum(axis=1) / np.maximum(cnt, 1), r.mean() if len(r) else 3.3)
    prof = (s + weight * mean_u[:, None]) / (c + weight)
    return prof[:, item_genre]


def genre_prior(model: MF, item_genre: np.ndarray, warm_items: np.ndarray, n_genres: int) -> np.ndarray:
    """ジャンルごとの [バイアス, 因子...] の平均（既存商品だけで計算）。(ジャンル × (1+k))"""
    feat = np.hstack([model.bi[:, None], model.Q])
    prior = np.zeros((n_genres, feat.shape[1]))
    overall = feat[warm_items].mean(axis=0)
    for g in range(n_genres):
        idx = warm_items[item_genre[warm_items] == g]
        prior[g] = feat[idx].mean(axis=0) if len(idx) else overall
    return prior


def fold_in_items(model: MF, ratings: pd.DataFrame, cold_items: np.ndarray, prior: np.ndarray | None,
                  reg: float = 5.0, reg_bias: float = 10.0) -> np.ndarray:
    """新しい商品の [バイアス, 因子] を、少ない評価から解く（ユーザー側は固定）。
    prior が None なら 0 に向けて、あれば prior（ジャンル平均）に向けて正則化する。
    戻り値: (ユーザー × 商品全体) の予測。cold_items 以外の列は、学習済みモデルのまま"""
    k = model.Q.shape[1]
    lam = np.diag([reg_bias] + [reg] * k)
    pred = model.predict()
    by_item = {int(i): d for i, d in ratings.groupby("item")}
    for n, i in enumerate(cold_items):
        p0 = np.zeros(k + 1) if prior is None else prior[n]
        sol = p0.copy()
        d = by_item.get(int(i))
        if d is not None and len(d):
            X = np.hstack([np.ones((len(d), 1)), model.P[d["user"].to_numpy()]])
            y = d["rating"].to_numpy(dtype=float) - model.mu - model.bu[d["user"].to_numpy()] - X @ p0
            sol = p0 + np.linalg.solve(X.T @ X + lam, X.T @ y)
        pred[:, i] = model.mu + model.bu + sol[0] + model.P @ sol[1:]
    return pred


def fold_in_users(model: MF, ratings: pd.DataFrame, cold_users: np.ndarray, reg: float = 5.0, reg_bias: float = 5.0) -> np.ndarray:
    """新しいユーザーの [バイアス, 因子] を、少ない評価から解く（商品側は固定）。(ユーザー全体 × 商品) の予測"""
    k = model.Q.shape[1]
    lam = np.diag([reg_bias] + [reg] * k)
    P, bu = model.P.copy(), model.bu.copy()
    P[cold_users] = 0.0; bu[cold_users] = 0.0
    cold_set = set(cold_users.tolist())
    for u, d in ratings.groupby("user"):
        if int(u) not in cold_set:
            continue
        it = d["item"].to_numpy()
        X = np.hstack([np.ones((len(d), 1)), model.Q[it]])
        y = d["rating"].to_numpy(dtype=float) - model.mu - model.bi[it]
        sol = np.linalg.solve(X.T @ X + lam, X.T @ y)
        bu[u], P[u] = sol[0], sol[1:]
    return model.mu + bu[:, None] + model.bi[None, :] + P @ model.Q.T
