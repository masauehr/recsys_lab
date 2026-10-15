"""基準（ベースライン）。どれも、観測データ（学習用）だけを使う。oracle だけは、本当の好みを使う上限（手法ではない）。

返すのは、スコアの行列（ユーザー × 商品）。大きいほど、薦めたい。
"""
import numpy as np
import pandas as pd


def random_scores(n_users: int, n_items: int, seed: int = 0) -> np.ndarray:
    """でたらめに薦める（下限）"""
    return np.random.default_rng(seed).random((n_users, n_items))


def popularity_scores(train: pd.DataFrame, n_users: int, n_items: int, liked_only: bool = False) -> np.ndarray:
    """人気順: 学習データで、評価された回数が多い商品ほど薦める（全員に同じ）。liked_only=True なら、高評価(>=4)の回数"""
    d = train[train["rating"] >= 4] if liked_only else train
    cnt = d.groupby("item").size().reindex(range(n_items), fill_value=0).to_numpy(dtype=float)
    return np.tile(cnt, (n_users, 1))


def item_mean_scores(train: pd.DataFrame, n_users: int, n_items: int, prior_weight: float = 5.0) -> np.ndarray:
    """商品の平均評価順（ベイズ平均で、評価の少ない商品を、全体平均に寄せる）。全員に同じ"""
    g = train.groupby("item")["rating"].agg(["sum", "count"]).reindex(range(n_items), fill_value=0)
    mu = train["rating"].mean()
    m = (g["sum"].to_numpy() + prior_weight * mu) / (g["count"].to_numpy() + prior_weight)
    return np.tile(m, (n_users, 1))


def oracle_scores(true_rating: np.ndarray) -> np.ndarray:
    """本当の好みで薦める（上限。現実には作れない。評価の天井を知るため）"""
    return true_rating.copy()
