"""評価点を予測する、簡単なモデル（Phase1）。どれも、観測データ（学習用）だけを使う。

返すのは、予測した評価点の行列（ユーザー × 商品）。順位を作るときは、この予測が大きい順に薦める。
"""
import numpy as np
import pandas as pd


def global_mean_pred(train: pd.DataFrame, n_users: int, n_items: int) -> np.ndarray:
    """全部の評価の平均を、全員・全商品に予測する"""
    return np.full((n_users, n_items), float(train["rating"].mean()))


def item_mean_pred(train: pd.DataFrame, n_users: int, n_items: int, prior_weight: float = 5.0) -> np.ndarray:
    """商品ごとの平均（評価の少ない商品は、全体平均に寄せる）"""
    mu = float(train["rating"].mean())
    g = train.groupby("item")["rating"].agg(["sum", "count"]).reindex(range(n_items), fill_value=0)
    m = (g["sum"].to_numpy() + prior_weight * mu) / (g["count"].to_numpy() + prior_weight)
    return np.tile(m, (n_users, 1))


def user_mean_pred(train: pd.DataFrame, n_users: int, n_items: int, prior_weight: float = 5.0) -> np.ndarray:
    """ユーザーごとの平均（甘辛）。全商品に同じ値を予測するので、順位は全員の中で同じ（商品の差がない）"""
    mu = float(train["rating"].mean())
    g = train.groupby("user")["rating"].agg(["sum", "count"]).reindex(range(n_users), fill_value=0)
    m = (g["sum"].to_numpy() + prior_weight * mu) / (g["count"].to_numpy() + prior_weight)
    return np.tile(m[:, None], (1, n_items))


def bias_pred(train: pd.DataFrame, n_users: int, n_items: int, reg_user: float = 5.0, reg_item: float = 10.0,
              iters: int = 10) -> np.ndarray:
    """ユーザーの甘辛と商品の質だけのモデル: 予測 = 全体平均 + b_user + b_item（正則化つきの最小二乗）。
    評価点の予測誤差（RMSE）では、強い基準。ただし、好みの違い（誰に何が合うか）は、表現できない。"""
    u, i, r = train["user"].to_numpy(), train["item"].to_numpy(), train["rating"].to_numpy(dtype=float)
    mu = r.mean()
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    for _ in range(iters):
        bi = np.bincount(i, weights=r - mu - bu[u], minlength=n_items) / (reg_item + np.bincount(i, minlength=n_items))
        bu = np.bincount(u, weights=r - mu - bi[i], minlength=n_users) / (reg_user + np.bincount(u, minlength=n_users))
    return mu + bu[:, None] + bi[None, :]
