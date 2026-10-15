"""偏りの補正。観測される確率（傾向）の逆数で重みを付ける（逆確率重み付け: IPS／IPW）。

観測データは「人気の商品ほど、よく評価される」。人気の商品の評価1件は、「たまたま見えた1件」の代表としては重複気味、
人気でない商品の評価1件は、「見えなかった多くの同類」の代表になる。そこで、観測される確率 p_i の逆数 1/p_i を重みにする。
ここでの傾向 p_i は、商品ごとの学習データでの評価件数から推定する（ユーザーの好みによる選ばれやすさ＝MNARは、推定できない）。
"""
import numpy as np
import pandas as pd

from recsys_synth import RELEVANT


def item_propensity(train: pd.DataFrame, n_users: int, n_items: int, floor: float = 0.01) -> np.ndarray:
    """商品ごとの、観測される確率の推定（評価件数 / ユーザー数）。小さすぎると重みが爆発するので floor で下から切る"""
    cnt = np.bincount(train["item"].to_numpy(), minlength=n_items).astype(float)
    return np.clip((cnt + 1.0) / (n_users + 1.0), floor, 1.0)


def ipw_weights(train: pd.DataFrame, prop: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """学習データの行ごとの重み (1/p_i)^gamma。平均が1になるようにそろえる。gamma=0 で重み無し"""
    w = (1.0 / prop[train["item"].to_numpy()]) ** gamma
    return w / w.mean()


def recall_observed(topk: np.ndarray, test: pd.DataFrame, n_users: int, n_items: int, weight: np.ndarray | None = None, k: int = 10) -> float:
    """観測テストでの recall@k。weight（商品ごとの重み）があれば、重み付き（自己正規化IPS）。
    テストで高く評価した商品が1つもないユーザーは除く"""
    rel = np.zeros((n_users, n_items), dtype=float)
    t = test[test["rating"] >= RELEVANT]
    rel[t["user"].to_numpy(), t["item"].to_numpy()] = 1.0
    if weight is not None:
        rel = rel * weight[None, :]
    hit = np.take_along_axis(rel, topk[:, :k], axis=1).sum(axis=1)
    tot = rel.sum(axis=1)
    ok = tot > 0
    return float((hit[ok] / tot[ok]).mean())
