"""近傍法（協調フィルタリング）。「似た人が好きなもの」「似た商品」から薦める。

  アイテム型: 商品 i と商品 j が、同じユーザーに評価されている度合い（類似度）から、
              ユーザーが好きな商品 j に似た商品 i を薦める。  スコア(u,i) = Σ_j 類似度(i,j) × (u が j を好き)
  ユーザー型: ユーザー u に似たユーザー v を探し、v が好きな商品を薦める。  スコア(u,i) = Σ_v 類似度(u,v) × (v が i を好き)

「好き」の入力は2通り:
  interactions（評価した＝1）: 何を評価したか。**露出（人気に偏った）を、そのまま学ぶ**
  liked（評価が4以上＝1）    : 好みの情報。評価した上で、高く評価したものだけ
類似度: コサイン類似度の拡張  sim(i,j) = 共起(i,j) / ( n_i^α × n_j^(1−α) ) × 共起 / (共起 + shrink)
  α=0.5 が通常のコサイン。α を動かすと、人気の商品の影響が変わる。shrink（収縮）は、共起が少ない類似度を、小さくする。
"""
import numpy as np
import pandas as pd


def interaction_matrix(df: pd.DataFrame, n_users: int, n_items: int, mode: str = "interactions") -> np.ndarray:
    """ユーザー×商品の 0/1 行列。mode: interactions（評価した）/ liked（評価が4以上）"""
    d = df[df["rating"] >= 4] if mode == "liked" else df
    m = np.zeros((n_users, n_items))
    m[d["user"].to_numpy(), d["item"].to_numpy()] = 1.0
    return m


def _similarity(B: np.ndarray, alpha: float, shrink: float) -> np.ndarray:
    """列（商品）どうしの類似度。B: (ユーザー × 商品)"""
    co = B.T @ B                                          # 共起（同じユーザーが両方を評価した数）
    n = np.diag(co).copy()
    n[n == 0] = 1.0
    sim = co / (n[:, None] ** alpha * n[None, :] ** (1 - alpha))
    if shrink > 0:
        sim = sim * co / (co + shrink)
    np.fill_diagonal(sim, 0.0)                            # 自分自身との類似度は使わない
    return sim


def item_similarity(B: np.ndarray, alpha: float = 0.5, shrink: float = 0.0, k: int | None = None) -> np.ndarray:
    """商品どうしの類似度。k を指定すると、各商品について、似ている上位k個だけを残す"""
    sim = _similarity(B, alpha, shrink)
    if k is not None and k < sim.shape[1]:
        thr = -np.partition(-sim, k - 1, axis=1)[:, k - 1: k]
        sim = np.where(sim >= thr, sim, 0.0)
    return sim


def user_similarity(B: np.ndarray, alpha: float = 0.5, shrink: float = 0.0, k: int | None = None) -> np.ndarray:
    """ユーザーどうしの類似度（行どうし）。k を指定すると、各ユーザーについて、似ている上位k人だけを残す"""
    sim = _similarity(B.T, alpha, shrink)
    if k is not None and k < sim.shape[1]:
        thr = -np.partition(-sim, k - 1, axis=1)[:, k - 1: k]
        sim = np.where(sim >= thr, sim, 0.0)
    return sim


def item_knn_scores(train: pd.DataFrame, n_users: int, n_items: int, mode: str = "interactions", alpha: float = 0.5,
                    shrink: float = 0.0, k: int | None = None) -> np.ndarray:
    """アイテム型。スコア(u,i) = Σ_j sim(i,j) × [u が j を好き]"""
    B = interaction_matrix(train, n_users, n_items, mode)
    like = B if mode == "liked" else interaction_matrix(train, n_users, n_items, "interactions")
    return like @ item_similarity(B, alpha, shrink, k)


def user_knn_scores(train: pd.DataFrame, n_users: int, n_items: int, mode: str = "interactions", alpha: float = 0.5,
                    shrink: float = 0.0, k: int | None = None) -> np.ndarray:
    """ユーザー型。スコア(u,i) = Σ_v sim(u,v) × [v が i を好き]"""
    B = interaction_matrix(train, n_users, n_items, mode)
    return user_similarity(B, alpha, shrink, k) @ B


def item_knn_rating_pred(train: pd.DataFrame, n_users: int, n_items: int, k: int | None = 30, shrink: float = 10.0,
                         prior_weight: float = 5.0) -> np.ndarray:
    """評価点の予測（アイテム型）: 商品の平均 ＋ 似た商品への、そのユーザーの評価の「平均からの差」の、類似度による加重平均。
    類似度は、評価した/しないの共起（コサイン）。順位にも使える（予測が大きい順）。"""
    B = interaction_matrix(train, n_users, n_items, "interactions")
    sim = item_similarity(B, 0.5, shrink, k)
    mu = float(train["rating"].mean())
    g = train.groupby("item")["rating"].agg(["sum", "count"]).reindex(range(n_items), fill_value=0)
    item_mean = (g["sum"].to_numpy() + prior_weight * mu) / (g["count"].to_numpy() + prior_weight)
    dev = np.zeros((n_users, n_items))
    dev[train["user"].to_numpy(), train["item"].to_numpy()] = train["rating"].to_numpy() - item_mean[train["item"].to_numpy()]
    num = dev @ sim
    den = B @ np.abs(sim)
    return item_mean[None, :] + np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
