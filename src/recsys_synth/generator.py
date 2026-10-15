"""合成データ生成器。

「本当の世界」（ユーザーの好み・商品の質と人気）を作り、そこから**偏った観測**（誰が何を評価したか）を作る。
手法に渡してよいのは、観測（user・item・rating・time）と、商品のジャンルだけ。本当の好みは**評価用の答え**で、手法に渡さない。

本当の好み（連続値）:  s[u,i] = 3.3 + b_u + q_i + (p_u · v_i の合致 − 平均) × scale
    b_u: ユーザーの甘辛（評価が高め／低め）  q_i: 商品の質  p_u, v_i: 好みと商品の特徴（潜在因子。ジャンルの影響を受ける）
観測（誰が何を評価するか）: ユーザー u は、商品 i を **人気 w_i が高いほど** かつ **好きなほど** 選びやすい。
    → 観測データは偏る（人気に偏る・好きなものしか評価されない）。評価しなかったものは「嫌い」ではなく「知らない」。
観測された評価 = round(s + ノイズ) を 1〜5 に収めたもの。時刻は 0〜1 の一様乱数。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

ORACLE_COLUMNS = ["true_rating"]           # 手法に渡してはいけない列（本当の好み）
SPLIT_TIMES = (0.70, 0.85)                  # 時間で分割: train < 0.70 ≤ val < 0.85 ≤ test
RELEVANT = 4                                # 評価がこれ以上なら「好き（関連あり）」とみなす


@dataclass
class World:
    """本当の世界（答え）。評価にだけ使う"""
    n_users: int
    n_items: int
    n_genres: int
    item_genre: np.ndarray        # 商品ごとのジャンル（これは、手法に渡してよい「商品の特徴」）
    item_pop: np.ndarray          # 商品の人気（選ばれやすさ）
    item_quality: np.ndarray
    user_bias: np.ndarray
    user_activity: np.ndarray     # ユーザーの活発さ（評価する件数の目安）
    true_score: np.ndarray        # (ユーザー × 商品) 本当の好み（連続値。1〜5に収める前）

    def true_rating(self) -> np.ndarray:
        return np.clip(self.true_score, 1.0, 5.0)


@dataclass
class Dataset:
    world: World
    interactions: pd.DataFrame    # user, item, rating, time, split（観測。手法に渡してよい）
    items: pd.DataFrame           # item, genre（商品の特徴。手法に渡してよい）

    def split(self, name: str) -> pd.DataFrame:
        return self.interactions[self.interactions["split"] == name]

    def train(self):
        return self.split("train")


def make_world(seed: int, n_users: int = 1000, n_items: int = 500, n_genres: int = 8, dim: int = 8,
               scale: float = 0.9, pop_sigma: float = 1.2) -> World:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1.0, (n_genres, dim))
    genre_share = rng.dirichlet(np.full(n_genres, 2.0))
    item_genre = rng.choice(n_genres, n_items, p=genre_share)
    item_vec = centers[item_genre] + 0.6 * rng.normal(0, 1.0, (n_items, dim))
    item_quality = rng.normal(0, 0.4, n_items)
    # 人気は、対数正規（少数の商品に集中）。質とは、わずかに相関
    item_pop = np.exp(pop_sigma * rng.normal(0, 1.0, n_items) + 0.5 * item_quality / 0.4 * 0.3)
    # ユーザーは、少数のジャンルを好む（ディリクレ分布の集中度を低くする）
    pref = rng.dirichlet(np.full(n_genres, 0.3), n_users)
    user_vec = pref @ centers + 0.5 * rng.normal(0, 1.0, (n_users, dim))
    user_bias = rng.normal(0, 0.4, n_users)
    user_activity = np.exp(rng.normal(0, 0.8, n_users))
    match = (user_vec @ item_vec.T) / np.sqrt(dim)
    match = match - match.mean()                              # 好みの合致の平均を0にそろえる（全体の平均評価を、基準の3.3に保つ）
    score = 3.3 + user_bias[:, None] + item_quality[None, :] + scale * match
    return World(n_users, n_items, n_genres, item_genre, item_pop, item_quality, user_bias, user_activity, score)


def simulate_interactions(world: World, seed: int, mean_events: float = 25.0, taste_effect: float = 0.5,
                          pop_power: float = 1.0, noise: float = 0.5) -> pd.DataFrame:
    """偏った観測を作る。商品の選ばれやすさ ∝ 人気^pop_power × exp(taste_effect × (好み − 3.5))"""
    rng = np.random.default_rng(seed)
    u_n, i_n = world.n_users, world.n_items
    counts = np.maximum(1, rng.poisson(mean_events * world.user_activity / world.user_activity.mean()))
    rows = []
    base = world.item_pop ** pop_power
    for u in range(u_n):
        w = base * np.exp(taste_effect * (world.true_rating()[u] - 3.5))
        p = w / w.sum()
        k = min(counts[u], i_n)
        items = rng.choice(i_n, size=k, replace=False, p=p)
        r = np.clip(np.round(world.true_score[u, items] + rng.normal(0, noise, k)), 1, 5).astype(int)
        t = rng.random(k)
        rows.append(pd.DataFrame({"user": u, "item": items, "rating": r, "time": t}))
    df = pd.concat(rows, ignore_index=True).sort_values("time").reset_index(drop=True)
    a, b = SPLIT_TIMES
    df["split"] = np.where(df["time"] < a, "train", np.where(df["time"] < b, "val", "test"))
    return df


def make_dataset(seed: int, **kw) -> Dataset:
    """世界（seed）と観測（seed+500000）を作る。同じseedなら同じデータ。
    seed の範囲: 開発用 1.0M〜、最終評価用 3.0M〜"""
    world_kw = {k: kw.pop(k) for k in ("n_users", "n_items", "n_genres", "dim", "scale", "pop_sigma") if k in kw}
    world = make_world(seed, **world_kw)
    inter = simulate_interactions(world, seed + 500_000, **kw)
    items = pd.DataFrame({"item": np.arange(world.n_items), "genre": world.item_genre})
    return Dataset(world, inter, items)
