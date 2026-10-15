"""推薦の評価指標。

ほしい答えは**順位**（上位k件に、好きなものが入るか）。評価点の予測誤差（RMSE）とは別物。
  precision@k : 上位k件のうち、好きなものの割合
  recall@k    : 好きなもの全体のうち、上位k件に入った割合
  NDCG@k      : 好きなものが、上位にあるほど高い（0〜1）。順位の良さをまとめた指標
  hit@k       : 上位k件に、好きなものが1つでもあった割合
「好き（関連あり）」の定義:
  観測テスト（evaluate_observed）: テスト期間に、そのユーザーが実際に高く評価した（rating >= 4）商品
  本当の好み（evaluate_true）   : 本当の好み（合成データだけが知っている）が 4 以上の商品。観測の偏りの影響を受けない
推薦の候補から、すでに知っている商品（学習データ等）は除く（exclude）。
"""
import numpy as np
import pandas as pd

from recsys_synth import RELEVANT


def exclusion_mask(n_users: int, n_items: int, *frames: pd.DataFrame) -> np.ndarray:
    """すでに知っている（学習等で評価した）商品の印。推薦の候補から外す"""
    m = np.zeros((n_users, n_items), dtype=bool)
    for df in frames:
        m[df["user"].to_numpy(), df["item"].to_numpy()] = True
    return m


def topk_items(scores: np.ndarray, exclude: np.ndarray, k: int = 10, seed: int = 0) -> np.ndarray:
    """各ユーザーの、上位k件の商品の番号（スコアの大きい順）。除外した商品は選ばない。同点はランダムに崩す"""
    rng = np.random.default_rng(seed)
    s = np.where(exclude, -np.inf, scores + rng.random(scores.shape) * 1e-9)
    part = np.argpartition(-s, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def _ndcg(hit_matrix: np.ndarray, n_relevant: np.ndarray, k: int) -> np.ndarray:
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hit_matrix * disc).sum(axis=1)
    ideal = np.array([disc[: min(int(n), k)].sum() for n in n_relevant])
    return np.where(ideal > 0, dcg / np.maximum(ideal, 1e-12), 0.0)


def _metrics(topk: np.ndarray, rel: np.ndarray, k: int) -> dict:
    """rel: (ユーザー × 商品) の、関連ありの印。関連ありが1つもないユーザーは、評価から除く"""
    n_rel = rel.sum(axis=1)
    ok = n_rel > 0
    hits = np.take_along_axis(rel, topk, axis=1).astype(float)
    prec = hits.sum(axis=1) / k
    rec = np.where(ok, hits.sum(axis=1) / np.maximum(n_rel, 1), 0.0)
    ndcg = _ndcg(hits, n_rel, k)
    mean = lambda x: float(x[ok].mean()) if ok.any() else float("nan")      # noqa: E731  評価できるユーザーがいなければ nan
    return {f"precision@{k}": mean(prec), f"recall@{k}": mean(rec), f"ndcg@{k}": mean(ndcg),
            f"hit@{k}": mean((hits.sum(axis=1) > 0).astype(float)), "評価したユーザー数": int(ok.sum())}


def evaluate_observed(scores, topk, test: pd.DataFrame, n_users: int, n_items: int, k: int = 10) -> dict:
    """観測テストでの順位指標。関連あり＝テストで、そのユーザーが rating>=4 と評価した商品"""
    rel = np.zeros((n_users, n_items), dtype=bool)
    t = test[test["rating"] >= RELEVANT]
    rel[t["user"].to_numpy(), t["item"].to_numpy()] = True
    return _metrics(topk, rel, k)


def evaluate_true(topk, true_rating: np.ndarray, exclude: np.ndarray, k: int = 10) -> dict:
    """本当の好みでの順位指標。関連あり＝本当の好みが 4 以上で、まだ知らない商品（観測の偏りの影響を受けない）"""
    rel = (true_rating >= RELEVANT) & ~exclude
    return _metrics(topk, rel, k)


def rating_errors(pred: np.ndarray, test: pd.DataFrame) -> dict:
    """評価点の予測誤差（RMSE・MAE）。テストで実際に評価された組だけで測る"""
    p = pred[test["user"].to_numpy(), test["item"].to_numpy()]
    e = p - test["rating"].to_numpy()
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e)))}


def rating_rmse_all(pred: np.ndarray, true_rating: np.ndarray) -> float:
    """全部の組（評価されていない組も含む）での、本当の好みに対するRMSE。合成データだけが測れる。
    観測テストでのRMSEは、評価された（＝好きなものに偏った）組だけで測るので、これより甘くなる"""
    return float(np.sqrt(np.mean((pred - true_rating) ** 2)))


def evaluate_observed_sampled(scores: np.ndarray, test: pd.DataFrame, exclude: np.ndarray, n_neg: int = 100, k: int = 10,
                              seed: int = 0, neg_by_pop: np.ndarray | None = None) -> dict:
    """サンプリング評価: 各ユーザーについて、候補を「テストで高く評価した商品 ＋ ランダムに選んだ n_neg 個の商品」だけに絞って、順位を測る。
    全商品を候補にする評価（evaluate_observed）より、高い数字が出る。neg_by_pop を渡すと、人気に比例して選ぶ。"""
    rng = np.random.default_rng(seed)
    n_users, n_items = scores.shape
    t = test[test["rating"] >= RELEVANT]
    rel_by_user = t.groupby("user")["item"].apply(set).to_dict()
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    prec, ndcg, hit = [], [], []
    for u, rel in rel_by_user.items():
        rel = np.array(sorted(rel))
        pool = np.flatnonzero(~exclude[u] & ~np.isin(np.arange(n_items), rel))
        p = None if neg_by_pop is None else (neg_by_pop[pool] + 1.0) / (neg_by_pop[pool] + 1.0).sum()
        neg = rng.choice(pool, size=min(n_neg, len(pool)), replace=False, p=p)
        cand = np.concatenate([rel, neg])
        order = cand[np.argsort(-(scores[u, cand] + rng.random(len(cand)) * 1e-9))][:k]
        h = np.isin(order, rel).astype(float)
        prec.append(h.sum() / k)
        ideal = disc[: min(len(rel), k)].sum()
        ndcg.append((h * disc[: len(h)]).sum() / ideal)
        hit.append(float(h.sum() > 0))
    return {f"precision@{k}": float(np.mean(prec)), f"ndcg@{k}": float(np.mean(ndcg)), f"hit@{k}": float(np.mean(hit)),
            "評価したユーザー数": len(rel_by_user)}


def gini(x) -> float:
    """偏りの指標（0=完全に均等、1=1つに集中）"""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.sum(np.arange(1, n + 1) * x)) / (n * x.sum()) - (n + 1) / n)


def beyond_accuracy(topk: np.ndarray, n_items: int, item_pop_train: np.ndarray, item_genre: np.ndarray) -> dict:
    """順位の良さ以外の性質:
    カバレッジ（薦められた商品の種類の割合）、露出のジニ係数（薦められる回数の偏り）、
    新奇性（薦めた商品の、人気の低さ。大きいほど珍しいものを薦めている）、ジャンルの多様性（1つのリスト内のジャンル数）"""
    counts = np.bincount(topk.ravel(), minlength=n_items)
    share = (item_pop_train + 1) / (item_pop_train + 1).sum()
    novelty = float(np.mean(-np.log2(share[topk])))
    div = float(np.mean([len(set(item_genre[row])) for row in topk]))
    return {"カバレッジ": float((counts > 0).mean()), "露出のジニ係数": gini(counts), "新奇性": novelty, "ジャンル数/リスト": div}
