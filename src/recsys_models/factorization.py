"""行列分解（潜在因子モデル）。ユーザーと商品を、少数（k個）の「潜在因子」で表す。

  予測 r̂[u,i] = μ + b_u + b_i + p_u · q_i
    μ: 全体の平均  b_u: ユーザーの甘辛  b_i: 商品の質  p_u, q_i: k個の潜在因子（好みと、商品の特徴）
  p_u · q_i が大きいほど、ユーザー u は商品 i を好むと予測する。

学習の2通り:
  ALS（交互最小二乗）: **評価の点数**を、二乗誤差で当てにいく。ユーザーの因子を固定して商品を解き、次に逆、を繰り返す。
  BPR（ベイズ個人化ランキング）: **順位**を直接学ぶ。「好きな商品 i」が、「知らない商品 j」より上に来るように、p_u·(q_i − q_j) を大きくする。
"""
import numpy as np
import pandas as pd


class MF:
    """学習済みモデル。predict() で、全（ユーザー×商品）の予測を返す"""

    def __init__(self, mu, bu, bi, P, Q):
        self.mu, self.bu, self.bi, self.P, self.Q = mu, bu, bi, P, Q

    def predict(self) -> np.ndarray:
        return self.mu + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T


def _solve_side(groups, other_feat, target, n, k, reg, reg_bias):
    """片側（ユーザー or 商品）の、バイアスと因子を、リッジ回帰で解く。other_feat は、相手側の [1, 因子]"""
    out = np.zeros((n, k + 1))
    lam = np.diag([reg_bias] + [reg] * k)
    for idx, (others, y) in groups.items():
        X = other_feat[others]
        out[idx] = np.linalg.solve(X.T @ X + lam, X.T @ y)
    return out


def als_fit(train: pd.DataFrame, n_users: int, n_items: int, k: int = 8, reg: float = 5.0, reg_bias_user: float = 5.0,
            reg_bias_item: float = 10.0, iters: int = 15, seed: int = 0) -> MF:
    """ALS。評価の点数（rating − μ）を、二乗誤差で当てる。reg は因子の正則化（大きいほど、因子が小さく＝単純に）"""
    rng = np.random.default_rng(seed)
    u, i, r = train["user"].to_numpy(), train["item"].to_numpy(), train["rating"].to_numpy(dtype=float)
    mu = r.mean()
    P = rng.normal(0, 0.1, (n_users, k)); Q = rng.normal(0, 0.1, (n_items, k))
    bu, bi = np.zeros(n_users), np.zeros(n_items)
    by_user = {int(x): (i[u == x], None) for x in np.unique(u)}
    by_item = {int(x): (u[i == x], None) for x in np.unique(i)}
    for it in range(iters):
        # ユーザーを更新（商品側の [1, q_i] を固定）。目標は、評価 − μ − b_i
        feat_i = np.hstack([np.ones((n_items, 1)), Q])
        groups = {x: (items, r[u == x] - mu - bi[items]) for x, (items, _) in by_user.items()}
        sol = _solve_side(groups, feat_i, None, n_users, k, reg, reg_bias_user)
        bu, P = sol[:, 0], sol[:, 1:]
        # 商品を更新（ユーザー側の [1, p_u] を固定）。目標は、評価 − μ − b_u
        feat_u = np.hstack([np.ones((n_users, 1)), P])
        groups = {x: (users, r[i == x] - mu - bu[users]) for x, (users, _) in by_item.items()}
        sol = _solve_side(groups, feat_u, None, n_items, k, reg, reg_bias_item)
        bi, Q = sol[:, 0], sol[:, 1:]
    return MF(mu, bu, bi, P, Q)


def bpr_fit(train: pd.DataFrame, n_users: int, n_items: int, k: int = 16, lr: float = 0.05, reg: float = 0.01, epochs: int = 40,
            batch: int = 128, liked_only: bool = True, neg_pop_power: float = 0.0, seed: int = 0) -> MF:
    """BPR。好きな商品（liked_only=True なら評価4以上）が、まだ評価していない商品より上に来るように、SGDで学ぶ。
    neg_pop_power: 「知らない商品」の選び方。0=一様、1=人気に比例（人気の商品を、わざと外し役に選ぶ）"""
    rng = np.random.default_rng(seed)
    pos = train[train["rating"] >= 4] if liked_only else train
    pu, pi_ = pos["user"].to_numpy(), pos["item"].to_numpy()
    seen = np.zeros((n_users, n_items), dtype=bool)
    seen[train["user"].to_numpy(), train["item"].to_numpy()] = True
    cnt = np.bincount(train["item"].to_numpy(), minlength=n_items).astype(float)
    neg_p = (cnt + 1.0) ** neg_pop_power
    neg_p /= neg_p.sum()
    P = rng.normal(0, 0.1, (n_users, k)); Q = rng.normal(0, 0.1, (n_items, k)); bi = np.zeros(n_items)
    n = len(pu)
    for ep in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            uu, ii = pu[idx], pi_[idx]
            jj = rng.choice(n_items, size=len(idx), p=neg_p)
            for _ in range(3):                                   # 知っている商品を、外し役に選んでしまったら、選び直す
                bad = seen[uu, jj]
                if not bad.any():
                    break
                jj[bad] = rng.choice(n_items, size=int(bad.sum()), p=neg_p)
            x = (P[uu] * (Q[ii] - Q[jj])).sum(axis=1) + bi[ii] - bi[jj]
            g = 1.0 / (1.0 + np.exp(x))                          # σ(−x): 順位が逆（x が小さい）ほど、大きく更新
            Pu, Qi, Qj = P[uu].copy(), Q[ii].copy(), Q[jj].copy()
            np.add.at(P, uu, lr * (g[:, None] * (Qi - Qj) - reg * Pu))
            np.add.at(Q, ii, lr * (g[:, None] * Pu - reg * Qi))
            np.add.at(Q, jj, lr * (-g[:, None] * Pu - reg * Qj))
            np.add.at(bi, ii, lr * (g - reg * bi[ii]))
            np.add.at(bi, jj, lr * (-g - reg * bi[jj]))
    return MF(0.0, np.zeros(n_users), bi, P, Q)
