import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from recsys_eval import (beyond_accuracy, evaluate_observed, evaluate_true, exclusion_mask, gini, item_mean_scores,  # noqa: E402
                         oracle_scores, popularity_scores, random_scores, rating_errors, topk_items)
from recsys_synth import ORACLE_COLUMNS, SPLIT_TIMES, make_dataset  # noqa: E402


class TestGenerator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = make_dataset(1_000_000)

    def test_再現性(self):
        b = make_dataset(1_000_000)
        self.assertTrue(self.d.interactions.equals(b.interactions))
        self.assertTrue(np.array_equal(self.d.world.true_score, b.world.true_score))

    def test_シードで変わる(self):
        self.assertFalse(self.d.interactions["rating"].equals(make_dataset(1_000_001).interactions["rating"]))

    def test_観測に本当の好みの列が含まれない(self):
        self.assertFalse(set(self.d.interactions.columns) & set(ORACLE_COLUMNS))
        self.assertEqual(set(self.d.items.columns), {"item", "genre"})

    def test_分割は時間順(self):
        df = self.d.interactions
        a, b = SPLIT_TIMES
        self.assertTrue((df[df.split == "train"]["time"] < a).all())
        self.assertTrue(((df[df.split == "val"]["time"] >= a) & (df[df.split == "val"]["time"] < b)).all())
        self.assertTrue((df[df.split == "test"]["time"] >= b).all())

    def test_同じユーザーが同じ商品を2回評価しない(self):
        self.assertFalse(self.d.interactions.duplicated(["user", "item"]).any())

    def test_評価は1から5の整数(self):
        self.assertTrue(set(self.d.interactions["rating"]) <= {1, 2, 3, 4, 5})

    def test_観測は好きなものに偏る(self):
        # 観測された評価の平均は、本当の好みの全体平均より高い（好きなものほど評価されやすい）
        self.assertGreater(self.d.interactions["rating"].mean(), self.d.world.true_rating().mean() + 0.2)

    def test_人気は少数に集中する(self):
        cnt = self.d.train().groupby("item").size().reindex(range(self.d.world.n_items), fill_value=0)
        self.assertGreater(gini(cnt.to_numpy()), 0.4)


class TestMetrics(unittest.TestCase):
    def test_完璧な順位ならNDCGは1(self):
        scores = np.array([[0.9, 0.8, 0.1, 0.0]])
        test = pd.DataFrame({"user": [0, 0], "item": [0, 1], "rating": [5, 4]})
        tk = topk_items(scores, np.zeros((1, 4), bool), k=2)
        m = evaluate_observed(scores, tk, test, 1, 4, k=2)
        self.assertAlmostEqual(m["ndcg@2"], 1.0)
        self.assertAlmostEqual(m["precision@2"], 1.0)
        self.assertAlmostEqual(m["recall@2"], 1.0)

    def test_逆順なら指標は低い(self):
        scores = np.array([[0.0, 0.1, 0.8, 0.9]])
        test = pd.DataFrame({"user": [0, 0], "item": [0, 1], "rating": [5, 5]})
        tk = topk_items(scores, np.zeros((1, 4), bool), k=2)
        m = evaluate_observed(scores, tk, test, 1, 4, k=2)
        self.assertEqual(m["precision@2"], 0.0)
        self.assertEqual(m["hit@2"], 0.0)

    def test_ndcgは順位を反映する(self):
        test = pd.DataFrame({"user": [0], "item": [2], "rating": [5]})
        top = lambda s: topk_items(np.array([s]), np.zeros((1, 4), bool), k=3)   # noqa: E731
        first = evaluate_observed(None, top([0.1, 0.2, 0.9, 0.0]), test, 1, 4, k=3)["ndcg@3"]    # 関連ありが1位
        third = evaluate_observed(None, top([0.9, 0.8, 0.7, 0.0]), test, 1, 4, k=3)["ndcg@3"]    # 関連ありが3位
        self.assertAlmostEqual(first, 1.0)
        self.assertAlmostEqual(third, 0.5)            # 1/log2(4) = 0.5
        self.assertGreater(first, third)

    def test_知っている商品は薦めない(self):
        scores = np.array([[0.9, 0.8, 0.7]])
        ex = np.array([[True, False, False]])
        tk = topk_items(scores, ex, k=2)
        self.assertNotIn(0, tk[0])

    def test_関連ありがないユーザーは評価から除く(self):
        scores = np.random.default_rng(0).random((3, 5))
        test = pd.DataFrame({"user": [0], "item": [1], "rating": [5]})      # ユーザー1,2 は関連ありなし
        m = evaluate_observed(scores, topk_items(scores, np.zeros((3, 5), bool), 3), test, 3, 5, k=3)
        self.assertEqual(m["評価したユーザー数"], 1)

    def test_低い評価は関連ありにならない(self):
        scores = np.array([[0.9, 0.1]])
        test = pd.DataFrame({"user": [0], "item": [0], "rating": [2]})
        m = evaluate_observed(scores, topk_items(scores, np.zeros((1, 2), bool), 1), test, 1, 2, k=1)
        self.assertEqual(m["評価したユーザー数"], 0)

    def test_ジニ係数(self):
        self.assertAlmostEqual(gini([1, 1, 1, 1]), 0.0)
        self.assertGreater(gini([0, 0, 0, 10]), 0.7)

    def test_RMSEとMAE(self):
        pred = np.array([[4.0, 3.0]])
        test = pd.DataFrame({"user": [0, 0], "item": [0, 1], "rating": [5, 3]})
        e = rating_errors(pred, test)
        self.assertAlmostEqual(e["MAE"], 0.5)
        self.assertAlmostEqual(e["RMSE"], np.sqrt(0.5))

    def test_カバレッジと多様性(self):
        topk = np.array([[0, 1], [0, 2]])
        b = beyond_accuracy(topk, 4, np.array([5, 3, 1, 0]), np.array([0, 0, 1, 1]))
        self.assertAlmostEqual(b["カバレッジ"], 0.75)
        self.assertAlmostEqual(b["ジャンル数/リスト"], 1.5)


class TestBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = make_dataset(1_000_000)
        w = cls.d.world
        cls.U, cls.I = w.n_users, w.n_items
        cls.excl = exclusion_mask(cls.U, cls.I, cls.d.train())

    def _eval(self, S):
        tk = topk_items(S, self.excl, 10)
        return (evaluate_observed(S, tk, self.d.split("val"), self.U, self.I), evaluate_true(tk, self.d.world.true_rating(), self.excl))

    def test_人気順は観測の評価で_ランダムよりずっと良い(self):
        pop, _ = self._eval(popularity_scores(self.d.train(), self.U, self.I))
        rnd, _ = self._eval(random_scores(self.U, self.I))
        self.assertGreater(pop["ndcg@10"], rnd["ndcg@10"] * 5)

    def test_観測の評価では人気順が上限を上回る(self):
        """観測データは人気に偏った露出で作られるので、本当の好みで薦める上限より、人気順のほうが当たる"""
        pop, _ = self._eval(popularity_scores(self.d.train(), self.U, self.I))
        orc, _ = self._eval(oracle_scores(self.d.world.true_rating()))
        self.assertGreater(pop["precision@10"], orc["precision@10"])

    def test_本当の好みで評価すると_順位が逆転する(self):
        _, pop = self._eval(popularity_scores(self.d.train(), self.U, self.I))
        _, mean = self._eval(item_mean_scores(self.d.train(), self.U, self.I))
        _, orc = self._eval(oracle_scores(self.d.world.true_rating()))
        self.assertGreater(orc["precision@10"], mean["precision@10"])
        self.assertGreater(mean["precision@10"], pop["precision@10"])
        self.assertGreater(orc["precision@10"], 0.95)

    def test_人気順はカバレッジが低い(self):
        S = popularity_scores(self.d.train(), self.U, self.I)
        b = beyond_accuracy(topk_items(S, self.excl, 10), self.I, S[0], self.d.world.item_genre)
        self.assertLess(b["カバレッジ"], 0.2)

    def test_ランダムの本当のprecisionは関連ありの割合に近い(self):
        _, rnd = self._eval(random_scores(self.U, self.I))
        base = (self.d.world.true_rating() >= 4).mean()
        self.assertAlmostEqual(rnd["precision@10"], base, delta=0.05)



if __name__ == "__main__":
    unittest.main()


class TestPhase1(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from recsys_eval import (bias_pred, evaluate_observed_sampled, global_mean_pred, item_mean_pred, rating_rmse_all,
                                 user_mean_pred)
        cls.fns = dict(bias=bias_pred, glob=global_mean_pred, item=item_mean_pred, user=user_mean_pred,
                       sampled=evaluate_observed_sampled, rmse_all=rating_rmse_all)
        cls.d = make_dataset(1_000_000)
        w = cls.d.world
        cls.U, cls.I = w.n_users, w.n_items
        cls.tr, cls.val = cls.d.train(), cls.d.split("val")
        cls.excl = exclusion_mask(cls.U, cls.I, cls.tr)

    def test_バイアスモデルはRMSEが最も小さい(self):
        r = {k: rating_errors(self.fns[k](self.tr, self.U, self.I), self.val)["RMSE"] for k in ("glob", "item", "user", "bias")}
        self.assertLess(r["bias"], r["item"])
        self.assertLess(r["item"], r["glob"])
        self.assertLess(r["user"], r["glob"])

    def test_ユーザー平均はRMSEが改善しても順位は変わらない(self):
        """ユーザー平均は、1人の全商品に同じ値を予測する。RMSEは下がるが、商品の差がないので、順位の良さは全体平均と同じ（同点）"""
        g, u = self.fns["glob"](self.tr, self.U, self.I), self.fns["user"](self.tr, self.U, self.I)
        self.assertLess(rating_errors(u, self.val)["RMSE"], rating_errors(g, self.val)["RMSE"] - 0.05)
        mg = evaluate_observed(g, topk_items(g, self.excl, 10), self.val, self.U, self.I)["ndcg@10"]
        mu = evaluate_observed(u, topk_items(u, self.excl, 10), self.val, self.U, self.I)["ndcg@10"]
        self.assertAlmostEqual(mg, mu, delta=0.003)

    def test_全部の組でのRMSE(self):
        w = self.d.world
        self.assertAlmostEqual(self.fns["rmse_all"](w.true_rating(), w.true_rating()), 0.0)
        self.assertGreater(self.fns["rmse_all"](np.full_like(w.true_rating(), 1.0), w.true_rating()), 1.5)

    def test_サンプリング評価は全商品の評価より高い数字が出る(self):
        S = random_scores(self.U, self.I)
        full = evaluate_observed(S, topk_items(S, self.excl, 10), self.val, self.U, self.I)["ndcg@10"]
        sam = self.fns["sampled"](S, self.val, self.excl, 100)["ndcg@10"]
        self.assertGreater(sam, full * 2)

    def test_サンプリング評価の対象ユーザー数は同じ(self):
        S = popularity_scores(self.tr, self.U, self.I)
        a = evaluate_observed(S, topk_items(S, self.excl, 10), self.val, self.U, self.I)["評価したユーザー数"]
        b = self.fns["sampled"](S, self.val, self.excl, 100)["評価したユーザー数"]
        self.assertEqual(a, b)

    def test_人気比例で選んだ候補では_人気順の優位が小さくなる(self):
        S = popularity_scores(self.tr, self.U, self.I)
        pop = S[0]
        uni = self.fns["sampled"](S, self.val, self.excl, 100)["ndcg@10"]
        bypop = self.fns["sampled"](S, self.val, self.excl, 100, neg_by_pop=pop)["ndcg@10"]
        self.assertGreater(uni, bypop)

    def test_評価の順位は偏りが強いほど本当の順位とずれる(self):
        def corr(pp):
            from scipy.stats import spearmanr
            d = make_dataset(1_000_000, pop_power=pp)
            w, tr, val = d.world, d.train(), d.split("val")
            ex = exclusion_mask(w.n_users, w.n_items, tr)
            S = {"人気順": popularity_scores(tr, w.n_users, w.n_items), "平均評価順": item_mean_scores(tr, w.n_users, w.n_items),
                 "ランダム": random_scores(w.n_users, w.n_items), "高評価人気": popularity_scores(tr, w.n_users, w.n_items, True)}
            o = [evaluate_observed(s, topk_items(s, ex, 10), val, w.n_users, w.n_items)["ndcg@10"] for s in S.values()]
            t = [evaluate_true(topk_items(s, ex, 10), w.true_rating(), ex)["precision@10"] for s in S.values()]
            return spearmanr(o, t)[0]
        self.assertGreater(corr(0.0), corr(1.5))


class TestNeighbors(unittest.TestCase):
    def test_行列の形と値(self):
        from recsys_models import interaction_matrix
        df = pd.DataFrame({"user": [0, 0, 1], "item": [0, 2, 1], "rating": [5, 2, 4]})
        m = interaction_matrix(df, 2, 3, "interactions")
        self.assertEqual(m.tolist(), [[1, 0, 1], [0, 1, 0]])
        self.assertEqual(interaction_matrix(df, 2, 3, "liked").tolist(), [[1, 0, 0], [0, 1, 0]])

    def test_アイテム類似度は対称で対角は0(self):
        from recsys_models import item_similarity
        B = (np.random.default_rng(0).random((30, 8)) < 0.4).astype(float)
        s = item_similarity(B, 0.5)
        self.assertTrue(np.allclose(s, s.T))
        self.assertTrue(np.allclose(np.diag(s), 0.0))

    def test_近傍数kで各行の非ゼロは高々k(self):
        from recsys_models import item_similarity
        B = (np.random.default_rng(1).random((60, 10)) < 0.5).astype(float)
        s = item_similarity(B, 0.5, k=3)
        self.assertTrue(((s > 0).sum(axis=1) <= 3).all())

    def test_収縮は共起の少ない類似度を小さくする(self):
        from recsys_models import item_similarity
        B = np.array([[1, 1, 0], [1, 1, 0], [1, 0, 1]], dtype=float)     # 商品0-1の共起は2、0-2の共起は1
        a, b = item_similarity(B, 0.5, 0.0), item_similarity(B, 0.5, 5.0)
        self.assertTrue((b <= a + 1e-12).all())
        self.assertLess(b[0, 2] / a[0, 2], b[0, 1] / a[0, 1])             # 共起が少ないほうが、より小さくなる

    def test_一緒に評価される商品を薦める(self):
        from recsys_models import item_knn_scores
        df = pd.DataFrame({"user": [0, 0, 1, 1, 2], "item": [0, 1, 0, 1, 0], "rating": [5] * 5})
        S = item_knn_scores(df, 3, 3, "interactions", 0.5, 0.0)
        self.assertGreater(S[2, 1], S[2, 2])        # ユーザー2は商品0を評価 → 商品0と一緒に評価される商品1が、商品2より上

    def test_ユーザー型は似たユーザーが好きな商品を薦める(self):
        from recsys_models import user_knn_scores
        df = pd.DataFrame({"user": [0, 0, 1, 1, 1, 2], "item": [0, 1, 0, 1, 2, 3], "rating": [5] * 6})
        S = user_knn_scores(df, 3, 4, "interactions", 0.5, 0.0)
        self.assertGreater(S[0, 2], S[0, 3])        # ユーザー0に似たユーザー1が評価した商品2のほうが、似ていないユーザー2の商品3より上

    def test_評価した入力の近傍法は人気順に近い(self):
        from scipy.stats import spearmanr
        from recsys_models import item_knn_scores
        d = make_dataset(1_000_000)
        w, tr = d.world, d.train()
        S = item_knn_scores(tr, w.n_users, w.n_items, "interactions", 0.5, 5.0)
        pop = popularity_scores(tr, w.n_users, w.n_items)[0]
        self.assertGreater(spearmanr(S.mean(axis=0), pop)[0], 0.9)

    def test_好きを入力にすると本当の好みでの当たりが上がる(self):
        from recsys_models import item_knn_scores
        d = make_dataset(1_000_000)
        w, tr = d.world, d.train()
        ex = exclusion_mask(w.n_users, w.n_items, tr)
        prec = lambda S: evaluate_true(topk_items(S, ex, 10), w.true_rating(), ex)["precision@10"]    # noqa: E731
        a = prec(item_knn_scores(tr, w.n_users, w.n_items, "interactions", 0.5, 5.0))
        b = prec(item_knn_scores(tr, w.n_users, w.n_items, "liked", 0.5, 5.0))
        self.assertGreater(b, a + 0.05)

    def test_評価ゼロのユーザーにはスコアが出ない(self):
        from recsys_models import item_knn_scores
        df = pd.DataFrame({"user": [0, 0], "item": [0, 1], "rating": [5, 5]})
        S = item_knn_scores(df, 2, 3, "interactions", 0.5, 0.0)
        self.assertTrue((S[1] == 0).all())           # ユーザー1は評価が無い（コールドスタート）


class TestFactorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = make_dataset(1_000_000)
        w = cls.d.world
        cls.U, cls.I = w.n_users, w.n_items
        cls.tr, cls.val = cls.d.train(), cls.d.split("val")
        cls.excl = exclusion_mask(cls.U, cls.I, cls.tr)

    def test_低ランクの評価行列を復元できる(self):
        from recsys_models import als_fit
        rng = np.random.default_rng(0)
        Pt, Qt = rng.normal(0, 1, (60, 2)), rng.normal(0, 1, (40, 2))
        R = 3.5 + Pt @ Qt.T * 0.5
        mask = rng.random(R.shape) < 0.5
        u, i = np.where(mask)
        df = pd.DataFrame({"user": u, "item": i, "rating": R[mask]})
        m = als_fit(df, 60, 40, k=2, reg=0.1, iters=30)
        pred = m.predict()
        self.assertLess(np.sqrt(np.mean((pred[~mask] - R[~mask]) ** 2)), 0.3)     # 見ていない組も、よく当てる

    def test_予測の形と学習データへの当てはまり(self):
        from recsys_models import als_fit
        m = als_fit(self.tr, self.U, self.I, k=4, reg=5.0)
        P = m.predict()
        self.assertEqual(P.shape, (self.U, self.I))
        train_rmse = rating_errors(P, self.tr)["RMSE"]
        self.assertLess(train_rmse, rating_errors(P, self.val)["RMSE"])            # 学習データのほうが、当てはまりが良い

    def test_評価ゼロのユーザーは平均と商品のバイアスだけ(self):
        from recsys_models import als_fit
        df = self.tr[self.tr["user"] != 0]
        m = als_fit(df, self.U, self.I, k=4, reg=5.0)
        self.assertTrue(np.allclose(m.predict()[0], m.mu + m.bi))

    def test_同じシードなら同じ結果(self):
        from recsys_models import als_fit, bpr_fit
        a = als_fit(self.tr, self.U, self.I, k=4, reg=5.0, seed=1).predict()
        b = als_fit(self.tr, self.U, self.I, k=4, reg=5.0, seed=1).predict()
        self.assertTrue(np.allclose(a, b))
        c = bpr_fit(self.tr, self.U, self.I, k=8, epochs=5, seed=2).predict()
        e = bpr_fit(self.tr, self.U, self.I, k=8, epochs=5, seed=2).predict()
        self.assertTrue(np.allclose(c, e))

    def test_ALSは本当の好みで平均評価順を上回り_観測では人気順を下回る(self):
        from recsys_models import als_fit
        P = als_fit(self.tr, self.U, self.I, k=4, reg=5.0).predict()
        tk = topk_items(P, self.excl, 10)
        true = evaluate_true(tk, self.d.world.true_rating(), self.excl)["precision@10"]
        base = evaluate_true(topk_items(item_mean_scores(self.tr, self.U, self.I), self.excl, 10), self.d.world.true_rating(), self.excl)["precision@10"]
        self.assertGreater(true, base + 0.08)
        obs = evaluate_observed(P, tk, self.val, self.U, self.I)["ndcg@10"]
        pop = popularity_scores(self.tr, self.U, self.I)
        self.assertLess(obs, evaluate_observed(pop, topk_items(pop, self.excl, 10), self.val, self.U, self.I)["ndcg@10"])

    def test_BPRは好きな商品を知らない商品より上にする(self):
        from recsys_models import bpr_fit
        m = bpr_fit(self.tr, self.U, self.I, k=16, epochs=30, seed=0)
        P = m.predict()
        pos = self.tr[self.tr["rating"] >= 4]
        rng = np.random.default_rng(0)
        j = rng.integers(0, self.I, len(pos))
        ok = ~((self.excl[pos["user"].to_numpy(), j]))
        auc = (P[pos["user"].to_numpy(), pos["item"].to_numpy()][ok] > P[pos["user"].to_numpy(), j][ok]).mean()
        self.assertGreater(auc, 0.8)

    def test_人気に比例した外し役は_人気の偏りを弱める(self):
        from recsys_models import bpr_fit
        pop = popularity_scores(self.tr, self.U, self.I)[0]
        def mean_pop(npp):
            S = bpr_fit(self.tr, self.U, self.I, k=16, epochs=30, neg_pop_power=npp, seed=0).predict()
            return pop[topk_items(S, self.excl, 10)].mean()
        self.assertGreater(mean_pop(0.0), mean_pop(1.0) * 2)

    def test_部分空間の重なり(self):
        from run_phase3 import subspace_overlap
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (50, 3))
        X = X - X.mean(0)                                   # 列の平均を0にしてから直交化（平均0の空間にする）
        V = np.linalg.qr(X)[0]
        self.assertAlmostEqual(subspace_overlap(V @ rng.normal(0, 1, (3, 3)), V), 1.0, places=6)   # 同じ空間なら 1
        W = rng.normal(0, 1, (50, 3))
        self.assertLess(subspace_overlap(W, V), 0.4)                                                # 無関係な空間なら、小さい


class TestColdStart(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = make_dataset(1_000_000)
        w = cls.d.world
        cls.U, cls.I = w.n_users, w.n_items
        cls.tr = cls.d.train()

    def test_ジャンル好みは_ユーザーの高評価ジャンルを上にする(self):
        from recsys_models import genre_profile_scores
        tr = pd.DataFrame({"user": [0, 0, 0, 0], "item": [0, 1, 2, 3], "rating": [5, 5, 1, 1]})
        genre = np.array([0, 0, 1, 1, 2])
        S = genre_profile_scores(tr, genre, 1, 3, weight=1.0)
        self.assertGreater(S[0, 0], S[0, 4])        # 高く評価したジャンル 0 > 評価の無いジャンル 2（全体平均に寄る）
        self.assertGreater(S[0, 4], S[0, 2])        # 評価の無いジャンル 2 > 低く評価したジャンル 1
        self.assertEqual(S[0, 0], S[0, 1])          # 同じジャンルは同点

    def test_評価0件の新商品は_事前分布そのもの(self):
        from recsys_models import als_fit, fold_in_items
        m = als_fit(self.tr, self.U, self.I, k=4, reg=5.0)
        cold = np.array([3, 7])
        prior = np.tile(np.array([0.5, 0.1, 0.2, 0.3, 0.4]), (2, 1))
        S = fold_in_items(m, self.tr.iloc[:0], cold, prior)
        want = m.mu + m.bu + 0.5 + m.P @ prior[0, 1:]
        self.assertTrue(np.allclose(S[:, 3], want))
        self.assertTrue(np.allclose(S[:, 0], m.predict()[:, 0]))      # 他の商品は変わらない

    def test_評価が増えると_新商品の因子は事前分布から離れてデータに近づく(self):
        from recsys_models import als_fit, fold_in_items
        m = als_fit(self.tr, self.U, self.I, k=4, reg=5.0)
        i = int(self.tr.groupby("item").size().idxmax())
        d = self.tr[self.tr["item"] == i]
        prior = np.zeros((1, 5))
        S0 = fold_in_items(m, d.iloc[:0], np.array([i]), prior)[:, i]
        S1 = fold_in_items(m, d, np.array([i]), prior)[:, i]
        full = m.predict()[:, i]
        self.assertLess(np.abs(S1 - full).mean(), np.abs(S0 - full).mean())

    def test_新ユーザーのフォールドインは_評価が無ければ_バイアスモデル(self):
        from recsys_models import als_fit, fold_in_users
        m = als_fit(self.tr, self.U, self.I, k=4, reg=5.0)
        S = fold_in_users(m, self.tr.iloc[:0], np.array([5]))
        self.assertTrue(np.allclose(S[5], m.mu + m.bi))
        self.assertTrue(np.allclose(S[0], m.predict()[0]))             # 他のユーザーは変わらない

    def test_新ユーザーは_評価が増えるほど本当の好みに合う(self):
        from recsys_eval import rating_rmse_all
        from recsys_models import als_fit, fold_in_users
        w = self.d.world
        cold = np.arange(0, 100)
        warm_tr = self.tr[~self.tr["user"].isin(cold)]
        m = als_fit(warm_tr, self.U, self.I, k=4, reg=5.0)
        full = self.tr[self.tr["user"].isin(cold)].sort_values("time")
        err = []
        for n in (0, 10, 40):
            S = fold_in_users(m, full.groupby("user").head(n) if n else full.iloc[:0], cold)
            err.append(rating_rmse_all(S[cold], w.true_rating()[cold]))
        self.assertGreater(err[0], err[2])


class TestDebias(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = make_dataset(1_000_000)
        w = cls.d.world
        cls.U, cls.I = w.n_users, w.n_items
        cls.tr = cls.d.train()

    def test_傾向は人気の商品ほど大きく_重みは逆になる(self):
        from recsys_eval import ipw_weights, item_propensity
        prop = item_propensity(self.tr, self.U, self.I)
        cnt = np.bincount(self.tr["item"], minlength=self.I)
        self.assertGreater(prop[cnt.argmax()], prop[cnt.argmin()])
        w = ipw_weights(self.tr, prop, 1.0)
        self.assertAlmostEqual(float(w.mean()), 1.0, places=9)
        lo = np.flatnonzero(cnt > 0)[cnt[cnt > 0].argmin()]                                # 評価が1件以上ある、最も少ない商品
        i_hi, i_lo = self.tr["item"].to_numpy() == cnt.argmax(), self.tr["item"].to_numpy() == lo
        self.assertLess(w[i_hi].mean(), w[i_lo].mean())
        self.assertTrue(np.allclose(ipw_weights(self.tr, prop, 0.0), 1.0))           # γ=0 は重み無し

    def test_重み付きrecallは_稀な商品の当たりを重く見る(self):
        from recsys_eval import recall_observed
        test = pd.DataFrame({"user": [0, 0], "item": [0, 1], "rating": [5, 5]})
        topk = np.array([[0, 2]])                                                     # 商品0だけ当たり
        self.assertAlmostEqual(recall_observed(topk, test, 1, 3), 0.5)
        self.assertAlmostEqual(recall_observed(topk, test, 1, 3, weight=np.array([1.0, 3.0, 1.0])), 0.25)   # 当たりが軽い商品なら下がる

    def test_ALSの重みが全部1なら重み無しと同じ_偏れば結果が変わる(self):
        from recsys_models import als_fit
        a = als_fit(self.tr, self.U, self.I, k=4, reg=5.0).predict()
        b = als_fit(self.tr, self.U, self.I, k=4, reg=5.0, sample_weight=np.ones(len(self.tr))).predict()
        self.assertTrue(np.allclose(a, b))
        wt = np.where(self.tr["item"].to_numpy() < 50, 5.0, 1.0)
        c = als_fit(self.tr, self.U, self.I, k=4, reg=5.0, sample_weight=wt).predict()
        self.assertFalse(np.allclose(a, c))

    def test_消費のシミュレーション_好きな商品ほど評価される(self):
        from run_phase5 import _consume
        w = self.d.world
        rng = np.random.default_rng(0)
        recs = np.tile(np.arange(10), (self.U, 1))
        df = _consume(w, recs, rng, organic=0)
        tru = w.true_rating()
        shown_liked = tru[:, :10] >= 4
        taken = np.zeros((self.U, 10), bool)
        taken[df["user"].to_numpy(), df["item"].to_numpy()] = True
        if shown_liked.any() and (~shown_liked).any():
            self.assertGreater(taken[shown_liked].mean(), taken[~shown_liked].mean())


class TestFinalEval(unittest.TestCase):
    def test_最終評価の世界は_開発用と重ならず_testは使用済みで拒否する(self):
        import final_eval
        dev = {1_000_000 + i for i in range(10)} | {2_000_000 + i for i in range(100)}
        for split, seeds in final_eval.SPLITS.items():
            self.assertFalse(dev & set(seeds))
        self.assertTrue(all(sd >= 3_000_000 for sd in final_eval.SPLITS["test"]))
        self.assertFalse(set(final_eval.SPLITS["val"]) & set(final_eval.SPLITS["test"]))

