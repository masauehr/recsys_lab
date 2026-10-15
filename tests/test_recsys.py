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
