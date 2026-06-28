"""Tests for the prediction engine and core views."""
from django.test import TestCase
from django.urls import reverse

from .models import TeamPerformance
from . import predictions, analytics


def _make_week(team, week, points, opponent, opp_points, year=2025):
    """Create one team-week row (result inferred from the score vs opponent)."""
    result = "W" if points > opp_points else ("L" if points < opp_points else "T")
    return TeamPerformance.objects.create(
        team_name=team, week=f"Week {week}", year=year,
        qb_points=0, wr_points=0, wr_points_total=0, rb_points=0, rb_points_total=0,
        te_points=0, te_points_total=0, k_points=0, def_points=0,
        total_points=points, expected_total=points, difference=0,
        points_against=opp_points, opponent=opponent, result=result,
        margin=points - opp_points,
    )


class PredictionEngineTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Strong team A (~120/wk), weak team B (~90/wk), over 4 head-to-heads.
        a_scores = [118, 122, 120, 124]
        b_scores = [92, 88, 95, 85]
        for wk, (a, b) in enumerate(zip(a_scores, b_scores), start=1):
            _make_week("A", wk, a, "B", b)
            _make_week("B", wk, b, "A", a)

    def test_team_stats_shape_and_values(self):
        stats = predictions.team_stats(2025)
        self.assertIn("A", stats)
        self.assertEqual(stats["A"]["games"], 4)
        self.assertAlmostEqual(stats["A"]["mean"], sum([118, 122, 120, 124]) / 4, places=4)
        self.assertEqual(stats["A"]["wins"], 4)   # A beat B every week
        self.assertEqual(stats["B"]["losses"], 4)
        self.assertGreater(stats["A"]["net"], 0)  # outscored opponents
        self.assertLess(stats["B"]["net"], 0)

    def test_win_probability_favors_stronger_team(self):
        p = predictions.win_probability("A", "B", 2025)
        self.assertIsNotNone(p)
        self.assertGreater(p["p_a"], 0.5)
        self.assertGreater(p["expected_margin"], 0)

    def test_win_probability_is_symmetric(self):
        ab = predictions.win_probability("A", "B", 2025)["p_a"]
        ba = predictions.win_probability("B", "A", 2025)["p_a"]
        self.assertAlmostEqual(ab + ba, 1.0, places=9)

    def test_win_probability_self_is_fifty_fifty(self):
        self.assertAlmostEqual(predictions.win_probability("A", "A", 2025)["p_a"], 0.5, places=9)

    def test_win_probability_unknown_team_returns_none(self):
        self.assertIsNone(predictions.win_probability("A", "Nobody", 2025))
        self.assertIsNone(predictions.win_probability("A", "B", 1999))  # no data that year

    def test_power_ratings_sorted_and_ranked(self):
        rows = predictions.power_ratings(2025)
        self.assertEqual([r["team"] for r in rows], ["A", "B"])  # A scores more
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[1]["rank"], 2)
        self.assertGreaterEqual(rows[0]["rating"], rows[1]["rating"])

    def test_empty_season(self):
        self.assertEqual(predictions.team_stats(2099), {})
        self.assertEqual(predictions.power_ratings(2099), [])


class CoreViewSmokeTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        scores = {"A": [118, 122, 120, 124], "B": [92, 88, 95, 85]}
        for wk in range(4):
            _make_week("A", wk + 1, scores["A"][wk], "B", scores["B"][wk])
            _make_week("B", wk + 1, scores["B"][wk], "A", scores["A"][wk])

    def test_pages_return_200(self):
        for name in ["home", "power_rankings", "versus", "win_probability_heatmap",
                     "win_probability_against_all_teams", "top_tens", "stats_charts"]:
            with self.subTest(view=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 200)

    def test_versus_with_teams_predicts(self):
        resp = self.client.get(reverse("versus"), {"team1": "A", "team2": "B"})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "favored")  # prediction sentence rendered


class LuckReportTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # One week, 4 teams, two matchups. Scores A > B > C > D.
        _make_week("A", 1, 110, "D", 80)   # A wins
        _make_week("D", 1, 80, "A", 110)   # D loses
        _make_week("B", 1, 100, "C", 95)   # B wins
        _make_week("C", 1, 95, "B", 100)   # C loses

    def test_allplay_expected_wins(self):
        # n=4 -> each team's expected wins = (teams outscored) / 3
        rep = {r["team"]: r for r in analytics.luck_report(2025)}
        self.assertAlmostEqual(rep["A"]["allplay_expected_wins"], 1.0, places=4)
        self.assertAlmostEqual(rep["B"]["allplay_expected_wins"], 2 / 3, places=4)
        self.assertAlmostEqual(rep["C"]["allplay_expected_wins"], 1 / 3, places=4)
        self.assertAlmostEqual(rep["D"]["allplay_expected_wins"], 0.0, places=4)

    def test_scoring_luck_and_sort_order(self):
        rep = analytics.luck_report(2025)
        by_team = {r["team"]: r for r in rep}
        # B won its game but only "deserved" 2/3 of a win -> mildly lucky.
        self.assertAlmostEqual(by_team["B"]["scoring_luck"], 1 - 2 / 3, places=4)
        self.assertAlmostEqual(by_team["C"]["scoring_luck"], -1 / 3, places=4)
        lucks = [r["scoring_luck"] for r in rep]
        self.assertEqual(lucks, sorted(lucks, reverse=True))  # luckiest first
