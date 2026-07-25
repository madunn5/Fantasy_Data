"""Tests for the prediction engine and core views."""
from django.test import TestCase
from django.urls import reverse

from .models import TeamPerformance, ScheduledMatchup, TeamOwnerMapping
from . import predictions, analytics
from .yahoo_collector import YahooFantasyCollector, is_starter_slot


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

    def test_through_week_limits_the_game_log(self):
        stats = predictions.team_stats(2025, through_week=2)
        self.assertEqual(stats["A"]["games"], 2)
        self.assertAlmostEqual(stats["A"]["mean"], (118 + 122) / 2, places=4)
        self.assertEqual(predictions.team_stats(2025, through_week=0), {})

    def test_mu_is_shrunk_toward_league_mean(self):
        # A (~121) and B (~90) straddle the league mean (~105.5), so shrinkage
        # pulls A's rating down and B's up — strictly between own mean and league mean.
        stats = predictions.team_stats(2025)
        league_mean = (stats["A"]["mean"] + stats["B"]["mean"]) / 2
        self.assertLess(stats["A"]["mu"], stats["A"]["recent_mean"])
        self.assertGreater(stats["A"]["mu"], league_mean)
        self.assertGreater(stats["B"]["mu"], stats["B"]["recent_mean"])
        self.assertLess(stats["B"]["mu"], league_mean)


class RecencyWeightingTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # Same season average (100), opposite trajectories: U is surging
        # (trade deadline win), D is fading (star injured).
        u_scores = [80, 90, 110, 120]
        d_scores = [120, 110, 90, 80]
        for wk, (u, d) in enumerate(zip(u_scores, d_scores), start=1):
            _make_week("U", wk, u, "D", d)
            _make_week("D", wk, d, "U", u)

    def test_recent_mean_tracks_form_not_season_average(self):
        stats = predictions.team_stats(2025)
        self.assertAlmostEqual(stats["U"]["mean"], 100.0, places=4)
        self.assertAlmostEqual(stats["D"]["mean"], 100.0, places=4)
        self.assertGreater(stats["U"]["recent_mean"], 100.0)
        self.assertLess(stats["D"]["recent_mean"], 100.0)

    def test_win_probability_favors_the_team_trending_up(self):
        p = predictions.win_probability("U", "D", 2025)
        self.assertGreater(p["p_a"], 0.5)
        self.assertGreater(p["expected_margin"], 0)

    def test_shrunk_single_game_team_is_not_taken_literally(self):
        # One 160-point week shouldn't rate as a 160-point team.
        _make_week("N", 4, 160, "X", 70)
        _make_week("X", 4, 70, "N", 160)
        stats = predictions.team_stats(2025)
        self.assertEqual(stats["N"]["games"], 1)
        self.assertLess(stats["N"]["mu"], 140)         # pulled well down from 160
        self.assertGreater(stats["N"]["mu"], 100)      # but still above average
        self.assertGreater(stats["N"]["std"], 0)       # borrowed league spread


class CoreViewSmokeTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        scores = {"A": [118, 122, 120, 124], "B": [92, 88, 95, 85]}
        for wk in range(4):
            _make_week("A", wk + 1, scores["A"][wk], "B", scores["B"][wk])
            _make_week("B", wk + 1, scores["B"][wk], "A", scores["A"][wk])

    def test_pages_return_200(self):
        for name in ["home", "power_rankings", "versus", "win_probability_heatmap",
                     "win_probability_against_all_teams", "top_tens", "stats_charts",
                     "playoff_odds", "luck_report"]:
            with self.subTest(view=name):
                self.assertEqual(self.client.get(reverse(name)).status_code, 200)

    def test_versus_with_teams_predicts(self):
        resp = self.client.get(reverse("versus"), {"team1": "A", "team2": "B"})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "favorite")  # prediction sentence rendered


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


class ScheduleAndSimulationTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        # 4 played weeks: A strong (~121), B weak (~90).
        a_scores = [118, 122, 120, 124]
        b_scores = [92, 88, 95, 85]
        for wk, (a, b) in enumerate(zip(a_scores, b_scores), start=1):
            _make_week("A", wk, a, "B", b)
            _make_week("B", wk, b, "A", a)
        # Future weeks 5-6 exist only in the stored Yahoo schedule, under the
        # raw Yahoo team names, mapped to owner names A/B.
        TeamOwnerMapping.objects.create(team_name="Team Alpha", owner_name="A", year=2025)
        TeamOwnerMapping.objects.create(team_name="Team Bravo", owner_name="B", year=2025)
        for wk in (5, 6):
            ScheduledMatchup.objects.create(year=2025, week=wk,
                                            team_a="Team Alpha", team_b="Team Bravo")

    def test_season_schedule_merges_played_and_future_with_name_mapping(self):
        sched = analytics.season_schedule(2025)
        self.assertEqual(sorted(sched), [1, 2, 3, 4, 5, 6])
        self.assertEqual(sched[5], [("A", "B")])  # Yahoo names -> owner names

    def test_simulation_covers_future_scheduled_games(self):
        sim = analytics.simulate_season(2025, n_sims=300, playoff_spots=1)
        self.assertEqual(sim["through_week"], 4)       # defaults to latest played
        self.assertEqual(sim["latest_played"], 4)
        self.assertEqual(sim["remaining_games"], 2)    # weeks 5 and 6 simulated
        by_team = {r["team"]: r for r in sim["teams"]}
        # A is 4-0 with a huge scoring edge; two remaining games can't cost the seed.
        self.assertGreater(by_team["A"]["playoff_pct"], 95)
        for r in sim["teams"]:
            self.assertGreaterEqual(r["playoff_pct"], 0)
            self.assertLessEqual(r["playoff_pct"], 100)

    def test_extreme_odds_are_labelled_not_absolute(self):
        # A is 4-0 with 2 games left against the only other team; the seed is
        # safe, but with games remaining the label must not claim 100%.
        sim = analytics.simulate_season(2025, n_sims=300, playoff_spots=1)
        by_team = {r["team"]: r for r in sim["teams"]}
        self.assertEqual(by_team["A"]["pct_label"], ">99")
        self.assertEqual(by_team["B"]["pct_label"], "<1")

    def test_retrospective_sim_has_no_lookahead(self):
        # Simulating from week 2 must estimate scoring from weeks 1-2 only.
        sim = analytics.simulate_season(2025, through_week=2, n_sims=50, playoff_spots=1)
        self.assertEqual(sim["through_week"], 2)
        self.assertEqual(sim["remaining_games"], 4)    # weeks 3-6
        stats = predictions.team_stats(2025, through_week=2)
        self.assertEqual(stats["A"]["games"], 2)


class PositionPointsTests(TestCase):
    """Flex normalization and starter detection in the Yahoo collector."""

    @staticmethod
    def _player(pos, slot, pts):
        return {
            "name": f"{pos} in {slot}",
            "selected_position": slot,
            "eligible_positions": [pos],
            "player_points": {"total": pts},
        }

    def test_wr_flex_normalizes_and_ir_is_ignored(self):
        roster = [
            self._player("QB", "QB", 20),
            self._player("WR", "WR", 10),
            self._player("WR", "WR", 12),
            self._player("WR", "W/R/T", 8),   # WR in the flex
            self._player("RB", "RB", 15),
            self._player("RB", "RB", 9),
            self._player("TE", "TE", 7),
            self._player("K", "K", 6),
            self._player("DEF", "DEF", 5),
            self._player("RB", "IR", 0.0),    # IR stash: must not count as a starter
            self._player("WR", "BN", 22.0),   # bench: never counted
        ]
        pts = YahooFantasyCollector.calculate_position_points(roster)
        self.assertAlmostEqual(pts["WR_Points_Total"], 30.0)
        self.assertAlmostEqual(pts["WR_Points"], 20.0)       # 3 WRs -> x 2/3
        self.assertAlmostEqual(pts["RB_Points_Total"], 24.0)
        self.assertAlmostEqual(pts["RB_Points"], 24.0)       # IR RB excluded: no normalization
        self.assertAlmostEqual(pts["TE_Points"], 7.0)
        self.assertAlmostEqual(pts["QB_Points"], 20.0)

    def test_te_flex_averages_both_tight_ends(self):
        roster = [
            self._player("TE", "TE", 10),
            self._player("TE", "W/R/T", 6),   # TE in the flex
        ]
        pts = YahooFantasyCollector.calculate_position_points(roster)
        self.assertAlmostEqual(pts["TE_Points_Total"], 16.0)
        self.assertAlmostEqual(pts["TE_Points"], 8.0)        # 2 TEs -> average

    def test_starter_slot_detection(self):
        for slot in ("QB", "WR", "RB", "TE", "K", "DEF", "W/R/T"):
            self.assertTrue(is_starter_slot(slot))
        for slot in ("BN", "IR", "IR-R", "NA", None, ""):
            self.assertFalse(is_starter_slot(slot))
