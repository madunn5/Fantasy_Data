"""Tests for the prediction engine and core views."""
from django.test import TestCase
from django.urls import reverse

from .models import TeamPerformance, ScheduledMatchup, TeamOwnerMapping
from . import predictions, analytics, player_impact
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


class PlayerImpactTests(TestCase):
    """League edge, costly benchings, and roster moves on a tiny synthetic season."""

    @staticmethod
    def _pp(name, pos, team, week, pts, started):
        from .models import Player, PlayerPerformance
        player, _ = Player.objects.get_or_create(
            yahoo_player_id=name, defaults={"name": name, "position": pos, "nfl_team": "N/A"})
        PlayerPerformance.objects.create(
            player=player, fantasy_team=team, week=f"Week {week}", year=2025,
            points_scored=pts, was_started=started)

    @classmethod
    def setUpTestData(cls):
        # Two matchups a week: T1 vs T2, T3 vs T4.
        _make_week("T1", 1, 100, "T2", 90)
        _make_week("T2", 1, 90, "T1", 100)
        _make_week("T3", 1, 95, "T4", 80)
        _make_week("T4", 1, 80, "T3", 95)
        _make_week("T1", 2, 105, "T2", 95)
        _make_week("T2", 2, 95, "T1", 105)
        _make_week("T3", 2, 99, "T4", 90)
        _make_week("T4", 2, 90, "T3", 99)

        # Week 1 QBs (median 21): the star's 19-point edge exceeds T1's
        # 10-point margin, so he swung that win.
        cls._pp("Star QB", "QB", "T1", 1, 40, True)
        cls._pp("QB Two", "QB", "T2", 1, 20, True)
        cls._pp("QB Three", "QB", "T3", 1, 22, True)
        cls._pp("QB Four", "QB", "T4", 1, 18, True)
        # Week 2 QBs (median 21): 9-point edge < 10-point margin, no swing.
        cls._pp("Star QB", "QB", "T1", 2, 30, True)
        cls._pp("QB Two", "QB", "T2", 2, 20, True)
        cls._pp("QB Three", "QB", "T3", 2, 22, True)
        cls._pp("QB Four", "QB", "T4", 2, 18, True)

        # Week 1 WRs. T2 benched a 25-point WR while starting a 5-pointer
        # and lost by 10: starting him wins by 10.
        cls._pp("Traded X", "WR", "T1", 1, 10, True)
        cls._pp("WR A", "WR", "T2", 1, 5, True)
        cls._pp("Traded Y", "WR", "T2", 1, 7, True)
        cls._pp("WR B", "WR", "T3", 1, 8, True)
        cls._pp("WR C", "WR", "T4", 1, 12, True)
        cls._pp("Benched WR", "WR", "T2", 1, 25, False)
        # Week 2: X and Y swap teams (a trade); FA Hero appears from waivers.
        cls._pp("Traded X", "WR", "T2", 2, 15, True)
        cls._pp("Traded Y", "WR", "T1", 2, 14, True)
        cls._pp("WR B", "WR", "T3", 2, 8, True)
        cls._pp("WR C", "WR", "T4", 2, 12, True)
        cls._pp("FA Hero", "WR", "T3", 2, 20, True)

    def test_league_edge_and_wins_swung(self):
        rows = {r["name"]: r for r in player_impact.league_edge(2025, min_starts=1)}
        star = rows["Star QB"]
        self.assertAlmostEqual(star["edge_total"], 28.0)   # 19 + 9
        self.assertEqual(star["wins_swung"], 1)
        self.assertEqual(star["swing_weeks"], [1])
        # A benched-only player never qualifies for the edge table.
        self.assertNotIn("Benched WR", rows)

    def test_costly_benching_flags_the_game(self):
        rep = player_impact.costly_benchings(2025)
        self.assertEqual(len(rep["games"]), 1)
        g = rep["games"][0]
        self.assertEqual((g["team"], g["week"]), ("T2", 1))
        self.assertAlmostEqual(g["lost_by"], 10.0)
        self.assertEqual(g["options"][0]["name"], "Benched WR")
        self.assertAlmostEqual(g["options"][0]["would_win_by"], 10.0)
        self.assertEqual(rep["team_counts"][0], {"team": "T2", "games": 1})

    def test_trade_is_grouped_with_both_sides_scored(self):
        moves = player_impact.roster_moves(2025)
        self.assertEqual(len(moves["trades"]), 1)
        t = moves["trades"][0]
        self.assertEqual(t["week"], 2)
        self.assertEqual({t["team_a"], t["team_b"]}, {"T1", "T2"})
        got = {t["team_a"]: t["got_a"], t["team_b"]: t["got_b"]}
        self.assertEqual([m["name"] for m in got["T2"]], ["Traded X"])
        self.assertEqual([m["name"] for m in got["T1"]], ["Traded Y"])
        # Week 2 WR median is 14: X (15 pts) is +1 for T2, Y (14 pts) +0 for T1.
        edge = {t["team_a"]: t["edge_a"], t["team_b"]: t["edge_b"]}
        self.assertAlmostEqual(edge["T2"], 1.0)
        self.assertAlmostEqual(edge["T1"], 0.0)

    def test_waiver_add_is_a_pickup(self):
        moves = player_impact.roster_moves(2025)
        pickups = {m["name"]: m for m in moves["pickups"]}
        self.assertIn("FA Hero", pickups)
        hero = pickups["FA Hero"]
        self.assertEqual(hero["from_team"], "Waivers")
        self.assertEqual(hero["to_team"], "T3")
        self.assertAlmostEqual(hero["edge_after"], 6.0)   # 20 vs the 14 median
        # The traded players are not pickups.
        self.assertNotIn("Traded X", pickups)

    def test_page_renders(self):
        resp = self.client.get(reverse("player_impact"))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, "Star QB")
        self.assertContains(resp, "Benched WR")


class TransactionParsingTests(TestCase):
    """parse_transactions against a canned Yahoo payload shape."""

    def test_trade_and_add_are_flattened(self):
        from .yahoo_collector import parse_transactions
        txns = [
            {
                "type": "trade", "status": "successful", "timestamp": "1730000000",
                "players": {
                    "count": 2,
                    "0": {"player": [
                        [{"player_id": "100"}, {"name": {"full": "Traded X"}},
                         {"display_position": "WR"}],
                        {"transaction_data": [{
                            "type": "trade",
                            "source_type": "team", "source_team_name": "Team One",
                            "destination_type": "team", "destination_team_name": "Team Two",
                        }]},
                    ]},
                    "1": {"player": [
                        [{"player_id": "101"}, {"name": {"full": "Traded Y"}},
                         {"display_position": "RB"}],
                        {"transaction_data": {
                            "type": "trade",
                            "source_type": "team", "source_team_name": "Team Two",
                            "destination_type": "team", "destination_team_name": "Team One",
                        }},
                    ]},
                },
            },
            {
                "type": "add", "status": "successful", "timestamp": "1730000500",
                "players": {
                    "count": 1,
                    "0": {"player": [
                        [{"player_id": "102"}, {"name": {"full": "Waiver Guy"}},
                         {"display_position": "TE"}],
                        {"transaction_data": [{
                            "type": "add",
                            "source_type": "waivers",
                            "destination_type": "team",
                            "destination_team_name": "Team One",
                        }]},
                    ]},
                },
            },
        ]
        rows = parse_transactions(txns)
        self.assertEqual(len(rows), 3)
        by_id = {r["player_id"]: r for r in rows}
        x = by_id["100"]
        self.assertEqual((x["transaction_type"], x["from_team"], x["to_team"]),
                         ("TRADE", "Team One", "Team Two"))
        self.assertEqual(x["timestamp"], 1730000000)
        y = by_id["101"]
        self.assertEqual((y["transaction_type"], y["from_team"], y["to_team"]),
                         ("TRADE", "Team Two", "Team One"))
        w = by_id["102"]
        self.assertEqual((w["transaction_type"], w["from_team"], w["to_team"]),
                         ("PICKUP", "Waivers", "Team One"))


class TransactionLogTradeTests(TestCase):
    """roster_moves prefers the real transaction log when it exists."""

    @classmethod
    def setUpTestData(cls):
        from datetime import datetime, timezone as dt_tz
        from .models import Player, PlayerTransaction
        _make_week("T1", 1, 100, "T2", 90)
        _make_week("T2", 1, 90, "T1", 100)
        _make_week("T3", 1, 95, "T4", 80)
        _make_week("T4", 1, 80, "T3", 95)
        _make_week("T1", 2, 105, "T2", 95)
        _make_week("T2", 2, 95, "T1", 105)
        _make_week("T3", 2, 99, "T4", 90)
        _make_week("T4", 2, 90, "T3", 99)

        pp = PlayerImpactTests._pp
        # A real trade: X and Y swap between T1 and T2 in week 2.
        pp("Traded X", "WR", "T1", 1, 10, True)
        pp("Traded Y", "WR", "T2", 1, 7, True)
        pp("Traded X", "WR", "T2", 2, 15, True)
        pp("Traded Y", "WR", "T1", 2, 14, True)
        # Defense churn that LOOKS like a trade: two DEFs swap T3/T4 in week 2.
        pp("DEF One", "DEF", "T3", 1, 5, True)
        pp("DEF Two", "DEF", "T4", 1, 6, True)
        pp("DEF One", "DEF", "T4", 2, 4, True)
        pp("DEF Two", "DEF", "T3", 2, 8, True)

        # The transaction log records only the real trade.
        when = datetime(2025, 10, 1, tzinfo=dt_tz.utc)
        for name, src, dst in (("Traded X", "T1", "T2"), ("Traded Y", "T2", "T1")):
            PlayerTransaction.objects.create(
                player=Player.objects.get(yahoo_player_id=name),
                from_team=src, to_team=dst, transaction_type="TRADE",
                week="", year=2025, transaction_date=when)

    def test_log_separates_real_trades_from_churn(self):
        moves = player_impact.roster_moves(2025)
        self.assertTrue(moves["exact"])
        self.assertEqual(len(moves["trades"]), 1)
        t = moves["trades"][0]
        self.assertEqual({t["team_a"], t["team_b"]}, {"T1", "T2"})
        traded = {m["name"] for m in t["got_a"] + t["got_b"]}
        self.assertEqual(traded, {"Traded X", "Traded Y"})
        # The DEF churn stays in pickups instead of masquerading as a trade.
        pickup_names = {m["name"] for m in moves["pickups"]}
        self.assertEqual(pickup_names, {"DEF One", "DEF Two"})

    def test_without_log_churn_is_guessed_as_trade(self):
        from .models import PlayerTransaction
        PlayerTransaction.objects.all().delete()
        moves = player_impact.roster_moves(2025)
        self.assertFalse(moves["exact"])
        self.assertEqual(len(moves["trades"]), 2)  # heuristic pairs both swaps
