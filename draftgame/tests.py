from django.test import TestCase, Client
from django.contrib.auth.models import User

from .models import (
    Season, Punishment, PunishmentVote, PunishmentRanking,
    WheelResult, BucketWeekResult, BucketTeamResult,
)
from .views import (
    parse_team_entry, compute_bucket_standings, retired_norms_for, _norm_punishment,
)


def arch_text():
    return Punishment.objects.filter(is_winner=True, season__year=2025).first().text


class HelperTests(TestCase):
    """Pure helpers — the migration-seeded 2025/2026 data is present."""

    def test_parse_team_entry_uses_last_separator(self):
        # Team name itself contains ' - ' — player must still parse correctly.
        team, player = parse_team_entry("Thank You For Your Service (Army Hockey - Dec 13th) - Austin")
        self.assertEqual(player, "Austin")
        self.assertEqual(team, "Thank You For Your Service (Army Hockey - Dec 13th)")

    def test_parse_team_entry_no_separator(self):
        self.assertEqual(parse_team_entry("Lone")[1], "")

    def test_retired_norms_for_prior_winner(self):
        s2026 = Season.objects.get(year=2026)
        retired = retired_norms_for(s2026)
        self.assertIn(_norm_punishment(arch_text()), retired)
        # 2025 has no prior season, so nothing retired.
        self.assertEqual(retired_norms_for(Season.objects.get(year=2025)), set())


class SubmissionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('submitter', password='pw')
        self.client.force_login(self.user)
        self.season = Season.objects.get(year=2026)

    def test_submissions_unlimited(self):
        for i in range(3):
            self.client.post('/punishments/submit-punishment/', {'text': f'Punishment {i}'})
        count = Punishment.objects.filter(season=self.season, submitted_by=self.user, is_seed=False).count()
        self.assertEqual(count, 3)

    def test_submissions_close_at_deadline(self):
        from datetime import timedelta
        from django.utils import timezone
        self.season.submissions_close_at = timezone.now() - timedelta(minutes=1)
        self.season.save()
        self.client.post('/punishments/submit-punishment/', {'text': 'Too late'})
        exists = Punishment.objects.filter(season=self.season, submitted_by=self.user, is_seed=False).exists()
        self.assertFalse(exists)

    def test_retired_punishment_blocked(self):
        self.client.post('/punishments/submit-punishment/', {'text': arch_text()})
        exists = Punishment.objects.filter(
            season=self.season, submitted_by=self.user, text=arch_text()
        ).exists()
        self.assertFalse(exists)


class VotingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('voter', password='pw')
        self.client.force_login(self.user)
        self.season = Season.objects.get(year=2026)

    def test_retired_excluded_from_pool(self):
        retired = retired_norms_for(self.season)
        pool = [p for p in self.season.punishments.all()
                if _norm_punishment(p.text) not in retired]
        self.assertFalse(any('Gateway Arch' in p.text for p in pool))

    def test_vote_saves_rankings(self):
        # The ballot pool is other users' new suggestions, not seeds.
        other = User.objects.create_user('other', password='pw')
        suggestions = [
            Punishment.objects.create(season=self.season, text=f'Suggestion {i}', submitted_by=other)
            for i in range(5)
        ]
        n = min(self.season.ranking_size, len(suggestions))
        data = {f'punishment_{i+1}': str(suggestions[i].id) for i in range(n)}
        self.client.post('/punishments/punishment-vote/', data)
        vote = PunishmentVote.objects.get(user=self.user, season=self.season)
        self.assertEqual(vote.rankings.count(), n)

    def test_ballot_order_shuffled_per_voter_but_stable(self):
        import json, re
        other = User.objects.create_user('other', password='pw')
        for i in range(12):
            Punishment.objects.create(season=self.season, text=f'Idea {i}', submitted_by=other)

        def ballot_order(client):
            html = client.get('/punishments/punishment-vote/').content.decode()
            pool = json.loads(re.search(r'<script id="poolData"[^>]*>(.*?)</script>', html, re.S).group(1))
            return [p['id'] for p in pool]

        first = ballot_order(self.client)
        self.assertEqual(first, ballot_order(self.client), 'same voter must see a stable order')

        c2 = Client()
        c2.force_login(User.objects.create_user('voter2', password='pw'))
        self.assertEqual(sorted(first), sorted(ballot_order(c2)), 'same candidates for everyone')
        self.assertNotEqual(first, ballot_order(c2), 'different voters should see different orders')

    def test_own_submission_banner(self):
        other = User.objects.create_user('other', password='pw')
        Punishment.objects.create(season=self.season, text='Someone elses idea', submitted_by=other)
        # No submissions: no banner.
        html = self.client.get('/punishments/punishment-vote/').content.decode()
        self.assertNotIn('Heads up', html)
        # With two submissions: banner with the right count.
        for i in range(2):
            Punishment.objects.create(season=self.season, text=f'My idea {i}', submitted_by=self.user)
        html = self.client.get('/punishments/punishment-vote/').content.decode()
        self.assertIn('Heads up', html)
        self.assertIn('your 2 submissions', html)

    def test_seeds_rankable_but_own_submissions_rejected(self):
        # Full grid reset: carryover seeds are on the ballot, own ideas are not.
        own = Punishment.objects.create(season=self.season, text='My own idea', submitted_by=self.user)
        seed = self.season.punishments.filter(is_seed=True).first()
        self.client.post('/punishments/punishment-vote/', {
            'punishment_1': str(own.id),
            'punishment_2': str(seed.id),
        })
        vote = PunishmentVote.objects.get(user=self.user, season=self.season)
        ranked_ids = set(vote.rankings.values_list('punishment_id', flat=True))
        self.assertEqual(ranked_ids, {seed.id})

    def test_finalize_takes_top_wheel_size_by_points(self):
        from .views import finalize_voting
        # Full grid reset: every voted punishment (seed or new) competes for
        # the 12 slots on points alone.
        retired = retired_norms_for(self.season)
        seeds = [p for p in self.season.punishments.filter(is_seed=True)
                 if _norm_punishment(p.text) not in retired]
        other = User.objects.create_user('other', password='pw')
        news = [Punishment.objects.create(season=self.season, text=f'New idea {i}', submitted_by=other)
                for i in range(4)]

        # Rank 12 candidates: 8 seeds and 4 new ones.
        picks = seeds[:8] + news
        vote = PunishmentVote.objects.create(user=self.user, season=self.season)
        for rank, p in enumerate(picks, start=1):
            PunishmentRanking.objects.create(vote=vote, punishment=p, rank=rank)

        n = finalize_voting(self.season)
        finalists = set(self.season.punishments.filter(is_finalist=True).values_list('id', flat=True))
        self.assertEqual(n, 12)
        self.assertEqual(finalists, {p.id for p in picks})

    def test_grid_not_seeded_before_any_votes(self):
        from .views import ensure_finalists
        # Voting closed, no ballots yet: the grid must stay unlocked rather
        # than snapshotting an arbitrary zero-point "top 12".
        self.season.voting_open = False
        self.season.save()
        self.season.punishments.update(is_finalist=False)
        ensure_finalists(self.season)
        self.assertFalse(self.season.punishments.filter(is_finalist=True).exists())


class SuggestionListTests(TestCase):
    def test_public_no_login_needed(self):
        resp = self.client.get('/punishments/suggested-so-far/')
        self.assertEqual(resp.status_code, 200)

    def test_lists_all_suggestions_without_names(self):
        season = Season.objects.get(year=2026)
        a = User.objects.create_user('alphonse', password='pw')
        b = User.objects.create_user('bartholomew', password='pw')
        Punishment.objects.create(season=season, text='First idea', submitted_by=a)
        Punishment.objects.create(season=season, text='Second idea', submitted_by=b)
        # Anonymous visitor: sees every suggestion, never a submitter name.
        html = self.client.get('/punishments/suggested-so-far/').content.decode()
        self.assertNotIn('alphonse', html)
        self.assertIn('First idea', html)
        self.assertIn('Second idea', html)
        self.assertNotIn('bartholomew', html)  # other submitters stay anonymous


class PendingDuplicateTests(TestCase):
    def setUp(self):
        from datetime import timedelta
        from django.utils import timezone
        self.season = Season.objects.get(year=2026)
        self.user = User.objects.create_user('submitter', password='pw')
        self.p = Punishment.objects.create(
            season=self.season, text='Copycat idea', submitted_by=self.user
        )
        self.past = timezone.now() - timedelta(minutes=5)
        self.future = timezone.now() + timedelta(hours=5)

    def _apply(self, pending=None):
        from .views import apply_pending_duplicates
        apply_pending_duplicates(self.season, pending=pending or [(self.p.id, self.p.text)])
        self.p.refresh_from_db()

    def test_not_applied_before_deadline(self):
        self.season.submissions_close_at = self.future
        self.season.save()
        self._apply()
        self.assertFalse(self.p.is_duplicate)

    def test_applied_after_deadline(self):
        self.season.submissions_close_at = self.past
        self.season.save()
        self._apply()
        self.assertTrue(self.p.is_duplicate)

    def test_skipped_if_text_changed(self):
        self.season.submissions_close_at = self.past
        self.season.save()
        self._apply(pending=[(self.p.id, 'The old wording that was replaced')])
        self.assertFalse(self.p.is_duplicate)

    def test_duplicates_off_ballot_and_out_of_finalize(self):
        from .views import finalize_voting
        self.season.submissions_close_at = self.past
        self.season.voting_open = True
        self.season.save()
        other = User.objects.create_user('other', password='pw')
        keeper_idea = Punishment.objects.create(season=self.season, text='Original idea', submitted_by=other)
        self.p.is_duplicate = True
        self.p.save()

        voter = User.objects.create_user('voter', password='pw')
        self.client.force_login(voter)
        html = self.client.get('/punishments/punishment-vote/').content.decode()
        self.assertIn('Original idea', html)
        self.assertNotIn('Copycat idea', html)

        finalize_voting(self.season)
        self.p.refresh_from_db()
        self.assertFalse(self.p.is_finalist)

    def test_shown_consolidated_on_public_list(self):
        self.p.is_duplicate = True
        self.p.save()
        html = self.client.get('/punishments/suggested-so-far/').content.decode()
        self.assertIn('Consolidated duplicates', html)
        self.assertIn('Copycat idea', html)


class LiveResultsTests(TestCase):
    def setUp(self):
        self.season = Season.objects.get(year=2026)
        self.season.voting_open = True
        self.season.save()
        self.staff = User.objects.create_user('boss', password='pw', is_staff=True)
        other = User.objects.create_user('ideaguy', password='pw')
        self.candidate = Punishment.objects.create(
            season=self.season, text='A fresh idea', submitted_by=other
        )

    def test_non_staff_blocked(self):
        member = User.objects.create_user('member', password='pw')
        self.client.force_login(member)
        self.assertEqual(self.client.get('/punishments/punishment-results/').status_code, 302)

    def test_live_tally_counts_only_saved_ballots(self):
        # A voter who saves rankings counts; one who only opened the ballot doesn't.
        voter = User.objects.create_user('voter', password='pw')
        lurker = User.objects.create_user('lurker', password='pw')
        vote = PunishmentVote.objects.create(user=voter, season=self.season)
        PunishmentRanking.objects.create(vote=vote, punishment=self.candidate, rank=1)
        PunishmentVote.objects.create(user=lurker, season=self.season)  # empty ballot

        self.client.force_login(self.staff)
        resp = self.client.get('/punishments/punishment-results/')
        html = resp.content.decode()
        self.assertIn('Voted so far (1)', html)
        self.assertIn('voter', html)
        self.assertNotIn('lurker', html)
        self.assertIn('A fresh idea', html)  # tally row with points


class RegisterTests(TestCase):
    def test_register_creates_and_logs_in(self):
        resp = self.client.post('/accounts/register/', {
            'username': 'newmember',
            'password1': 'a-strong-pass-9182',
            'password2': 'a-strong-pass-9182',
        })
        self.assertEqual(resp.status_code, 302)
        self.assertTrue(User.objects.filter(username='newmember').exists())
        # The new user should be logged in right away.
        resp = self.client.get('/punishments/')
        self.assertContains(resp, 'newmember')


class VotingCompleteNavTests(TestCase):
    """Once voting closes and the grid is locked, the Vote tab disappears."""

    def test_vote_tab_hidden_after_grid_locked(self):
        season = Season.get_active()
        season.submissions_open = False
        season.voting_open = False
        season.save()
        season.punishments.filter(is_seed=True).update(is_finalist=True)
        html = self.client.get('/punishments/').content.decode()
        self.assertNotIn('>Vote</a>', html)
        self.assertIn('locked', html)

    def test_vote_tab_shown_while_voting_open(self):
        season = Season.get_active()
        season.voting_open = True
        season.save()
        html = self.client.get('/punishments/').content.decode()
        self.assertIn('>Vote</a>', html)


class BucketGateTests(TestCase):
    def test_non_staff_redirected_home(self):
        user = User.objects.create_user('member', password='pw')
        self.client.force_login(user)
        resp = self.client.get('/punishments/bucket-of-death/')
        self.assertEqual(resp.status_code, 302)
        self.assertEqual(resp.url, '/punishments/')

    def test_staff_can_access(self):
        staff = User.objects.create_user('gamemaster', password='pw', is_staff=True)
        self.client.force_login(staff)
        resp = self.client.get('/punishments/bucket-of-death/')
        self.assertEqual(resp.status_code, 200)

    def test_everyone_can_access_once_bucket_open(self):
        season = Season.get_active()
        season.bucket_open = True
        season.save()
        user = User.objects.create_user('member', password='pw')
        self.client.force_login(user)
        resp = self.client.get('/punishments/bucket-of-death/')
        self.assertEqual(resp.status_code, 200)


class GridTests(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('boss', password='pw', is_staff=True)
        self.s2025 = Season.objects.get(year=2025)
        # Lock in finalists for 2025 including the winner (Arch).
        winner = self.s2025.punishments.filter(is_winner=True).first()
        others = list(self.s2025.punishments.exclude(id=winner.id)
                      .order_by('id').values_list('id', flat=True)[:11])
        Punishment.objects.filter(id__in=others + [winner.id]).update(is_finalist=True)

    def _grid(self, year, client):
        client.get(f'/punishments/punishment-history/?year={year}')
        return client.get('/punishments/punishment-grid/').content.decode()

    def test_finalized_grid_shows_winner(self):
        c = Client(); c.force_login(self.staff)
        body = self._grid(2025, c)
        self.assertIn('Gateway Arch', body)
        self.assertIn('ultimately had to complete', body)

    def test_unfinalized_grid_is_empty(self):
        c = Client(); c.force_login(self.staff)
        self.assertIn("haven't been locked in", self._grid(2026, c))

    def test_admin_controls_hidden_from_anonymous(self):
        anon = Client()
        body = anon.get('/punishments/punishment-grid/').content.decode()
        self.assertNotIn('Finalize assignments', body)
        self.assertNotIn('Assigned to', body)


class FinalizeVotingTests(TestCase):
    def setUp(self):
        self.season = Season.objects.get(year=2026)

    def test_finalize_excludes_retired(self):
        from .views import finalize_voting, season_finalists
        finalize_voting(self.season)
        texts = [p.text for p in season_finalists(self.season)]
        self.assertFalse(any('Gateway Arch' in t for t in texts))

    def test_finalize_respects_wheel_size(self):
        from .views import finalize_voting
        self.season.wheel_size = 5
        self.season.save()
        finalize_voting(self.season)
        self.assertEqual(self.season.punishments.filter(is_finalist=True).count(), 5)


class WheelLockTests(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('boss2', password='pw', is_staff=True)
        self.client.force_login(self.staff)

    def test_wheel_locked_when_grid_finalized(self):
        # 2025 grid is finalized by migration; selecting it locks the wheel.
        self.client.get('/punishments/punishment-history/?year=2025')
        body = self.client.get('/punishments/punishment-wheel/').content.decode()
        self.assertIn('Wheel locked', body)
        before = WheelResult.objects.filter(season__year=2025).count()
        self.client.post('/punishments/punishment-wheel/', {'choose_punishment': 'X', 'player_name': 'Y'})
        self.assertEqual(WheelResult.objects.filter(season__year=2025).count(), before)

    def test_wheel_blocked_while_voting_open(self):
        # 2026 grid not locked and voting still open -> wheel shows the prompt.
        self.client.get('/punishments/punishment-history/?year=2026')
        body = self.client.get('/punishments/punishment-wheel/').content.decode()
        self.assertNotIn('Wheel locked', body)
        self.assertIn('Voting is still open', body)

    def test_closing_voting_auto_seeds_wheel(self):
        # Closing voting via the toggle seeds finalists and makes the wheel runnable.
        self.client.get('/punishments/punishment-history/?year=2026')
        self.client.post('/punishments/toggle-voting/')
        s2026 = Season.objects.get(year=2026)
        self.assertFalse(s2026.voting_open)
        self.assertTrue(s2026.punishments.filter(is_finalist=True).exists())
        body = self.client.get('/punishments/punishment-wheel/').content.decode()
        self.assertNotIn('Voting is still open', body)
        self.assertNotIn('Wheel locked', body)

    def test_grid_auto_seeds_when_voting_closed(self):
        # Even if voting is closed directly (e.g. admin), the grid auto-seeds
        # from the results, provided at least one ballot was cast.
        s2026 = Season.objects.get(year=2026)
        s2026.voting_open = False
        s2026.save()
        voter = User.objects.create_user('gridvoter', password='pw')
        vote = PunishmentVote.objects.create(user=voter, season=s2026)
        PunishmentRanking.objects.create(
            vote=vote, punishment=s2026.punishments.filter(is_seed=True).first(), rank=1
        )
        self.client.get('/punishments/punishment-history/?year=2026')
        self.client.get('/punishments/punishment-grid/')  # triggers ensure_finalists
        self.assertTrue(s2026.punishments.filter(is_finalist=True).exists())


class BucketStandingsTests(TestCase):
    def setUp(self):
        self.season = Season.objects.get(year=2025)
        wk = BucketWeekResult.objects.create(season=self.season, week_number=1, punishment='x')
        # Big winner with more games, and a 1-0 small sample.
        for _ in range(10):
            BucketTeamResult.objects.create(week_result=wk, team_name='Bears', player_name='Big', is_win=True)
        for _ in range(5):
            BucketTeamResult.objects.create(week_result=wk, team_name='Bears', player_name='Big', is_win=False)
        BucketTeamResult.objects.create(week_result=wk, team_name='Jets', player_name='Tiny', is_win=True)
        # Special-team variants that should group.
        BucketTeamResult.objects.create(week_result=wk, team_name='Any Underdog (Jets)', player_name='Big', is_win=True)
        BucketTeamResult.objects.create(week_result=wk, team_name='Any Underdog (Rams)', player_name='Big', is_win=False)

    def test_more_wins_outranks_small_perfect_sample(self):
        lb, _ = compute_bucket_standings(self.season)
        names = [r['name'] for r in lb]
        self.assertLess(names.index('Big'), names.index('Tiny'))  # 10-5+ above 1-0

    def test_special_teams_grouped(self):
        _, tallies = compute_bucket_standings(self.season)
        underdog = [t for t in tallies if t['team'] == 'Any Underdog']
        self.assertEqual(len(underdog), 1)
        self.assertEqual(underdog[0]['total'], 2)


class HistoryVisibilityTests(TestCase):
    def setUp(self):
        self.client_ = Client()

    def test_history_redirects_for_unfinished_season(self):
        self.client_.get('/punishments/punishment-history/?year=2026')  # 2026 has no winner
        resp = self.client_.get('/punishments/punishment-history/')
        self.assertEqual(resp.status_code, 302)
        self.assertIn('/punishments/punishment-grid/', resp.url)

    def test_history_renders_for_finished_season(self):
        self.client_.get('/punishments/punishment-history/?year=2025')
        resp = self.client_.get('/punishments/punishment-history/')
        self.assertEqual(resp.status_code, 200)

    def test_history_tab_hidden_for_unfinished(self):
        c = Client()
        c.get('/punishments/punishment-history/?year=2026')
        grid = c.get('/punishments/punishment-grid/').content.decode()
        self.assertNotIn('>\n        History', grid.replace('\r', ''))
        self.assertNotIn('ti-history', grid)

    def test_history_tab_shown_for_finished(self):
        c = Client()
        c.get('/punishments/punishment-history/?year=2025')
        grid = c.get('/punishments/punishment-grid/').content.decode()
        self.assertIn('ti-history', grid)


class YearSelectorTests(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('boss3', password='pw', is_staff=True)
        self.client.force_login(self.staff)

    def test_defaults_to_latest_and_persists(self):
        # No ?year -> most recent (2026).
        results = self.client.get('/punishments/punishment-results/').content.decode()
        self.assertIn('2026 voting results', results)
        # Pick 2025 on one page -> persists to another via session.
        self.client.get('/punishments/punishment-grid/?year=2025')
        results = self.client.get('/punishments/punishment-results/').content.decode()
        self.assertIn('2025 voting results', results)
