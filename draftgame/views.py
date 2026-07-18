from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from .models import Season, Punishment, PunishmentVote, PunishmentRanking, WheelResult, BucketSession, BucketWeekResult, BucketTeamResult
from django.contrib.auth.decorators import user_passes_test
import random
import json
from django.db.models import Avg
from urllib.parse import quote
from .context_processors import get_selected_season

def home(request):
    return render(request, 'draftgame/home.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Two auth backends are configured (password + Google), so login()
            # can't infer which one authenticated this brand-new user.
            login(request, user, backend='django.contrib.auth.backends.ModelBackend')
            messages.success(request, 'Account created successfully!')
            return redirect('draftgame:home')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def submit_punishment(request):
    season = Season.get_active()
    if season is None:
        messages.error(request, 'There is no active season right now.')
        return redirect('draftgame:home')

    my_submissions = Punishment.objects.filter(
        season=season, submitted_by=request.user, is_seed=False
    ).order_by('created_at')

    if request.method == 'POST':
        # Handle deletion of one of the user's own submissions.
        delete_id = request.POST.get('delete_id')
        if delete_id:
            target = my_submissions.filter(id=delete_id).first()
            if target is None:
                messages.error(request, 'That submission was not found.')
            elif not season.submissions_are_open:
                messages.error(request, 'Submissions are closed, so it can no longer be removed.')
            else:
                target.delete()
                messages.success(request, 'Your submission was removed.')
            return redirect('draftgame:submit_punishment')

        # Otherwise treat it as a new submission.
        text = (request.POST.get('text') or '').strip()
        if not season.submissions_are_open:
            messages.error(request, 'Submissions are closed for this season.')
        elif not text:
            messages.error(request, 'Please enter a punishment before submitting.')
        elif _norm_punishment(text) in retired_norms_for(season):
            messages.error(request, 'That punishment won a previous season and has been retired. Pick something new.')
        else:
            Punishment.objects.create(
                season=season, text=text, submitted_by=request.user, is_seed=False
            )
            messages.success(request, 'Your punishment has been submitted!')
        return redirect('draftgame:submit_punishment')

    return render(request, 'draftgame/submit_punishment.html', {
        'season': season,
        'my_submissions': my_submissions,
    })


# Duplicate rulings agreed with the submitters ahead of the deadline. Each is
# matched by BOTH id and exact text, so an entry the submitter deleted or
# swapped before the deadline is left alone. Applied automatically (see
# apply_pending_duplicates) once submissions close.
PENDING_DUPLICATES = [
    (63, 'Some sort of controversial (mildly) sticker on their car (I.e. flat earth society)'),
    (67, 'Take an inflatable doll on a proper, fully dressed, dinner date.'),
    (74, 'Make $20 in tips at a karaoke bar in one night'),
]


def apply_pending_duplicates(season, pending=None):
    """Mark agreed duplicates once the submission deadline has passed.

    Safe to call on every read, mirroring ensure_finalists: it only acts
    after submissions_close_at, and only on entries still matching the
    (id, text) pair recorded when the ruling was made.
    """
    from django.utils import timezone
    if season is None or not season.submissions_close_at:
        return
    if timezone.now() < season.submissions_close_at:
        return
    for pid, text in (PENDING_DUPLICATES if pending is None else pending):
        Punishment.objects.filter(
            id=pid, season=season, is_seed=False, text=text, is_duplicate=False
        ).update(is_duplicate=True)


def suggestion_list(request):
    """Everything submitted for the active season, anonymously (texts only).
    Public on purpose: no names are shown, and the league wants to browse
    without logging in. Submitting and voting stay login-gated."""
    season = Season.get_active()
    if season is None:
        messages.error(request, 'There is no active season right now.')
        return redirect('draftgame:home')
    apply_pending_duplicates(season)
    submissions = season.punishments.filter(is_seed=False).order_by('-created_at')
    return render(request, 'draftgame/suggestion_list.html', {
        'season': season,
        'all_suggestions': [p.text for p in submissions if not p.is_duplicate],
        'duplicates': [p.text for p in submissions if p.is_duplicate],
    })


@login_required
def punishment_vote(request):
    season = Season.get_active()
    if season is None:
        messages.error(request, 'There is no active season right now.')
        return redirect('draftgame:home')
    if not season.voting_open:
        messages.info(request, 'Voting is not open yet for this season.')
        return redirect('draftgame:home')

    # Ballot pool: the full grid reset — carryover punishments and new
    # suggestions compete equally. Excluded: consolidated duplicates,
    # retired past winners, and the voter's own submissions (carryovers
    # have no submitter, so everyone can rank those).
    apply_pending_duplicates(season)
    retired = retired_norms_for(season)
    pool = [
        p for p in season.punishments.filter(is_duplicate=False).order_by('id')
        if _norm_punishment(p.text) not in retired
        and p.submitted_by_id != request.user.id
    ]
    # Each voter sees the pool in their own random order so no submission
    # benefits from sitting at the top for everyone; seeding by user keeps
    # the order stable every time the same voter reopens their ballot.
    random.Random(f'{season.year}:{request.user.id}').shuffle(pool)
    ranking_size = min(season.ranking_size, len(pool))
    ranks = list(range(1, ranking_size + 1))

    vote, _ = PunishmentVote.objects.get_or_create(user=request.user, season=season)
    existing_rankings = {r.rank: r.punishment_id for r in vote.rankings.all()}

    if request.method == 'POST':
        valid_ids = {p.id for p in pool}
        vote.rankings.all().delete()
        seen = set()
        for rank in ranks:
            raw = request.POST.get(f'punishment_{rank}')
            if not raw:
                continue
            try:
                pid = int(raw)
            except ValueError:
                continue
            if pid in valid_ids and pid not in seen:
                seen.add(pid)
                PunishmentRanking.objects.create(vote=vote, punishment_id=pid, rank=rank)
        messages.success(request, 'Your rankings have been saved!')
        return redirect('draftgame:punishment_vote')

    pool_json = [{'id': p.id, 'text': p.text} for p in pool]
    existing_order = [existing_rankings[r] for r in ranks if r in existing_rankings]
    my_submission_count = season.punishments.filter(
        is_seed=False, submitted_by=request.user
    ).count()
    return render(request, 'draftgame/punishment_vote.html', {
        'my_submission_count': my_submission_count,
        'season': season,
        'punishments': pool,
        'pool_json': pool_json,
        'existing_order': existing_order,
        'ranks': ranks,
        'ranking_size': ranking_size,
        'existing_rankings': existing_rankings,
    })


def punishment_results(request):
    if not request.user.is_staff:
        messages.error(request, 'Admin access required')
        return redirect('draftgame:home')

    season, all_seasons = get_selected_season(request)
    if season is None:
        messages.error(request, 'There are no seasons yet.')
        return redirect('draftgame:home')

    results = []
    for punishment in season.punishments.all():
        rankings = punishment.rankings.all()
        if rankings.exists():
            ranking_size = season.ranking_size
            total_points = sum((ranking_size + 1) - r.rank for r in rankings)
            results.append({
                'punishment': punishment.text,
                'total_points': total_points,
                'vote_count': rankings.count(),
                'rankings': list(rankings.values('rank', 'vote__user__username')),
            })

    results.sort(key=lambda x: x['total_points'], reverse=True)
    # Only ballots with actual rankings count as votes; opening the ballot
    # page creates an empty PunishmentVote row that shouldn't inflate this.
    voted = (
        PunishmentVote.objects.filter(season=season, rankings__isnull=False)
        .distinct().select_related('user').order_by('updated_at')
    )
    finalist_count = season.punishments.filter(is_finalist=True).count()
    return render(request, 'draftgame/punishment_results.html', {
        'season': season,
        'all_seasons': all_seasons,
        'results': results,
        'voter_count': voted.count(),
        'voted_users': [v.user.username for v in voted],
        'voting_finalized': finalist_count > 0,
        'finalist_count': finalist_count,
    })


def punishment_history(request):
    season, all_seasons = get_selected_season(request)
    if season is None:
        return render(request, 'draftgame/punishment_history.html', {'season': None})

    # The single punishment that was ultimately enforced, and its vote stats.
    winner_p = season.punishments.filter(is_winner=True).first()

    # No outcome yet -> the season hasn't ended, so there's no history to show.
    if winner_p is None:
        return redirect('draftgame:punishment_grid')
    winner_points = winner_votes = 0
    if winner_p:
        rankings = winner_p.rankings.all()
        winner_votes = rankings.count()
        winner_points = sum((season.ranking_size + 1) - r.rank for r in rankings)

    # Others who drew the same (enforced) punishment on the wheel, besides the loser.
    also_assigned = []
    if winner_p:
        wn = _norm_punishment(winner_p.text)
        names = [w.name for w in WheelResult.objects.filter(season=season)
                 if _norm_punishment(w.punishment) == wn]
        also_assigned = [n for n in names if n != season.loser_name]

    return render(request, 'draftgame/punishment_history.html', {
        'season': season,
        'winner': winner_p.text if winner_p else None,
        'winner_points': winner_points,
        'winner_votes': winner_votes,
        'loser': season.loser_name,
        'voter_count': PunishmentVote.objects.filter(season=season).count(),
        'also_assigned': also_assigned,
    })

def _norm_punishment(text):
    """Normalize punishment text for matching across the hardcoded list and DB rows."""
    return ' '.join((text or '').split()).strip().lower()


def retired_norms_for(season):
    """Normalized texts of punishments retired by winning a PRIOR season."""
    if not season:
        return set()
    prior_winners = Punishment.objects.filter(is_winner=True, season__year__lt=season.year)
    return {_norm_punishment(p.text) for p in prior_winners}


def season_finalists(season):
    """Punishments locked in (via Finalize voting) as the season's wheel/grid set."""
    if not season:
        return []
    return list(season.punishments.filter(is_finalist=True).order_by('id'))


def finalize_voting(season):
    """Snapshot the top `wheel_size` voted punishments as finalists and
    return the count. Full grid reset: carryovers and new submissions
    compete equally; retired winners and consolidated duplicates sit out.
    """
    retired = retired_norms_for(season)
    scored = []
    for p in season.punishments.filter(is_duplicate=False):
        if _norm_punishment(p.text) in retired:
            continue
        rankings = p.rankings.all()
        points = sum((season.ranking_size + 1) - r.rank for r in rankings) if rankings.exists() else 0
        scored.append((points, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_ids = [p.id for _, p in scored[:season.wheel_size]]
    season.punishments.update(is_finalist=False)
    Punishment.objects.filter(id__in=top_ids).update(is_finalist=True)
    return len(top_ids)


def ensure_finalists(season):
    """Auto-seed the wheel set from the vote once voting is closed.

    Keeps the wheel punishments in sync without a manual step: as soon as voting
    is closed for a season, the top votes become the finalists. Safe to call on
    every read; it only seeds when voting is closed, nothing is locked in yet,
    and at least one ballot exists (otherwise the "top" would be arbitrary).
    """
    if (
        season and not season.voting_open
        and not season.punishments.filter(is_finalist=True).exists()
        and PunishmentRanking.objects.filter(vote__season=season).exists()
    ):
        finalize_voting(season)


def punishment_grid(request):
    season, _ = get_selected_season(request)
    ensure_finalists(season)

    # The season's locked-in set (snapshot of top votes). Retired punishments
    # are already excluded at finalize time.
    finalists = season_finalists(season)

    this_winner = season.punishments.filter(is_winner=True).first() if season else None
    winner_norm = _norm_punishment(this_winner.text) if this_winner else None
    champion = season.loser_name if season else ''

    # Wheel assignments for the selected season, keyed by normalized punishment text.
    wheel_assignments = {}
    wheel_qs = WheelResult.objects.filter(season=season) if season else WheelResult.objects.none()
    for result in wheel_qs:
        wheel_assignments.setdefault(_norm_punishment(result.punishment), []).append(result.name)

    punishments_with_names = []
    for p in finalists:
        norm = _norm_punishment(p.text)
        is_winner = norm == winner_norm
        punishments_with_names.append({
            'punishment': p.text,
            'names': wheel_assignments.get(norm, []),
            'is_winner': is_winner,
            'winner_name': champion if is_winner else '',
        })

    return render(request, 'draftgame/punishment_grid.html', {
        'punishments_with_names': punishments_with_names,
        'is_admin': request.user.is_staff,
        'grid_season': season,
        'voting_finalized': bool(finalists),
    })


@user_passes_test(lambda u: u.is_staff)
def toggle_grid_finalized(request):
    season, _ = get_selected_season(request)
    if season and request.method == 'POST':
        season.grid_finalized = not season.grid_finalized
        season.save()
        if season.grid_finalized:
            messages.success(request, f'{season.year} grid finalized. The wheel is now locked.')
        else:
            messages.success(request, f'{season.year} grid unlocked. The wheel can be run again.')
    return redirect('draftgame:punishment_grid')


@user_passes_test(lambda u: u.is_staff)
def toggle_voting(request):
    """Open/close voting. Closing auto-seeds the wheel from the results;
    reopening clears the locked-in set so a fresh close re-seeds."""
    season, _ = get_selected_season(request)
    if season and request.method == 'POST':
        apply_pending_duplicates(season)
        season.voting_open = not season.voting_open
        season.save()
        if season.voting_open:
            season.punishments.update(is_finalist=False)
            messages.success(request, f'{season.year} voting reopened.')
        else:
            n = finalize_voting(season)
            messages.success(request, f'{season.year} voting closed. Top {n} locked in as the wheel set.')
    return redirect('draftgame:punishment_results')


BUCKET_PUNISHMENTS = [
    "Jäger Bomb (Jägermeister + Red Bull)",
]

NFL_TEAMS = [
    "Arizona Cardinals", "Atlanta Falcons", "Baltimore Ravens", "Buffalo Bills",
    "Carolina Panthers", "Chicago Bears", "Cincinnati Bengals", "Cleveland Browns",
    "Dallas Cowboys", "Denver Broncos", "Detroit Lions", "Green Bay Packers",
    "Houston Texans", "Indianapolis Colts", "Jacksonville Jaguars", "Kansas City Chiefs",
    "Las Vegas Raiders", "Los Angeles Chargers", "Los Angeles Rams", "Miami Dolphins",
    "Minnesota Vikings", "New England Patriots", "New Orleans Saints", "New York Giants",
    "New York Jets", "Philadelphia Eagles", "Pittsburgh Steelers", "San Francisco 49ers",
    "Seattle Seahawks", "Tampa Bay Buccaneers", "Tennessee Titans", "Washington Commanders",
    "The Big Apple", "The Wizard of Oz (Lions, Tigers, Bears)", "Instant Death", "Thank You For Your Service",
    "Golden Helmet of Life", "Any Underdog"
]

SPECIAL_TEAMS = ["The Big Apple", "The Wizard of Oz (Lions, Tigers, Bears)", "Thank You For Your Service", "Any Underdog"]

def get_base_team_name(team_name):
    """Extract base team name, handling special teams with (TBD) or custom selections"""
    for special_team in SPECIAL_TEAMS:
        if team_name.startswith(special_team):
            return special_team
    return team_name


def parse_team_entry(entry):
    """Split a 'Team Name - Player' bucket entry into (team, player).

    Splits on the LAST ' - ' so a team name containing ' - ' (e.g. a custom
    special-team selection like 'Thank You For Your Service (Army Hockey - Dec 13th)')
    doesn't bleed into the player name.
    """
    parts = entry.rsplit(' - ', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return entry, ''


def compute_bucket_standings(season, session=None):
    """Build (player leaderboard, team tallies) for a season's Bucket of Death.

    Both are sorted by wins, then win %, then fewer losses. An in-progress
    ``session``'s kept teams are folded into the leaderboard as wins (matching
    the historical behavior), but not into the team tallies.
    """
    player_stats = {}
    team_tallies = {}
    for tr in BucketTeamResult.objects.filter(week_result__season=season):
        # Group special-team variants (e.g. "Thank You For Your Service (Navy)")
        # under their base team name in the tallies.
        team_key = get_base_team_name(tr.team_name)
        ps = player_stats.setdefault(tr.player_name, {'wins': 0, 'losses': 0})
        ts = team_tallies.setdefault(team_key, {'wins': 0, 'losses': 0, 'total': 0})
        ts['total'] += 1
        if tr.is_win is True:
            ps['wins'] += 1
            ts['wins'] += 1
        elif tr.is_win is False:
            ps['losses'] += 1
            ts['losses'] += 1

    if session:
        for entry in session.kept_teams:
            person = parse_team_entry(entry)[1] or entry
            player_stats.setdefault(person, {'wins': 0, 'losses': 0})['wins'] += 1

    leaderboard = []
    for name, st in player_stats.items():
        games = st['wins'] + st['losses']
        leaderboard.append({
            'name': name, 'wins': st['wins'], 'losses': st['losses'],
            'win_pct': st['wins'] / games if games else 0,
        })
    leaderboard.sort(key=lambda x: (x['wins'], x['win_pct'], -x['losses']), reverse=True)

    tallies = []
    for name, st in team_tallies.items():
        tallies.append({
            'team': name, 'wins': st['wins'], 'losses': st['losses'], 'total': st['total'],
            'win_pct': st['wins'] / st['total'] if st['total'] else 0,
        })
    tallies.sort(key=lambda x: (x['wins'], x['win_pct'], -x['losses']), reverse=True)
    return leaderboard, tallies

def bucket_of_death(request):
    # Closed until the admin flips Season.bucket_open; staff can always get in
    # to set things up.
    active = Season.get_active()
    if not request.user.is_staff and not (active and active.bucket_open):
        messages.info(request, 'Bucket of Death opens when the season starts.')
        return redirect('draftgame:home')
    season, _ = get_selected_season(request)

    if not request.session.session_key:
        request.session.create()

    # Determine the week. An explicit ?week= wins; otherwise remember the last
    # week within a season, but reset to week 1 whenever the season changes
    # (a new season starts fresh at week 1).
    season_changed = season is not None and request.session.get('bucket_season') != season.year
    if request.GET.get('week'):
        current_week = int(request.GET.get('week'))
    elif season_changed:
        current_week = 1
    else:
        current_week = request.session.get('bucket_week', 1)
    request.session['bucket_week'] = current_week
    if season is not None:
        request.session['bucket_season'] = season.year
    
    # Only create session for admins
    session = None
    if request.user.is_staff:
        if not request.session.session_key:
            request.session.create()
        try:
            session = BucketSession.objects.get(session_key=request.session.session_key, season=season)
        except BucketSession.DoesNotExist:
            session = BucketSession.objects.create(
                session_key=request.session.session_key,
                season=season,
                punishment=random.choice(BUCKET_PUNISHMENTS),
                week_number=current_week,
                is_test_mode=False
            )
        # Update week if it changed
        if session.week_number != current_week:
            session.week_number = current_week
            session.save()
    
    if request.method == 'POST' and request.user.is_staff and session and request.POST.get('action') != 'end_bucket':
        action = request.POST.get('action')
        
        if action == 'start_player':
            player_name = request.POST.get('player_name')
            week = int(request.POST.get('week', current_week))
            request.session['bucket_week'] = week
            session.current_player = player_name
            session.week_number = week
            session.save()
            return redirect('draftgame:bucket_rummage')
        elif action == 'reset':
            session.delete()
            return redirect(f'/punishments/bucket-of-death/?week={current_week}')
        elif action == 'clear_teams':
            session.kept_teams = []
            session.save()
            messages.success(request, 'Teams cleared!')
            return redirect(f'/punishments/bucket-of-death/?week={current_week}')
        elif action.startswith('mark_') and request.user.is_staff:
            parts = action.split('_')
            if len(parts) == 4:  # mark_session_team_win/loss format
                team_index = int(parts[2])
                result_type = parts[3]  # 'win' or 'loss'
                # Store W/L in session for current teams
                if 'team_wl_status' not in request.session:
                    request.session['team_wl_status'] = {}
                request.session['team_wl_status'][str(team_index)] = result_type == 'win'
                request.session.modified = True
                messages.success(request, f'Marked as {result_type}!')
            else:  # existing format for completed weeks
                team_result_id = parts[1]
                result_type = parts[2]  # 'win' or 'loss'
                try:
                    team_result = BucketTeamResult.objects.get(id=team_result_id)
                    team_result.is_win = result_type == 'win'
                    team_result.save()
                    messages.success(request, f'Marked as {result_type}!')
                except BucketTeamResult.DoesNotExist:
                    pass
            return redirect(f'/punishments/bucket-of-death/?week={current_week}')
        elif action.startswith('delete_') and request.user.is_staff:
            parts = action.split('_')
            if len(parts) == 3 and parts[1] == 'session':  # delete_session_index format
                team_index = int(parts[2])
                if 0 <= team_index < len(session.kept_teams):
                    del session.kept_teams[team_index]
                    session.save()
                    messages.success(request, 'Team deleted!')
            elif len(parts) == 2:  # delete_id format for completed teams
                team_result_id = parts[1]
                try:
                    team_result = BucketTeamResult.objects.get(id=team_result_id)
                    team_result.delete()
                    messages.success(request, 'Team deleted!')
                except BucketTeamResult.DoesNotExist:
                    pass
            return redirect(f'/punishments/bucket-of-death/?week={current_week}')
        elif action == 'finalize_tbd':
            # Update TBD teams with final names
            tbd_count = 0
            for i, team_entry in enumerate(session.kept_teams):
                if '(TBD)' in team_entry:
                    final_name = request.POST.get(f'tbd_team_{tbd_count}', '').strip()
                    if final_name:
                        base_team = team_entry.split(' (TBD)')[0]
                        player_name = parse_team_entry(team_entry)[1]
                        session.kept_teams[i] = f"{base_team} ({final_name}) - {player_name}"
                    tbd_count += 1
            session.save()
            
            # Now complete the week with finalized names
            week_result, created = BucketWeekResult.objects.get_or_create(
                season=season,
                week_number=current_week,
                defaults={
                    'punishment': session.punishment,
                    'kept_teams': []
                }
            )
            week_result.kept_teams.extend(session.kept_teams)
            week_result.save()
            
            team_wl_status = request.session.get('team_wl_status', {})
            for i, team_entry in enumerate(session.kept_teams):
                team_name, player_name = parse_team_entry(team_entry)
                is_win = team_wl_status.get(str(i))
                
                if team_name == "Golden Helmet of Life":
                    is_win = True
                elif team_name == "Instant Death":
                    is_win = False
                
                BucketTeamResult.objects.create(
                    week_result=week_result,
                    team_name=team_name,
                    player_name=player_name,
                    is_win=is_win
                )
            
            if 'team_wl_status' in request.session:
                del request.session['team_wl_status']
            session.delete()
            messages.success(request, f'Week {current_week} selections finalized and saved!')
            return redirect(f'/punishments/bucket-of-death/?week={current_week}')
        elif action == 'complete_week':
            # Check for TBD teams that need final names
            tbd_teams = [team for team in session.kept_teams if '(TBD)' in team]
            if tbd_teams:
                return render(request, 'draftgame/bucket_finalize_tbd.html', {
                    'session': session,
                    'tbd_teams': tbd_teams,
                    'current_week': current_week
                })
            
            week_result, created = BucketWeekResult.objects.get_or_create(
                season=season,
                week_number=current_week,
                defaults={
                    'punishment': session.punishment,
                    'kept_teams': []
                }
            )
            # Append new teams to existing kept_teams list
            week_result.kept_teams.extend(session.kept_teams)
            week_result.save()
            
            # Create individual team results (append, don't delete existing)
            team_wl_status = request.session.get('team_wl_status', {})
            for i, team_entry in enumerate(session.kept_teams):
                team_name, player_name = parse_team_entry(team_entry)
                is_win = team_wl_status.get(str(i))  # Get W/L from session
                
                # Auto-set win/loss for special teams
                if team_name == "Golden Helmet of Life":
                    is_win = True
                elif team_name == "Instant Death":
                    is_win = False
                
                BucketTeamResult.objects.create(
                    week_result=week_result,
                    team_name=team_name,
                    player_name=player_name,
                    is_win=is_win
                )
            # Clear session data but preserve week selection
            if 'team_wl_status' in request.session:
                del request.session['team_wl_status']
            session.delete()
            messages.success(request, f'Week {current_week} selections appended and saved!')
            return redirect(f'/punishments/bucket-of-death/?week={current_week}')
        elif action == 'end_bucket':
            leaderboard, sorted_tallies = compute_bucket_standings(season, session)
            return render(request, 'draftgame/bucket_summary.html', {
                'session': session,
                'current_week': current_week,
                'leaderboard': leaderboard,
                'team_tallies': sorted_tallies
            })

    # Handle leaderboard request for non-staff users
    if request.method == 'POST' and request.POST.get('action') == 'end_bucket':
        leaderboard, sorted_tallies = compute_bucket_standings(season)
        return render(request, 'draftgame/bucket_summary.html', {
            'session': None,
            'current_week': current_week,
            'leaderboard': leaderboard,
            'team_tallies': sorted_tallies
        })

    # Get kept teams and team results data
    # For non-admin users, try to get the most recent session to show current progress
    if not session and not request.user.is_staff:
        try:
            session = BucketSession.objects.filter(season=season, week_number=current_week).order_by('-updated_at').first()
        except:
            pass
    
    if session:
        kept_team_names = [get_base_team_name(parse_team_entry(entry)[0]) for entry in session.kept_teams]
        current_kept_teams = session.kept_teams
    else:
        kept_team_names = []
        current_kept_teams = []
    
    # Get team results for W/L buttons and add to excluded teams
    team_results_data = []
    try:
        week_result = BucketWeekResult.objects.get(season=season, week_number=current_week)
        team_results_data = list(week_result.team_results.all())
        # Add previously selected teams to the exclusion list
        for team_result in team_results_data:
            base_name = get_base_team_name(team_result.team_name)
            if base_name not in kept_team_names:
                kept_team_names.append(base_name)
    except BucketWeekResult.DoesNotExist:
        pass
    
    available_teams = [team for team in NFL_TEAMS if team not in kept_team_names]
    
    # Get previous weeks for tabs
    previous_weeks = BucketWeekResult.objects.filter(season=season).order_by('week_number')
    previous_week_numbers = [w.week_number for w in previous_weeks]

    # Get completed week data for the current week being viewed
    completed_week_teams = []
    try:
        week_result = BucketWeekResult.objects.get(season=season, week_number=current_week)
        for team_result in week_result.team_results.all():
            completed_week_teams.append({
                'name': f"{team_result.team_name} - {team_result.player_name}",
                'is_win': team_result.is_win,
                'id': team_result.id
            })
    except BucketWeekResult.DoesNotExist:
        pass
    
    # Create team objects with W/L status for current teams
    team_wl_status = request.session.get('team_wl_status', {})
    current_teams_with_status = []
    for i, team in enumerate(current_kept_teams):
        wl_status = team_wl_status.get(str(i))
        current_teams_with_status.append({
            'name': team,
            'index': i,
            'is_win': wl_status if wl_status is not None else None
        })
    
    # Calculate total teams kept for this week (completed + current session)
    total_teams_kept = len(completed_week_teams) + len(current_kept_teams)
    
    context = {
        'session': session,
        'available_teams': available_teams,
        'teams_remaining': len(available_teams),
        'total_teams_kept': total_teams_kept,
        'current_week': current_week,

        'previous_weeks': previous_weeks,
        'previous_week_numbers': previous_week_numbers,
        'completed_week_teams': completed_week_teams,
        'can_complete': len(current_kept_teams) > 0 and request.user.is_staff,
        'weeks_range': range(1, 19),  # NFL season weeks
        'current_kept_teams': current_kept_teams,
        'current_teams_with_status': current_teams_with_status,
        'team_results_data': team_results_data
    }
    
    return render(request, 'draftgame/bucket_of_death.html', context)

@user_passes_test(lambda u: u.is_staff)
def bucket_rummage(request):
    season, _ = get_selected_season(request)
    week = request.session.get('bucket_week', 1)
    if not request.session.session_key:
        return redirect(f'/punishments/bucket-of-death/?week={week}')

    try:
        session = BucketSession.objects.get(session_key=request.session.session_key, season=season)
    except BucketSession.DoesNotExist:
        return redirect(f'/punishments/bucket-of-death/?week={week}')

    # Get kept teams with base name extraction for special teams
    kept_team_names = [get_base_team_name(parse_team_entry(entry)[0]) for entry in session.kept_teams]
    # Also exclude teams already selected in previous completions of this week
    try:
        week_result = BucketWeekResult.objects.get(season=season, week_number=session.week_number)
        for team_result in week_result.team_results.all():
            base_name = get_base_team_name(team_result.team_name)
            if base_name not in kept_team_names:
                kept_team_names.append(base_name)
    except BucketWeekResult.DoesNotExist:
        pass

    available_teams = [team for team in NFL_TEAMS if team not in kept_team_names]

    if not available_teams:
        return redirect(f'/punishments/bucket-of-death/?week={session.week_number}')

    selected_team = random.choice(available_teams)
    session.selected_team = selected_team
    session.save()



    return render(request, 'draftgame/bucket_rummage.html', {
        'session': session,
        'selected_team': selected_team
    })

@user_passes_test(lambda u: u.is_staff)
def bucket_decision(request):
    season, _ = get_selected_season(request)
    week = request.session.get('bucket_week', 1)
    if not request.session.session_key:
        return redirect(f'/punishments/bucket-of-death/?week={week}')

    try:
        session = BucketSession.objects.get(session_key=request.session.session_key, season=season)
    except BucketSession.DoesNotExist:
        return redirect(f'/punishments/bucket-of-death/?week={week}')

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'keep':
            # Check if it's a special team that needs team selection
            if session.selected_team in ["The Big Apple", "The Wizard of Oz (Lions, Tigers, Bears)", "Thank You For Your Service", "Any Underdog"]:
                return redirect('draftgame:bucket_special_team')
            else:
                # Handle auto-win/loss teams with messages
                if session.selected_team == "Golden Helmet of Life":
                    messages.success(request, f'🏆 {session.current_player} got Golden Helmet of Life - automatic win!')
                elif session.selected_team == "Instant Death":
                    messages.success(request, f'💀 {session.current_player} got Instant Death - automatic loss!')
                
                session.kept_teams.append(f"{session.selected_team} - {session.current_player}")
                session.must_keep_next = False
                session.current_player = ''
                session.selected_team = ''
                session.save()
                return redirect(f'/punishments/bucket-of-death/?week={session.week_number}')
        elif action == 'put_back':
            session.must_keep_next = True
            session.save()
            return redirect('draftgame:bucket_rummage_must_keep')
    
    # Check if it's an auto-keep team
    is_auto_keep = session.selected_team in ["Golden Helmet of Life", "Instant Death"]
    
    return render(request, 'draftgame/bucket_decision.html', {
        'session': session,
        'selected_team': session.selected_team,
        'must_keep': session.must_keep_next,
        'is_auto_keep': is_auto_keep
    })

@user_passes_test(lambda u: u.is_staff)
def bucket_rummage_must_keep(request):
    season, _ = get_selected_season(request)
    week = request.session.get('bucket_week', 1)
    if not request.session.session_key:
        return redirect(f'/punishments/bucket-of-death/?week={week}')

    try:
        session = BucketSession.objects.get(session_key=request.session.session_key, season=season)
    except BucketSession.DoesNotExist:
        return redirect(f'/punishments/bucket-of-death/?week={week}')

    # Get kept teams with base name extraction for special teams
    kept_team_names = [get_base_team_name(parse_team_entry(entry)[0]) for entry in session.kept_teams]
    # Also exclude teams already selected in previous completions of this week
    try:
        week_result = BucketWeekResult.objects.get(season=season, week_number=session.week_number)
        for team_result in week_result.team_results.all():
            base_name = get_base_team_name(team_result.team_name)
            if base_name not in kept_team_names:
                kept_team_names.append(base_name)
    except BucketWeekResult.DoesNotExist:
        pass
    
    available_teams = [team for team in NFL_TEAMS if team not in kept_team_names]
    
    if not available_teams:
        return redirect(f'/punishments/bucket-of-death/?week={session.week_number}')
    
    selected_team = random.choice(available_teams)
    session.selected_team = selected_team
    session.save()
    

    
    return render(request, 'draftgame/bucket_must_keep.html', {
        'session': session,
        'selected_team': selected_team
    })

@user_passes_test(lambda u: u.is_staff)
def bucket_special_team(request):
    season, _ = get_selected_season(request)
    week = request.session.get('bucket_week', 1)
    if not request.session.session_key:
        return redirect(f'/punishments/bucket-of-death/?week={week}')

    try:
        session = BucketSession.objects.get(session_key=request.session.session_key, season=season)
    except BucketSession.DoesNotExist:
        return redirect(f'/punishments/bucket-of-death/?week={week}')
    
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'decide_later':
            final_team = f"{session.selected_team} (TBD)"
            session.kept_teams.append(f"{final_team} - {session.current_player}")
            session.must_keep_next = False
            session.current_player = ''
            session.selected_team = ''
            session.save()
            return redirect(f'/punishments/bucket-of-death/?week={session.week_number}')
        else:
            chosen_team = request.POST.get('chosen_team', '').strip()
            if chosen_team:
                final_team = f"{session.selected_team} ({chosen_team})"
                session.kept_teams.append(f"{final_team} - {session.current_player}")
                session.must_keep_next = False
                session.current_player = ''
                session.selected_team = ''
                session.save()
                return redirect(f'/punishments/bucket-of-death/?week={session.week_number}')
    
    return render(request, 'draftgame/bucket_special_team.html', {
        'session': session,
        'special_team': session.selected_team
    })

@user_passes_test(lambda u: u.is_staff)
def bucket_results_admin(request):
    season, _ = get_selected_season(request)
    if request.method == 'POST':
        for key, value in request.POST.items():
            if key.startswith('result_'):
                team_result_id = key.split('_')[1]
                try:
                    team_result = BucketTeamResult.objects.get(id=team_result_id)
                    team_result.is_win = value == 'win'
                    team_result.save()
                except BucketTeamResult.DoesNotExist:
                    pass
        messages.success(request, 'Results updated!')
        return redirect('draftgame:bucket_results_admin')

    # Get all team results grouped by week for the selected season
    weeks_data = []
    for week_result in BucketWeekResult.objects.filter(season=season).order_by('-week_number'):
        team_results = week_result.team_results.all().order_by('player_name')
        weeks_data.append({
            'week_result': week_result,
            'team_results': team_results
        })
    
    return render(request, 'draftgame/bucket_results_admin.html', {
        'weeks_data': weeks_data
    })



# Draft lottery participants with their ball counts
DRAFT_PARTICIPANTS = {
    "Alex": 11,
    "Carter": 6,
    "Sachin": 6,
    "Ari": 6,
    "Nick": 5,
    "Ricky": 5,
    "Matt": 5,
    "Jeff": 5,
    "Danny": 4,
    "Tobin": 4,
    "Jacob": 4,
    "Austin": 4
}

def draft_lottery(request):
    # Create lottery balls based on participant weights
    lottery_balls = []
    for name, count in DRAFT_PARTICIPANTS.items():
        for _ in range(count):
            lottery_balls.append(name)
    
    # Store the draft order if we've run the simulation
    draft_order = []
    animation_complete = False
    
    if request.method == 'POST' and 'run_lottery' in request.POST:
        # Simulate the lottery
        remaining_balls = lottery_balls.copy()
        remaining_participants = list(DRAFT_PARTICIPANTS.keys())
        
        while remaining_participants:
            # Pick a random ball
            selected_ball = random.choice(remaining_balls)
            draft_order.append(selected_ball)
            
            # Remove all balls for this participant
            remaining_participants.remove(selected_ball)
            remaining_balls = [ball for ball in remaining_balls if ball != selected_ball]
            
            # If we've selected all participants, we're done
            if not remaining_participants:
                break
        
        animation_complete = True
    
    # Convert participants dict to items for template
    participants_items = DRAFT_PARTICIPANTS.items()
    
    return render(request, 'draftgame/draft_lottery.html', {
        'participants': participants_items,
        'draft_order': draft_order,
        'animation_complete': animation_complete
    })

def punishment_wheel(request):
    if not request.user.is_staff:
        return render(request, 'draftgame/punishment_wheel.html', {'under_construction': True})

    season, _ = get_selected_season(request)

    # Once the grid is finalized, the wheel is locked.
    if season and season.grid_finalized:
        if request.method == 'POST':
            messages.error(request, f'The {season.year} grid is finalized. Unlock it on the Grid page to run the wheel.')
            return redirect('draftgame:punishment_wheel')
        return render(request, 'draftgame/punishment_wheel.html', {'locked': True, 'locked_year': season.year})

    # The wheel spins the season's locked-in finalists (auto-seeded when voting closes).
    ensure_finalists(season)
    wheel_punishments = [p.text for p in season_finalists(season)]
    if not wheel_punishments:
        if request.method == 'POST':
            messages.error(request, 'Close voting first to lock in the punishments for this season.')
            return redirect('draftgame:punishment_wheel')
        return render(request, 'draftgame/punishment_wheel.html', {
            'not_finalized': True,
            'wheel_year': season.year if season else '',
        })

    if request.method == 'POST':
        if 'setup_wheel' in request.POST:
            name = request.POST.get('name')
            excluded = request.POST.get('excluded_punishment')

            available = [p for i, p in enumerate(wheel_punishments) if str(i) != excluded]
            if len(available) < 2:
                messages.error(request, 'Not enough punishments available for spinning.')
                return redirect('draftgame:punishment_wheel')

            return render(request, 'draftgame/punishment_wheel.html', {
                'punishments': wheel_punishments,
                'available_punishments': available,
                'player_name': name,
                'excluded': excluded,
                'show_wheel': True
            })

        elif 'choose_punishment' in request.POST:
            name = request.POST.get('player_name')
            chosen = request.POST.get('choose_punishment')

            if name and chosen:
                try:
                    WheelResult.objects.filter(name=name, season=season).delete()
                    result = WheelResult.objects.create(
                        name=name,
                        punishment=chosen,
                        season=season
                    )
                    return render(request, 'draftgame/punishment_wheel.html', {
                        'punishments': wheel_punishments,
                        'show_confirmation': True,
                        'saved_name': name,
                        'saved_punishment': chosen[:100]
                    })
                except Exception:
                    messages.error(request, 'Something went wrong saving the assignment. Please try again.')
                    return redirect('draftgame:punishment_wheel')
            else:
                messages.error(request, 'Please choose both a player and a punishment.')
                return redirect('draftgame:punishment_wheel')

    return render(request, 'draftgame/punishment_wheel.html', {
        'punishments': wheel_punishments
    })

def logout_view(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('landing')
