"""
Fake "the podcast is back" pages.

There is no podcast this year. Each episode below is a bait-and-switch: link
previews (Discord, iMessage, Slack) read the Open Graph tags and show a
real-looking episode card using last year's YouTube thumbnail. A person who
clicks gets the payoff instead, either a redirect or a full-screen gif.

To add a week, add an entry here. The URL is /podcast/<slug>/.
"""
from django.http import Http404
from django.shortcuts import render

EPISODES = {
    'week-1': {
        'title': 'Week 1 Fantasy Podcast - We Are Back!',
        'description': 'The boys are back for another season. Week 1 preview, '
                       'bold takes, and who is already cooked.',
        'youtube_id': 'IV52wTMZV3M',          # last year's real episode, for the thumbnail
        'redirect': 'https://www.youtube.com/watch?v=dQw4w9WgXcQ',
    },
    'week-2': {
        'title': 'Week 2 Fantasy Podcast - New Faces and Old Faces',
        'description': 'Week 2 is here. Who broke out, who is washed, and the '
                       'waiver adds you need before Sunday.',
        'youtube_id': 'B8JIQZz5jqs',
        # South Park cable company guy. Shown full screen on our page.
        'gif_mp4': 'https://media.giphy.com/media/gaZ51cn7sUY4U/giphy.mp4',
        'gif': 'https://media.giphy.com/media/gaZ51cn7sUY4U/giphy.gif',
    },
}


def episode(request, slug):
    ep = EPISODES.get(slug)
    if ep is None:
        raise Http404
    return render(request, 'podcast.html', {'ep': ep, 'week': slug.replace('-', ' ').title()})
