"""
Production settings for Heroku deployment
"""
import os
from .settings import *

# Production settings
DEBUG = False
ALLOWED_HOSTS = [
    'dunn-right-fantasy-048860865119.herokuapp.com',
    '.herokuapp.com',
]

# Django 4+ requires the full https origin for CSRF on POSTs.
CSRF_TRUSTED_ORIGINS = [
    'https://dunn-right-fantasy-048860865119.herokuapp.com',
]

# The Heroku app (dunn-right-fantasy, formerly fantasy-draft-order) sets
# SECRET_KEY rather than DJANGO_SECRET_KEY; accept either.
if os.environ.get('SECRET_KEY'):
    SECRET_KEY = os.environ['SECRET_KEY']

# Use PostgreSQL in production
import dj_database_url
DATABASES = {
    'default': dj_database_url.config(default=os.environ.get('DATABASE_URL'))
}

# Yahoo Fantasy API Configuration for Production
YAHOO_FANTASY_CONFIG = {
    # Yahoo gives the league a new key every season (SNAKES NOT ALIVE:
    # 461.l.605174 in 2025, renewed as 470.l.773198 in 2026).
    'LEAGUE_ID': '773198',
    'LEAGUE_KEY': os.environ.get('YAHOO_LEAGUE_KEY', '470.l.773198'),
    'SEASON': 2026,
    'CLIENT_ID': os.environ.get('YAHOO_CLIENT_ID', ''),
    'CLIENT_SECRET': os.environ.get('YAHOO_CLIENT_SECRET', ''),
    'OAUTH_JSON': os.environ.get('YAHOO_OAUTH_JSON', '')
}

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')