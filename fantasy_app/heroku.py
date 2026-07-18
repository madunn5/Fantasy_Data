"""
Production settings for Heroku deployment
"""
import os
from .settings import *

# Production settings
DEBUG = False
ALLOWED_HOSTS = [
    'fantasy-draft-order-890f802bbbec.herokuapp.com',
    'dunn-right-fantasy-a91a2b941097.herokuapp.com',
    '.herokuapp.com',
]

# Django 4+ requires the full https origin for CSRF on POSTs.
CSRF_TRUSTED_ORIGINS = [
    'https://fantasy-draft-order-890f802bbbec.herokuapp.com',
    'https://dunn-right-fantasy-a91a2b941097.herokuapp.com',
]

# The surviving Heroku app (fantasy-draft-order) sets SECRET_KEY rather than
# DJANGO_SECRET_KEY; accept either.
if os.environ.get('SECRET_KEY'):
    SECRET_KEY = os.environ['SECRET_KEY']

# Use PostgreSQL in production
import dj_database_url
DATABASES = {
    'default': dj_database_url.config(default=os.environ.get('DATABASE_URL'))
}

# Yahoo Fantasy API Configuration for Production
YAHOO_FANTASY_CONFIG = {
    'LEAGUE_ID': '605174',
    'LEAGUE_KEY': 'nfl.l.605174',
    'SEASON': 2025,
    'CLIENT_ID': os.environ.get('YAHOO_CLIENT_ID', ''),
    'CLIENT_SECRET': os.environ.get('YAHOO_CLIENT_SECRET', ''),
    'OAUTH_JSON': os.environ.get('YAHOO_OAUTH_JSON', '')
}

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')