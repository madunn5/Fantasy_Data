"""
Production settings for Heroku deployment
"""
import os
from .settings import *

# Production settings
DEBUG = False
ALLOWED_HOSTS = ['dunn-right-fantasy-a91a2b941097.herokuapp.com', 'yourdomain.com', '.herokuapp.com']

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
    'CLIENT_ID': os.environ.get('YAHOO_CLIENT_ID_PROD', ''),
    'CLIENT_SECRET': os.environ.get('YAHOO_CLIENT_SECRET_PROD', '')
}

# Security settings
SECURE_SSL_REDIRECT = True
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')