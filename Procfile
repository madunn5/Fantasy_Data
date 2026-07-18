release: python manage.py migrate --noinput
web: gunicorn fantasy_app.wsgi --timeout 120 --graceful-timeout 30 --threads 2
