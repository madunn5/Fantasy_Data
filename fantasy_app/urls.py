"""
URL configuration for the combined fantasy site.

Two halves share one project:
  /data/        — league stats & analytics (fantasy_data)
  /punishments/ — punishment voting, wheel, Bucket of Death (draftgame)
  /             — landing page that routes to either half
  /accounts/    — single sign-in for both (Google via allauth + username/password)
"""
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from django.views.generic import TemplateView

from draftgame.auth_views import CustomLoginView
from draftgame.views import logout_view, register

urlpatterns = [
    path('', TemplateView.as_view(template_name='landing.html'), name='landing'),
    path('admin/', admin.site.urls),
    path('django-rq/', include('django_rq.urls')),
    path('accounts/login/', CustomLoginView.as_view(), name='login'),
    path('accounts/logout/', logout_view, name='logout'),
    path('accounts/register/', register, name='register'),
    path('accounts/', include('django.contrib.auth.urls')),
    path('accounts/', include('allauth.urls')),  # Google sign-in (and social routes)
    path('data/', include('fantasy_data.urls')),
    path('punishments/', include('draftgame.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
