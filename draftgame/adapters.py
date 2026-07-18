from allauth.socialaccount.adapter import DefaultSocialAccountAdapter
from django.contrib.auth.models import User


class ConnectByEmailAdapter(DefaultSocialAccountAdapter):
    """Link a Google sign-in to an existing account that has the same email.

    Google verifies email addresses, and in a closed league each person owns
    one email, so it's safe to connect the social login to the matching user
    rather than forcing a second account. New emails fall through to the normal
    (auto) signup flow.
    """

    def pre_social_login(self, request, sociallogin):
        if sociallogin.is_existing:
            return
        email = (sociallogin.user.email or '').strip()
        if not email:
            return
        try:
            user = User.objects.get(email__iexact=email)
        except (User.DoesNotExist, User.MultipleObjectsReturned):
            return
        sociallogin.connect(request, user)
