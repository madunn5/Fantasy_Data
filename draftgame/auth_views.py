from django.contrib.auth.views import LoginView
from django.contrib import messages

class CustomLoginView(LoginView):
    def form_valid(self, form):
        response = super().form_valid(form)
        messages.success(self.request, 'Welcome back!')
        return response