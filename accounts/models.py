from django.db import models
from django.contrib.auth.models import User

class AppUser(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    mobile = models.CharField(max_length=15)

    def __str__(self) -> str:
        return self.user.username

class DhanClient(models.Model):
    app_user = models.OneToOneField(AppUser, on_delete=models.CASCADE)
    client_id = models.CharField(max_length=100, null=True, blank=True)
    access_token = models.CharField(max_length=1000, null=True, blank=True)
    api_key = models.CharField(max_length=100, default='dhan')
    app_passcode = models.CharField(max_length=4)

    def __str__(self) -> str:
        return self.app_user.user.username

class AngelClient(models.Model):
    app_user = models.OneToOneField(AppUser, on_delete=models.CASCADE)
    client_id = models.CharField(max_length=100, null=True, blank=True)
    access_token = models.CharField(max_length=100, null=True, blank=True)
    api_key = models.CharField(max_length=100, default='angel')
    app_passcode = models.CharField(max_length=4)

    def __str__(self) -> str:
        return self.app_user.user.username
