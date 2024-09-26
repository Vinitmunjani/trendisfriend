from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
    path('login/', views.login, name='login'),
    path('success/', views.success, name='success'),
    path('logout/', auth_views.LogoutView.as_view(next_page='login'), name='logout'),
    # other url patterns
]
