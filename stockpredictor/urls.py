from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('predictions.urls')),
    path('login/', include('accounts.urls'))
]
