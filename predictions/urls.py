from django.urls import path
from . import views

urlpatterns = [
    path('',views.index,name='main'),
    path('home/', views.home, name='home'),
    path('predict/', views.predict_stocks, name='predict_stocks'),
    path('results/', views.results, name='results'),
    path('upload/', views.upload_csv, name='upload_csv'),
    path('set_loss/<str:order_id>/',views.set_loss,name='set_loss'),
    path('set_profit/<str:order_id>/',views.set_profit,name='set_profit')

]
