from django.urls import path
from .views import model_page

app_name = 'model'

urlpatterns = [
    path('', model_page, name='model_page'),
]
