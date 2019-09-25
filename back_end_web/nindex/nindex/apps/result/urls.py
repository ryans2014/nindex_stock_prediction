from django.urls import path
from .views import result_page, result_csv

app_name = 'result'

urlpatterns = [
    path('<symbol_name>/', result_page, name='result_page'),
    path('csv/<symbol_name>/', result_csv, name='result_csv'),
]
