from django.urls import path
from .views import result_page, result_csv

app_name = 'result'

urlpatterns = [
    path('<symbol_name>/', result_page, name='result_page'),
    path('<symbol_name>/csv/', result_csv, name='result_csv'),
]
