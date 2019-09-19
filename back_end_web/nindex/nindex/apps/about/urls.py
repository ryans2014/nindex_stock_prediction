from django.conf.urls import url, include
from .views import

urlpatterns = [
    url('/', include('nindex.apps.home.urls'), name='home'),
    url('index/', include('nindex.apps.about.urls'), name='about'),
    url('home/', include('nindex.apps.model.urls'), name='model'),
]
