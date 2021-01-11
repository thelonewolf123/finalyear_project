from django.urls import path
from .views import *

urlpatterns = [ path('',index,name='home'),
                 path('add_new/',add_person,name='add_new'),
                 path('single_person/<id>/',single_person,name='single_person'),
                 path('cross_ref/<id>/',cross_ref,name='cross_ref')
                 ]
