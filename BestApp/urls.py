# -*- coding: utf-8 -*-
"""
Created on Thu May  6 15:38:03 2021

@author: Sristi
"""

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home')
    ]
