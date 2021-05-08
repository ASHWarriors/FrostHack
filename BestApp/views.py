# -*- coding: utf-8 -*-
"""
Created on Sat May  8 19:30:26 2021

@author: Sristi
"""


from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
def index(request):
    #template=loader.get_template('index.html')
    #return HttpResponse(template.render())
    return render(request,'index.html')