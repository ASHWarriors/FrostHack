from django.shortcuts import render

# Create your views here.

def first(request):
    return render(request,'index.html')

def second(request):
    return render(request,'dsc_sesh1_b.html')
