from django.shortcuts import render


def webpage(request):
    return render(request,'index.html')
