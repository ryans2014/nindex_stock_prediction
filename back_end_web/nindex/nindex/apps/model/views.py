from django.shortcuts import render


def model_page(request):
    return render(request, 'model/model_page.html')
