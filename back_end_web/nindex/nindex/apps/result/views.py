from django.shortcuts import render
from django.http import HttpResponse
from nindex.settings import STATIC_DIR
import os
import time


def result_page(request, symbol_name: str):
    # validate ticker name
    ticker_validity = True
    if type(symbol_name) is not str:
        ticker_validity = False
    elif len(symbol_name) > 5:
        ticker_validity = False
    if ticker_validity is False:
        return HttpResponse("symbol_name_invalid_error", status=400)

    # response
    data = {'symbol_name': symbol_name.upper()}
    return render(request, 'result/result_page.html', context=data)


def result_csv(request, symbol_name):
    time.sleep(3)

    # validate ticker name
    ticker_validity = True
    if type(symbol_name) is not str:
        ticker_validity = False
    elif len(symbol_name) > 5:
        ticker_validity = False
    if ticker_validity is False:
        return HttpResponse("symbol_name_invalid_error", status=400)

    # return csv file
    path = os.path.join(STATIC_DIR, "data.csv")
    with open(path) as fp:
        csv_data = fp.read()
        response = HttpResponse(csv_data, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="data.csv"'
        return response
