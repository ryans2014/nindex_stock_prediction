from django.shortcuts import render, redirect


def home_page(request):
    data = {"error_message": ""}
    if request.method == "POST":
        symbol_name = ""
        # detect error symbol name
        if request.POST.__contains__("symbol"):
            symbol_name = request.POST["symbol"]
        ticker_validity = True
        if type(symbol_name) is not str:
            ticker_validity = False
        elif len(symbol_name) > 5 or len(symbol_name) < 1:
            ticker_validity = False
        if ticker_validity is False:
            data["error_message"] = "invalid symbol name"
            return render(request, 'home/home_page.html', context=data)
        # redirect to result page
        symbol_name = symbol_name.upper()
        return redirect('/result/' + symbol_name + '/')
    else:
        return render(request, 'home/home_page.html', context=data)

