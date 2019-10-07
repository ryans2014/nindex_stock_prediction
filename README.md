# Nindex Stock Prediction
This repository contains the implementation of a stock prediction website. It contains three major aspects:
* Under back_end_web folder, there is a Django based application that generates the web-related stuff of the website.
* Under stock_analysis_tool folder, there are several Python modules that can be deployed to provide real-time stock predictions. These modules can also be used for researches about machine learning based stock market prediction.

## Service Architecture
The website is currently deployed on http://18.191.170.241/. 

The service architecture is shown below:

<p align="center">
<img src="https://github.com/ryans2014/nindex_stock_prediction/blob/master/contents/service_arch.svg" width="500">
</p>

Nginx handles three types of requests. Static contents are handled by Nginx itself. Request for the dynamic web pages are delegated to Django-based apps. There are also stock prediction requests initiated by an asynchronous javascript call. This request is routed to an stock predictor application. If it found pre-computed results from MongoDB, the results are sent back. Otherwise, it will fetch real-time stock data from a data provider, process the data, and send it to TensorFlow to calculate the prediction. 

After the prediction data is sent back to the user, it will be used by a javascript function (based on D3.js) to render the interactive stock prediction chart.

The stock prediction request involves fetching data from a third party data provider. There are times when the data provider restricts the frequency of the data requests. As a result, a stock prediction request may take quite a long time at extreme situations. To improve the user experience, this request is made asynchronous. The server-side request handler is also made asynchronous to avoid overhead caused by  multithreading and multiprocessing.

