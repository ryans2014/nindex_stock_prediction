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

## Code Structure
Folder back_end_web contains the Django application that provides dynamic web pages. Folder stock_analysis_tool contains modules for data fetching, data processing, feature extraction, asynchrouse http server/dummy client, asynchronous socket server/dummy client, tensorflow/keras model definition, and some utility functions.

#### Module: configuration
This module should be imported at the beginning of the application. It reads configuration files (.config), setup work environment, setup logging system, and do some basic checkes.

#### Module: data_extractor
It provide an abstract base class for data extractor and a DataExtractor implementation based on AlphaVantage stock data provider (https://www.alphavantage.co/).
```python
def get_data(ticker: str, force_update=False, save=True)
async def get_data_async(ticker: str)
```
Both function provide a wrapper for DataExtractor. The first function is in synchronous mode and should be used for research and data processing. The second function is in asynchronous mode and is designed for production. 

"save" flag controls whether to store raw json files for later reuse. 

"force_update" flag controls whether to use the previously saved json file or use the data pulled from data provider.

#### Module: model_research
This module stores the TensorFlow/Keras model that has been tried. To define a new model, create a new function that takes no parameter and return the Keras model. Name function by the model name and decorate the function by 
```python 
@utility.named_model 
```
This is needed for the model save/load function to work.  

#### Module: models
This module offers function for data processing, feature extraction, model loading, model saving, model training, result plotting, result evaluating.
(More to be added here)

#### Module: test
Unit test for some functions.

#### Module: utility
Utility functions that are used across the project.
(More to be added here)
