from models import get_batch_input_array
from models.keras_model_utility import load
import numpy as np
import matplotlib.pyplot as plt
import utility


def plot_prediction(keras_model, ticker="AAPL", num_year=5, threshold=0.03):
    """
    :param keras_model: can be the model name ,can be the keras model, can be the function to generate model
    :param ticker: which stock you want to plot
    :param num_year: how many years do you want to see
    :param threshold: where the arrow color changess
    :return: none
    """

    if type(keras_model).__name__ == "function":
        keras_model = keras_model()
    elif type(keras_model) is str:
        keras_model = utility.make_keras_model_by_name(keras_model)

    # load model
    load(keras_model)

    # get data
    xx, yy, date, price = next(get_batch_input_array(-1, 1, 20, year_cutoff=num_year, single_ticker=ticker))

    # predict and transform (tanh -> percentage -> ratio)
    yp = keras_model.predict(xx)
    yp = np.arctanh(yp) * 10.0
    predict_change = price * yp / 100.0

    # change to list and reverse
    date = [i for i in range(len(date))]
    price = price.reshape(-1).tolist()
    price.reverse()
    predict_change = predict_change.reshape(-1).tolist()
    predict_change.reverse()

    # plot
    plt.plot(date, price)
    ax = plt.axes()
    for x, y, dy in zip(date, price, predict_change):
        if dy / y > threshold:
            ax.arrow(x, y, 0.0, dy, color='r')
        elif dy / y < -threshold:
            ax.arrow(x, y, 0.0, dy, color='b')
        else:
            ax.arrow(x, y, 0.0, dy, color='y')
