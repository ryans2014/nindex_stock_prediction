import models
import csv
import numpy as np
import matplotlib.pyplot as plt
from models import DataPreprocessor
import utility.date


def plot_prediction_bars(keras_model, ticker="AAPL", num_year=5, threshold=0.03):
    """
    :param keras_model: can be the model name ,can be the keras model, can be the function to generate model
    :param ticker: which stock you want to plot
    :param num_year: how many years do you want to see
    :param threshold: where the arrow color changess
    :return: none

    ------------ To load a single ticker -------------------------
    from models import plot_prediction_bars
    from models.phase2_models.test_iteration3 import lstm_v3_stack
    plot_prediction_bars(lstm_v3_stack, "AAPL")
    """

    if type(keras_model).__name__ == "function":
        keras_model = keras_model()

    # load model
    models.load(keras_model)

    # get data
    separate_input = hasattr(keras_model, "multi_input")
    ret = DataPreprocessor().load_from_raw_json(single_ticker=ticker)\
                            .expand()\
                            .extract_sequence(sample_offset=1, year_cutoff=num_year)\
                            .get(separate_input=separate_input)
    xx, _, yy, _, date, price = ret

    # predict and transform (tanh -> percentage -> ratio)
    yp = keras_model.predict(xx)
    yp = np.arctanh(yp) * 10.0
    predict_change = price * yp / 100.0

    # change to list and reverse
    date_idx = [i for i in range(len(date))]
    price = price.reshape(-1).tolist()
    predict_change = predict_change.reshape(-1).tolist()

    # plot
    plt.plot(date_idx, price, color='k')
    for x, y, dy in zip(date_idx, price, predict_change):
        if dy / y > threshold:
            plt.arrow(x, y, 0.0, dy, color='r')
        elif dy / y < -threshold:
            plt.arrow(x, y, 0.0, dy, color='b')
        else:
            plt.arrow(x, y, 0.0, dy, color='y')

    # write csv
    with open('data.csv', 'w', newline='') as fp:
        csv_writer = csv.writer(fp, delimiter=',')
        csv_writer.writerow(("date", "close", "predict"))
        for x, y, dy in zip(date, price, predict_change):
            x = int(x)
            y = float(y)
            dy = float(dy)
            dt = utility.date.int_to_date(x).strftime("%Y-%m-%d")
            csv_writer.writerow((dt, y, dy))
