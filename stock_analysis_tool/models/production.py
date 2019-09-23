import models
import csv
import io
import logging
import numpy as np
from models import DataPreprocessor
import utility.date
from models.model_1_0_0 import p4_3_cnn_multi_input_deep_no_dropout as deploy_model


class TensorflowProduction:

    def __init__(self):
        self._keras_model = deploy_model()
        models.load(self._keras_model)
        self._separate_input = hasattr(self._keras_model, "multi_input")
        logging.info("TensorflowProduction initiated.")

    def predict(self, ticker: str, num_year: int) -> str:
        """
          :param ticker: which stock you want to plot
          :param num_year: how many years do you want to see
          :return: none
        """
        logging.info("TensorflowProduction is now requested to predict %s" % ticker)

        # get data
        ret = DataPreprocessor()\
            .load_from_raw_json(single_ticker=ticker, force_update=True, save=False) \
            .expand() \
            .extract_sequence(sample_offset=1, year_cutoff=num_year) \
            .get(separate_input=self._separate_input)

        xx, _, yy, _, date, price = ret

        # predict and transform (tanh -> percentage -> ratio)
        yp = self._keras_model.predict(xx)
        yp = np.arctanh(yp) * 10.0
        predict_change = price * yp / 100.0

        # change to list and reverse
        price = price.reshape(-1).tolist()
        predict_change = predict_change.reshape(-1).tolist()

        # write csv in memory
        output = io.StringIO()
        csv_writer = csv.writer(output, delimiter=',')
        csv_writer.writerow(("date", "close", "predict"))
        for x, y, dy in zip(date, price, predict_change):
            x = int(x)
            y = float(y)
            dy = float(dy)
            dt = utility.date.int_to_date(x).strftime("%Y-%m-%d")
            csv_writer.writerow((dt, y, dy))
        return output.getvalue()


if __name__ == "__main__":
    tf = TensorflowProduction()
    a = tf.predict("AAPL", 5)
    print(a)
