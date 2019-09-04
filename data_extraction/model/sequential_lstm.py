import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from model import get_batch_input_array
from model.util import get_accuracy_matrix
import functools
import pickle as pk


def get_date():
    x_total, y_total = next(get_batch_input_array(batch_size=-1, sample_offset=20, year_cutoff=15))
    x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.2, random_state=123)
    return x_train, x_test, y_train, y_test


def save(model):
    weight_file = "model_weights\\" + model.cname
    model.save_weights(weight_file)


def load(model):
    weight_file = "model_weights\\" + model.cname
    model.load_weights(weight_file)


def train(model, epochs, x_train, x_test, y_train, y_test, save_weight=False):
    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=20,
                        validation_data=(x_test, y_test),
                        use_multiprocessing=True,
                        verbose=1)
    if save_weight:
        save(model)
    return history


def evaluate(model, xx, yy, history=None, cutoff=0.3):
    if history is not None:
        plt.plot(history.epoch, history.history["mae"])
        plt.plot(history.epoch, history.history["val_mae"])
    accuracy_matrix = get_accuracy_matrix(model, xx, yy, cutoff)
    score = accuracy_matrix[0][0] + accuracy_matrix[2][2] - accuracy_matrix[2][0] - accuracy_matrix[0][2]
    score = float(score) / float(sum(accuracy_matrix[0]) + sum(accuracy_matrix[2]))
    print(accuracy_matrix)
    print("Score = %f" % score)


def _attach_model_cname(fun):
    @functools.wraps(fun)
    def make_model_and_set_name():
        model = fun()
        model_name = fun.__name__
        model.cname = model_name
        return model
    return make_model_and_set_name


@_attach_model_cname
def lstm_v1():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@_attach_model_cname
def lstm_v1_leaky_relu():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(4, activation=tf.nn.leaky_relu))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@_attach_model_cname
def lstm_v1_sgd():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=1.e-5, momentum=0.9),
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@_attach_model_cname
def lstm_v1_adam():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@_attach_model_cname
def lstm_v1_loss_mae_adam():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(32, input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer='Adam',
                  loss='mae',
                  metrics=['mae', 'mse'])
    return model
