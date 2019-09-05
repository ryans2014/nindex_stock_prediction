import tensorflow as tf
from model.model_util import attach_model_cname


@attach_model_cname
def lstm_v1_mse_rmsprop():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(5, input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@attach_model_cname
def lstm_v1_mse_adam():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(5, input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@attach_model_cname
def lstm_v1_mae_adam():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(5, input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    model.compile(optimizer='Adam',
                  loss='mae',
                  metrics=['mae', 'mse'])
    return model


@attach_model_cname
def lstm_v2_stack2():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(5, input_shape=(20, 5), return_sequences=True))
    model.add(tf.keras.layers.LSTM(5, input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@attach_model_cname
def lstm_v2_stack2_dense_connect():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(5, input_shape=(20, 5), return_sequences=True))
    model.add(tf.keras.layers.LSTM(5, input_shape=(20, 5), return_sequences=True))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


lstm_v1 = [lstm_v1_mse_rmsprop,
           lstm_v1_mse_adam,
           lstm_v1_mae_adam]

lstm_v2 = [lstm_v2_stack2,
           lstm_v2_stack2_dense_connect]
