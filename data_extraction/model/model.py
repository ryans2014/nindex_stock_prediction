import tensorflow as tf
from model import get_batch_input_array
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

x_total, y_total = next(get_batch_input_array(batch_size=5000, sample_offset=200))
x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.2, random_state=123)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(32, input_shape=(20, 5)))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation=tf.keras.activations.linear))

model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
              loss='mse',
              metrics=['mae', 'mse'])

history = model.fit(x_train,
                    y_train,
                    epochs=40,
                    batch_size=20,
                    validation_data=(x_test, y_test),
                    verbose=1)

plt.plot(history.epoch, history.history["mae"])
plt.plot(history.epoch, history.history["val_mae"])


def get_accuracy_matrix(trained_model, x_test, y_test):
    y_predict = trained_model.predict(x_test).reshape(-1)
    y_real = y_test.reshape(-1)
    y_predict = (y_predict > 3.0) + 1 - (y_predict < -3.0)
    y_real = (y_real > 3.0) + 1 - (y_real < -3.0)
    ret = np.zeros(shape=(3, 3), dtype=int)
    for yp, yr in zip(y_predict, y_real):
        ret[yp][yr] += 1
    return ret

