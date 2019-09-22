import tensorflow as tf
import utility


@utility.named_model
def mlp_v1_basic():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@utility.named_model
def mlp_v1_basic_dropout():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@utility.named_model
def mlp_v1_basic_fat():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(100, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dense(20, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


mlp_v1 = [mlp_v1_basic, mlp_v1_basic_dropout, mlp_v1_basic_fat]

