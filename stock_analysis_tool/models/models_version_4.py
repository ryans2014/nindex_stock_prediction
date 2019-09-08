import tensorflow as tf
import utility


@utility.named_model
def lstm_v4_stack():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(5, input_shape=(20, 5)))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='tanh'))
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@utility.multi_input_model
@utility.named_model
def lstm_v4_stack4_multi_input():

    def get_submodel():
        input_ts = tf.keras.layers.Input(shape=(20, 1))
        temp1 = tf.keras.layers.LSTM(5, input_shape=(20, 1), return_sequences=True)(input_ts)
        temp1 = tf.keras.layers.LSTM(5, input_shape=(20, 1), return_sequences=True)(temp1)
        temp1 = tf.keras.layers.LSTM(5, input_shape=(20, 1), return_sequences=True)(temp1)
        temp1 = tf.keras.layers.LSTM(5, input_shape=(20, 1))(temp1)
        output_ts = tf.keras.layers.Flatten()(temp1)
        return input_ts, output_ts

    inp_list = []
    opt_list = []
    for _ in range(5):
        inp, opt = get_submodel()
        inp_list.append(inp)
        opt_list.append(opt)

    temp2 = tf.concat(opt_list, axis=1)
    temp2 = tf.keras.layers.Dense(16, activation="relu")(temp2)
    output_final = tf.keras.layers.Dense(1, activation="tanh")(temp2)

    model = tf.keras.Model(inputs=inp_list, outputs=output_final)
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


test_iteration4 = [lstm_v4_stack,
                   lstm_v4_stack4_multi_input]
