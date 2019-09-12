import tensorflow as tf
import utility


@utility.multi_input_model
@utility.named_model
def p4_2_cnn_multi_input():

    def get_submodel():
        input_ts = tf.keras.layers.Input(shape=(20, 1))
        temp1 = tf.keras.layers.Convolution1D(8, 3, activation="relu")(input_ts)
        temp1 = tf.keras.layers.MaxPooling1D(2)(temp1)
        temp1 = tf.keras.layers.Convolution1D(16, 3, activation="relu")(temp1)
        temp1 = tf.keras.layers.MaxPooling1D(2)(temp1)
        output_ts = tf.keras.layers.Flatten()(temp1)
        return input_ts, output_ts

    inp_list = []
    opt_list = []
    for _ in range(5):
        inp, opt = get_submodel()
        inp_list.append(inp)
        opt_list.append(opt)

    temp2 = tf.concat(opt_list, axis=1)
    temp2 = tf.keras.layers.Dropout(0.5)(temp2)
    temp2 = tf.keras.layers.Dense(32, activation="relu")(temp2)
    temp2 = tf.keras.layers.Dense(8, activation="relu")(temp2)
    output_final = tf.keras.layers.Dense(1, activation="tanh")(temp2)

    model = tf.keras.Model(inputs=inp_list, outputs=output_final)
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


@utility.multi_input_model
@utility.named_model
def p4_2_cnn_multi_input_deep():

    def get_submodel():
        input_ts = tf.keras.layers.Input(shape=(20, 1))
        temp1 = tf.keras.layers.Convolution1D(8, 3)(input_ts)
        temp1 = tf.keras.layers.BatchNormalization(axis=1)(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3)(temp1)
        temp1 = tf.keras.layers.BatchNormalization(axis=1)(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3)(temp1)
        temp1 = tf.keras.layers.BatchNormalization(axis=1)(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3)(temp1)
        temp1 = tf.keras.layers.BatchNormalization(axis=1)(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3)(temp1)
        temp1 = tf.keras.layers.BatchNormalization(axis=1)(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        output_ts = tf.keras.layers.Flatten()(temp1)
        return input_ts, output_ts

    inp_list = []
    opt_list = []
    for _ in range(5):
        inp, opt = get_submodel()
        inp_list.append(inp)
        opt_list.append(opt)

    temp2 = tf.concat(opt_list, axis=1)
    temp2 = tf.keras.layers.Dense(32, activation="relu")(temp2)
    temp2 = tf.keras.layers.Dropout(0.5)(temp2)
    temp2 = tf.keras.layers.Dense(8, activation="relu")(temp2)
    output_final = tf.keras.layers.Dense(1, activation="tanh")(temp2)

    model = tf.keras.Model(inputs=inp_list, outputs=output_final)
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


def p4_2_cnn_multi_input_deep_mae():

    def get_submodel():
        input_ts = tf.keras.layers.Input(shape=(20, 1))
        temp1 = tf.keras.layers.Convolution1D(8, 3)(input_ts)
        temp1 = tf.keras.layers.BatchNormalization(axis=1)(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3)(temp1)
        temp1 = tf.keras.layers.BatchNormalization(axis=1)(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3)(temp1)
        temp1 = tf.keras.layers.BatchNormalization(axis=1)(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3)(temp1)
        temp1 = tf.keras.layers.BatchNormalization(axis=1)(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3)(temp1)
        temp1 = tf.keras.layers.BatchNormalization(axis=1)(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        output_ts = tf.keras.layers.Flatten()(temp1)
        return input_ts, output_ts

    inp_list = []
    opt_list = []
    for _ in range(5):
        inp, opt = get_submodel()
        inp_list.append(inp)
        opt_list.append(opt)

    temp2 = tf.concat(opt_list, axis=1)
    temp2 = tf.keras.layers.Dense(32, activation="relu")(temp2)
    temp2 = tf.keras.layers.Dropout(0.5)(temp2)
    temp2 = tf.keras.layers.Dense(8, activation="relu")(temp2)
    output_final = tf.keras.layers.Dense(1, activation="tanh")(temp2)

    model = tf.keras.Model(inputs=inp_list, outputs=output_final)
    model.compile(optimizer='Adam',
                  loss='mae',
                  metrics=['mae', 'mse'])
    return model


def p4_2_cnn_multi_input_deep_regularize():

    def get_submodel():
        input_ts = tf.keras.layers.Input(shape=(20, 1))
        temp1 = tf.keras.layers.Convolution1D(8, 3, kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_ts)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3, kernel_regularizer=tf.keras.regularizers.l2(0.001))(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3, kernel_regularizer=tf.keras.regularizers.l2(0.001))(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3, kernel_regularizer=tf.keras.regularizers.l2(0.001))(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        temp1 = tf.keras.layers.Convolution1D(8, 3, kernel_regularizer=tf.keras.regularizers.l2(0.001))(temp1)
        temp1 = tf.keras.layers.LeakyReLU()(temp1)
        output_ts = tf.keras.layers.Flatten()(temp1)
        return input_ts, output_ts

    inp_list = []
    opt_list = []
    for _ in range(5):
        inp, opt = get_submodel()
        inp_list.append(inp)
        opt_list.append(opt)

    temp2 = tf.concat(opt_list, axis=1)
    temp2 = tf.keras.layers.Dense(32, activation="relu")(temp2)
    temp2 = tf.keras.layers.Dropout(0.5)(temp2)
    temp2 = tf.keras.layers.Dense(8, activation="relu")(temp2)
    output_final = tf.keras.layers.Dense(1, activation="tanh")(temp2)

    model = tf.keras.Model(inputs=inp_list, outputs=output_final)
    model.compile(optimizer='Adam',
                  loss='mse',
                  metrics=['mae', 'mse'])
    return model


p4_2 = [p4_2_cnn_multi_input,
        p4_2_cnn_multi_input_deep,
        p4_2_cnn_multi_input_deep_regularize]
