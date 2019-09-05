from model.model_util import *
from model.model_lstm import lstm_v1, lstm_v2
from model.model_mlp import mlp_v1
from model.model_multi_inp_cnn import cnn_v1
from model.model_multi_inp_lstm import lstm_multi_v1


def compare_list_of_models(model_functions, comparison_name):
    # get data
    x_train, x_test, y_train, y_test = get_data(load_from_file=True, separate_input=True)

    # run all models
    hist_list = []
    model_list = []
    for my_model in model_functions:
        md1 = my_model()
        print("Working on model %s..." % md1.cname)
        history = train(md1, 500, x_train, x_test, y_train, y_test)
        hist_list.append(history)
        model_list.append(md1)
        save(md1)

    # evaluate and plot
    plot_history(hist_list, model_list, comparison_name)
    evaluate(model_list, x_test, y_test, comparison_name, cutoff=5.0)


def load_model(model_obj):
    md2 = model_obj()
    load(md2)
    return md2


compare_list_of_models(lstm_multi_v1, "lstm_multi_v1")

