from models.keras_model_utility import *
from models.test_iteration4 import test_iteration4 as model_functions

comparison_name = "test_iteration4"

# run all models
hist_list = []
model_list = []
y_pred_list = []
y_real = []

for my_model in model_functions:
    md1 = my_model()
    print("Working on model %s..." % md1.cname)

    # check if multiple input pipes
    separate_x = False
    if hasattr(md1, "multi_input"):
        separate_x = md1.multi_input

    # get data
    x_train, x_test, y_train, y_test = get_data(load_from_file=True, separate_input=separate_x)

    # train
    history = train(md1, 1500, x_train, x_test, y_train, y_test)

    hist_list.append(history)
    model_list.append(md1)
    y_pred_list.append(md1.predict(x_test))
    y_real.append(y_test)
    save(md1, save_model=True)

# evaluate and plot
plot_history(hist_list, model_list, comparison_name)
evaluate(model_list, y_pred_list, y_real, comparison_name, cutoff=5.0)
