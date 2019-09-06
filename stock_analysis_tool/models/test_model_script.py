import configuration
from models.keras_model_utility import *
from models.test_iteration3 import test_iteration3 as model_functions

comparison_name = "test_iteration3"

# check if multiple input pipes
separate_x = False
if hasattr(model_functions[0](), "multi_input"):
    separate_x = True

# get data
x_train, x_test, y_train, y_test = get_data(load_from_file=True, separate_input=separate_x)

# run all models
hist_list = []
model_list = []
for my_model in model_functions:
    md1 = my_model()
    print("Working on model %s..." % md1.cname)
    history = train(md1, 5000, x_train, x_test, y_train, y_test)
    hist_list.append(history)
    model_list.append(md1)
    save(md1, save_model=True)

# evaluate and plot
plot_history(hist_list, model_list, comparison_name)
evaluate(model_list, x_test, y_test, comparison_name, cutoff=5.0)
