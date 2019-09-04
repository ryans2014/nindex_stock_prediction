from model.sequential_lstm import get_date, save, train, load, evaluate, lstm_v1, lstm_v1_leaky_relu

# get data
x_train, x_test, y_train, y_test = get_date()


# build and train
md1 = lstm_v1_loss_mae_adam()
history = train(md1, 10, x_train, x_test, y_train, y_test)
save(md1)
evaluate(md1, x_test, y_test, history=history, cutoff=0.5)


# load weight and evaluate
md2 = lstm_v1()
load(md2)
evaluate(md2, x_test, y_test, cutoff=0.5)


