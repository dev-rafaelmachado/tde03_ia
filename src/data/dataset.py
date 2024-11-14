import pandas as pd
import os

actual_path = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(actual_path, "..", ".."))


def load_data():
    mnist_test = pd.read_csv(f"{project_root}/static/input/mnist_test.csv")
    mnist_train = pd.read_csv(f"{project_root}/static/input/mnist_train.csv")

    X_train = mnist_train.iloc[:, 1:]
    y_train = mnist_train.iloc[:, 0]
    X_test = mnist_test.iloc[:, 1:]
    y_test = mnist_test.iloc[:, 0]

    return X_train, y_train, X_test, y_test
