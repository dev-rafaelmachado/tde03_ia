from data.dataset import load_data

from utils.evaluation import evaluate_models
from utils.save import save_results

import sys

EXEC_BASELINE = True
EXEC_WRAPPER = True
EXEC_GA = True
NUM_FEATURES = None

if __name__ == "__main__":

    if len(sys.argv) > 1:
        exec_models = int(sys.argv[1])
        EXEC_BASELINE = exec_models % 1000 >= 100
        EXEC_WRAPPER = exec_models % 100 >= 10
        EXEC_GA = exec_models % 10
    if len(sys.argv) > 2:
        NUM_FEATURES = int(sys.argv[2])

    X_train, y_train, X_test, y_test = load_data()

    evaluation_results = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        {"baseline": EXEC_BASELINE, "wrapper": EXEC_WRAPPER, "ga": EXEC_GA},
        NUM_FEATURES,
    )

    save_results(evaluation_results, "results")

    print("Bye bye!")
