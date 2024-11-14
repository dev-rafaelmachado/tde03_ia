from typing import Dict
from models.baseline import baseline_model
from models.wrapper import wrapper
from ga.genetic_alg import genetic_algorithm

from .os import clear
from .save import save_results


def evaluate_models(
    X_train, X_test, y_train, y_test, model_to_execute: Dict, p_num_features: int | None
):
    clear()
    results = {}

    baseline_result = {}
    wrapper_result = {}
    ga_result = {}

    total_features = X_train.shape[1]

    num_features = total_features
    if num_features is not None:
        num_features = p_num_features

    if model_to_execute["baseline"]:
        print("-- Baseline --")
        accuracy, bs_features, bs_feature_time, training_time = baseline_model(
            X_train, y_train, X_test, y_test
        )
        baseline_result = {
            "accuracy": accuracy,
            "num_features": bs_features,
            "percentage_of_features": (bs_features / total_features) * 100,
            "feature_search_time": bs_feature_time,
            "training_time": training_time,
        }
        save_results(baseline_result, "baseline")
        clear()

    if model_to_execute["wrapper"]:
        print("-- Wrapper --")
        (
            wrapper_accuracy,
            wrapper_num_features,
            wrapper_feature_time,
            wrapper_training_time,
        ) = wrapper(X_train, y_train, X_test, y_test, num_features)
        wrapper_result = {
            "accuracy": wrapper_accuracy,
            "num_features": len(wrapper_num_features),
            "percentage_of_features": (len(wrapper_num_features) / total_features)
            * 100,
            "feature_search_time": wrapper_feature_time,
            "training_time": wrapper_training_time,
        }
        save_results(wrapper_result, "wrapper")
        clear()

    if model_to_execute["ga"]:
        print("-- Genetic Algorithm --")
        ga_accuracy, ga_num_features, ga_feature_time, ga_training_time = (
            genetic_algorithm(X_train, y_train, X_test, y_test, num_features)
        )
        ga_result = {
            "accuracy": ga_accuracy,
            "num_features": len(ga_num_features),
            "percentage_of_features": (len(ga_num_features) / total_features) * 100,
            "feature_search_time": ga_feature_time,
            "training_time": ga_training_time,
        }
        save_results(ga_result, "ga")
        clear()

    results = {
        "baseline": baseline_result,
        "wrapper": wrapper_result,
        "ga": ga_result,
    }

    return results
