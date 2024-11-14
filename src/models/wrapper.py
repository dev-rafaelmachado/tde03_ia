from re import X
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from typing import Callable, Tuple, List
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import time


def wrapper(
    X_train,
    y_train,
    X_test,
    y_test,
    num_features,
    metric: Callable = accuracy_score,
    penalty_function: Callable[[float, int], float] = lambda acc, features: acc
    - 0.01 * features,
    verbose: bool = True,
) -> Tuple[float, List[int], float, float]:
    start_time_feature = time.time()
    selected_features = []

    model = DecisionTreeClassifier(random_state=1)

    progress_bar = tqdm(
        total=100,
        desc="Feature Selection Progress",
        bar_format="{l_bar}{bar}| {n_fmt}% Complete",
        position=0,
    )

    while len(selected_features) < num_features:
        current_best_metric = 0
        best_feature = None

        with tqdm(
            total=num_features,
            desc="Evaluating Features",
            leave=False,
            disable=not verbose,
            position=1,
        ) as inner_pbar:
            for feature in range(num_features):
                if feature in selected_features:
                    inner_pbar.update(1)
                    continue

                model.fit(X_train.iloc[:, selected_features + [feature]], y_train)
                y_pred = model.predict(X_test.iloc[:, selected_features + [feature]])
                metric_value = metric(y_test, y_pred)

                penalized_metric = penalty_function(
                    metric_value, len(selected_features) + 1
                )

                if penalized_metric > current_best_metric:
                    current_best_metric = penalized_metric
                    best_feature = feature

                if verbose:
                    inner_pbar.set_postfix(
                        {
                            "Current Best Metric": f"{current_best_metric:.4f}",
                            "Selected Features": len(selected_features),
                        }
                    )
                inner_pbar.update(1)

        if best_feature is None:
            break

        selected_features.append(best_feature)

        progress_percentage = (len(selected_features) / num_features) * 100
        progress_bar.n = int(progress_percentage)
        progress_bar.refresh()

    progress_bar.close()

    feature_search_time = time.time() - start_time_feature

    start_time_model = time.time()
    model.fit(X_train.iloc[:, selected_features], y_train)
    training_time = time.time() - start_time_model

    y_pred = model.predict(X_test.iloc[:, selected_features])
    final_metric = metric(y_test, y_pred)
    return final_metric, selected_features, feature_search_time, training_time
