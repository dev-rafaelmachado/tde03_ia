from typing import Tuple
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time


def baseline_model(X_train, y_train, X_test, y_test) -> Tuple[float, int, float, float]:
    print("loading...")
    model = DecisionTreeClassifier(random_state=1)

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    num_features = X_train.shape[1]

    return float(accuracy), num_features, 0, training_time
