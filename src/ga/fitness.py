from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def fitness(individual, X_train, y_train, X_val, y_val, penalty=0.01):
    selected_features = [i for i, gene in enumerate(individual) if gene == 1]

    if len(selected_features) == 0:
        return 0

    model = DecisionTreeClassifier()
    model.fit(X_train.iloc[:, selected_features], y_train)
    y_pred = model.predict(X_val.iloc[:, selected_features])

    acc = accuracy_score(y_val, y_pred)

    return acc - penalty * len(selected_features)
