import json
import os

actual_path = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(actual_path, "..", ".."))


def save_results(results, name):
    file_name = f"{project_root}/static/output/{name}.json"
    with open(file_name, "w") as f:
        json.dump(results, f, indent=4)
