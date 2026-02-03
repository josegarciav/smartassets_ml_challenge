import json
import os
from datetime import datetime

class ExperimentTracker:
    """
    Simple local experiment tracker to log model runs, hyperparameters, and metrics.
    """
    def __init__(self, log_path: str = "data/experiments.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump([], f)

    def log_experiment(self, name: str, params: dict, metrics: dict):
        """
        Logs a single experiment run.
        """
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "experiment_name": name,
            "parameters": params,
            "metrics": metrics
        }

        try:
            with open(self.log_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []

        data.append(entry)

        with open(self.log_path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"--- Experiment Tracked: {name} ---")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

    def get_history(self) -> list:
        """
        Returns all logged experiments.
        """
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path, "r") as f:
            return json.load(f)

    def clear_history(self):
        """
        Clears the experiment history.
        """
        with open(self.log_path, "w") as f:
            json.dump([], f)
