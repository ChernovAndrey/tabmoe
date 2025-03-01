import json


class HyperparamLogger:
    """A class to store and save hyperparameters in a structured dictionary format."""

    def __init__(self):
        self.params = {}  # Stores parameters in memory
    def log(self, category, **kwargs):
        """Log function parameters by updating the existing"""
        if category not in self.params:
            self.params[category] = {}  # Initialize if category doesn't exist

        self.params[category].update(kwargs)  # Merge new key-value pairs

    def save_hyperparams(self, filepath="hyperparams.json"):
        """Save all collected parameters to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.params, f, indent=4)