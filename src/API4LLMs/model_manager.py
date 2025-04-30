import json
import os

__all__ = [
    "model_manager",
    "init_model_manager",
]

class ModelManager:
    def __init__(self):
        self.models = {}

    def load_from(self, path):
        if os.path.exists(path) and os.path.isfile(path):
            with open(path, 'r') as file:
                self.models = json.load(file)

    def save_to(self, path):
        with open(path, 'w') as file:
            json.dump(self.models, file, indent=4)

    def add_model(self, model_name, api_key, base_url, model):
        self.delete_model(model_name)
        self.models[model_name] = {
            'api_key': api_key,
            'base_url': base_url,
            'model': model
        }

    def delete_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]

    def update_model(self, model_name, api_key=None, base_url=None, model=None):
        if model_name in self.models:
            if api_key is not None:
                self.models[model_name]['api_key'] = api_key
            if base_url is not None:
                self.models[model_name]['base_url'] = base_url
            if model is not None:
                self.models[model_name]['model'] = model
                         
                
model_manager = ModelManager()


def init_model_manager(
    api_keys_path: str,
)-> None:
    model_manager.load_from(api_keys_path)

