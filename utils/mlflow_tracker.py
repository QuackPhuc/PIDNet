import os
import mlflow


class MLflowTracker:

    def __init__(self, tracking_uri, experiment_name, tags=None):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.tags = tags or {}

    @staticmethod
    def read_data_version(path, prefixes=None):
        if not path:
            return None
        if prefixes is None:
            prefixes = []
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return None
        tags = []
        for line in lines[1:]:
            parts = line.strip().split('|')
            if len(parts) != 3:
                continue
            data_path, commit, tag = parts
            for prefix in list(prefixes):
                if data_path.startswith(prefix):
                    tags.append(tag)
                    prefixes.remove(prefix)
        if len(tags) == 1:
            return tags[0]
        return ",".join(tags) if tags else None

    def start_run(self, run_name=None):
        mlflow.start_run(run_name=run_name)
        if self.tags:
            mlflow.set_tags(self.tags)

    def log_params_from_config(self, config):
        params = {}
        for key, value in config.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    params[f"{key}.{sub_key}"] = str(sub_value)
            else:
                params[key] = str(value)
        mlflow.log_params(params)

    def log_tag(self, key, value):
        mlflow.set_tag(key, value)

    def log_metrics(self, metrics, step):
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, file_path):
        if os.path.isfile(file_path):
            mlflow.log_artifact(file_path)

    def end_run(self):
        mlflow.end_run()
