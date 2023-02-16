from collections import defaultdict

import pandas as pd
import yaml

MODES = ["eager_mode", "scripted_mode"]


class Report:
    def __init__(self):
        self.properties = {}
        self.mode = None
        self.throughput = 0
        self.batch_size = 0
        self.workers = 0
        self.total_latency_p50 = 0
        self.total_latency_p90 = 0
        self.total_latency_p99 = 0
        self.model_latency_p50 = 0
        self.model_latency_p90 = 0
        self.model_latency_p99 = 0
        self.memory_percentage_mean = 0
        self.gpu_used_memory_mean = 0
        self.cpu_percentage_mean = 0
        self.gpu_percentage_mean = 0

    def _get_mode(self, csv_file):

        cfg = csv_file.split("/")[-2]
        cfg = cfg.split("_")
        mode = cfg[0] + "_" + cfg[1]
        self.mode = mode

    def read_csv(self, csv_file):

        df = pd.read_csv(csv_file)
        values = df.values.tolist()
        self._populate_csv(values[0])
        self._get_mode(csv_file)

    def read_yaml(self, yaml_file, config):

        with open(yaml_file, "r") as f:
            yaml_dict = yaml.safe_load(f)
        self._populate_yaml(yaml_dict, config)

    def _populate_yaml(self, yaml_dict, config):
        for model, cfg in yaml_dict.items():
            for mode in MODES:
                if mode in cfg:
                    self.properties[mode] = defaultdict(int)
                    values = cfg[mode]["batch_size"][config["batch_size"]]
                    self.properties[mode]["deviation"] = cfg["deviation"]
                    self.properties[mode]["throughput"] = values["throughput"]
                    self.properties[mode]["total_latency_p50"] = values[
                        "total_latency_p50"
                    ]
                    self.properties[mode]["model_latency_p50"] = values[
                        "model_latency_p50"
                    ]
                    self.properties[mode]["total_latency_p90"] = values[
                        "total_latency_p90"
                    ]
                    self.properties[mode]["model_latency_p90"] = values[
                        "model_latency_p90"
                    ]
                    self.properties[mode]["total_latency_p99"] = values[
                        "total_latency_p99"
                    ]
                    self.properties[mode]["model_latency_p99"] = values[
                        "model_latency_p99"
                    ]

    def _populate_csv(self, values):
        self.properties["throughput"] = values[9]
        self.properties["batch_size"] = values[1]
        self.properties["total_latency_p50"] = values[10]
        self.properties["total_latency_p90"] = values[11]
        self.properties["total_latency_p99"] = values[12]
        self.properties["model_latency_p50"] = values[15]
        self.properties["model_latency_p90"] = values[16]
        self.properties["model_latency_p99"] = values[17]
        self.properties["memory_percentage_mean"] = values[23]
        self.properties["gpu_used_memory_mean"] = values[26]
        self.properties["cpu_percentage_mean"] = values[22]
        self.properties["gpu_percentage_mean"] = values[24]
