import pandas as pd

MODES = ["eager_mode", "scripted_mode"]
METRICS_VALIDATED = [
    "throughput",
    "total_latency_p50",
    "model_latency_p50",
    "total_latency_p90",
    "model_latency_p90",
    "total_latency_p99",
    "model_latency_p99",
    "memory_percentage_mean",
    "gpu_used_memory_mean",
    "cpu_percentage_mean",
    "gpu_percentage_mean",
]


# Acceptable metric deviation needs a more complicated logic.
# Example: For latencies in 2 digits, 50% might be acceptable
# For 3 digit latencies, 20-30% might be the right value
# For cpu_memory < 15%, 50% deviation works but for CPU > 40%, 10-15%
# might be the right value

ACCEPTABLE_METRIC_DEVIATION = 0.3


class Report:
    def __init__(self):
        self.properties = {}
        self.mode = None
        self.throughput = 0
        self.batch_size = 0
        self.workers = 0
        self.deviation = ACCEPTABLE_METRIC_DEVIATION

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

    def update(self, report):
        for property in self.properties:
            self.properties[property] = (
                self.properties[property] + report.properties[property]
            ) / 2.0
