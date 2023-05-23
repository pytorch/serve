import csv

METRICS_VALIDATED = [
    "TS throughput",
    "TS latency P50",
    "TS latency P90",
    "TS latency P99",
    "Model_p50",
    "Model_p90",
    "Model_p99",
    "memory_percentage_mean",
    "gpu_memory_used_mean",
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
    def __init__(self, deviation=0, num_reports=0):
        self.properties = {}
        self.mode = None
        self.throughput = 0
        self.batch_size = 0
        self.workers = 0
        self.deviation = deviation
        self.num_reports = num_reports

    def _get_mode(self, csv_file):
        cfg = csv_file.split("/")[-2]
        cfg = cfg.split("_")
        mode = cfg[0] + "_" + cfg[1]
        self.mode = mode

    def read_csv(self, csv_file):
        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            for k, v in next(reader).items():
                if k in METRICS_VALIDATED:
                    self.properties[k] = float(v)
        self._get_mode(csv_file)

    def update(self, report):
        for property in self.properties:
            # sum the properties to find the mean later
            self.properties[property] += report.properties[property]

    def mean(self):
        for k, v in self.properties.items():
            self.properties[k] = v / self.num_reports


def metric_valid(key, obs_val, exp_val, threshold):
    # In case of throughput, higher is better
    # In case of memory, lower is better.
    # We ignore lower values for memory related metrices
    lower = False
    if "throughput" not in key:
        lower = True
    return check_if_within_threshold(exp_val, obs_val, threshold) or (
        (obs_val < exp_val and lower) or (obs_val > exp_val and not lower)
    )


def check_if_within_threshold(value1, value2, threshold):
    if float(value1) == 0.0:
        return True
    return abs((value1 - value2) / float(value1)) <= threshold
