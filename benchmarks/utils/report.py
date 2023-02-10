import pandas as pd


class Report:
    def __init__(self, csv_file):
        self.throughput = 0
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

        self._read_csv(csv_file)

    def _read_csv(self, csv_file):

        df = pd.read_csv(csv_file)
        values = df.values.tolist()
        self._populate(values[0])

    def _populate(self, values):
        self.throughput = values[9]
        self.total_latency_p50 = values[10]
        self.total_latency_p90 = values[11]
        self.total_latency_p99 = values[12]
        self.model_latency_p50 = values[15]
        self.model_latency_p90 = values[16]
        self.model_latency_p99 = values[17]
        self.memory_percentage_mean = values[23]
        self.gpu_used_memory_mean = values[26]
        self.cpu_percentage_mean = values[22]
        self.gpu_percentage_mean = values[24]
        print(f"Throughput {self.throughput}")
