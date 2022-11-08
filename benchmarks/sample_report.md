
TorchServe Benchmark on gpu
===========================

# Date: 2022-03-22 04:27:24

# TorchServe Version: master

## eager_mode_mnist

|version|Benchmark|Batch size|Batch delay|Workers|Model|Concurrency|Requests|TS failed requests|TS throughput|TS latency P50|TS latency P90|TS latency P99|TS latency mean|TS error rate|Model_p50|Model_p90|Model_p99|predict_mean|handler_time_mean|waiting_time_mean|worker_thread_mean|cpu_percentage_mean|memory_percentage_mean|gpu_percentage_mean|gpu_memory_percentage_mean|gpu_memory_used_mean|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|master|AB|1|100|4|[.mar](https://torchserve.pytorch.org/mar_files/mnist_v2.mar)|10|10000|0|2345.99|4|5|6|4.263|0.0|1.04|1.15|1.53|1.06|1.02|1.93|0.28|0.0|0.0|0.0|0.0|0.0|
|master|AB|2|100|4|[.mar](https://torchserve.pytorch.org/mar_files/mnist_v2.mar)|10|10000|0|3261.31|3|4|5|3.066|0.0|1.36|1.91|2.18|1.45|1.41|0.17|0.44|0.0|0.0|0.0|0.0|0.0|
|master|AB|4|100|4|[.mar](https://torchserve.pytorch.org/mar_files/mnist_v2.mar)|10|10000|0|2457.64|4|6|7|4.069|0.0|1.89|2.2|2.96|1.97|1.94|0.53|0.59|0.0|0.0|0.0|0.0|0.0|
|master|AB|8|100|4|[.mar](https://torchserve.pytorch.org/mar_files/mnist_v2.mar)|10|10000|0|1640.2|5|9|11|6.097|0.0|2.95|3.15|3.43|3.0|2.96|1.06|0.8|0.0|0.0|0.0|0.0|0.0|
|master|AB|1|100|8|[.mar](https://torchserve.pytorch.org/mar_files/mnist_v2.mar)|10|10000|0|3444.57|3|3|4|2.903|0.0|1.32|1.68|1.87|1.37|1.34|0.08|0.46|0.0|0.0|0.0|0.0|0.0|
|master|AB|2|100|8|[.mar](https://torchserve.pytorch.org/mar_files/mnist_v2.mar)|10|10000|0|3275.88|3|4|5|3.053|0.0|1.61|2.23|2.51|1.72|1.68|0.01|0.55|0.0|0.0|0.0|0.0|0.0|
|master|AB|4|100|8|[.mar](https://torchserve.pytorch.org/mar_files/mnist_v2.mar)|10|10000|0|2346.15|4|6|8|4.262|0.0|2.01|2.42|3.19|2.1|2.06|0.57|0.57|0.0|0.0|0.0|0.0|0.0|
|master|AB|8|100|8|[.mar](https://torchserve.pytorch.org/mar_files/mnist_v2.mar)|10|10000|0|1572.82|5|9|12|6.358|0.0|3.09|3.39|4.7|3.15|3.11|1.1|0.82|0.0|0.0|0.0|0.0|0.0|

## eager_mode_vgg16

|version|Benchmark|Batch size|Batch delay|Workers|Model|Concurrency|Requests|TS failed requests|TS throughput|TS latency P50|TS latency P90|TS latency P99|TS latency mean|TS error rate|Model_p50|Model_p90|Model_p99|predict_mean|handler_time_mean|waiting_time_mean|worker_thread_mean|cpu_percentage_mean|memory_percentage_mean|gpu_percentage_mean|gpu_memory_percentage_mean|gpu_memory_used_mean|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|master|AB|1|100|4|[.mar](https://torchserve.pytorch.org/mar_files/vgg16.mar)|100|10000|0|277.64|353|384|478|360.178|0.0|13.27|14.49|18.55|13.61|13.57|343.11|0.35|69.2|11.3|22.25|12.4|2004.0|
|master|AB|2|100|4|[.mar](https://torchserve.pytorch.org/mar_files/vgg16.mar)|100|10000|0|284.7|344|377|462|351.248|0.0|25.69|29.79|49.7|26.86|26.82|320.57|0.84|33.3|11.29|16.25|12.39|2002.0|
|master|AB|4|100|4|[.mar](https://torchserve.pytorch.org/mar_files/vgg16.mar)|100|10000|0|298.66|331|355|386|334.831|0.0|50.61|54.65|72.63|51.69|51.64|278.95|1.33|66.7|11.63|16.0|12.81|2070.0|
|master|AB|8|100|4|[.mar](https://torchserve.pytorch.org/mar_files/vgg16.mar)|100|10000|0|302.97|321|367|401|330.066|0.0|100.17|108.43|134.97|102.03|101.97|222.5|2.62|0.0|12.1|15.25|13.4|2166.0|

## scripted_mode_vgg16

|version|Benchmark|Batch size|Batch delay|Workers|Model|Concurrency|Requests|TS failed requests|TS throughput|TS latency P50|TS latency P90|TS latency P99|TS latency mean|TS error rate|Model_p50|Model_p90|Model_p99|predict_mean|handler_time_mean|waiting_time_mean|worker_thread_mean|cpu_percentage_mean|memory_percentage_mean|gpu_percentage_mean|gpu_memory_percentage_mean|gpu_memory_used_mean|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|master|AB|1|100|4|[.mar](https://torchserve.pytorch.org/mar_files/vgg16_scripted.mar)|100|10000|0|282.06|351|368|430|354.53|0.0|13.18|13.91|18.68|13.41|13.37|337.73|0.33|80.0|11.32|23.25|12.4|2004.0|
|master|AB|2|100|4|[.mar](https://torchserve.pytorch.org/mar_files/vgg16_scripted.mar)|100|10000|0|288.03|345|363|406|347.18|0.0|25.68|29.08|40.61|26.53|26.49|316.93|0.83|37.5|11.31|16.5|12.39|2002.0|
|master|AB|4|100|4|[.mar](https://torchserve.pytorch.org/mar_files/vgg16_scripted.mar)|100|10000|0|296.25|332|356|447|337.552|0.0|50.72|55.09|84.0|52.09|52.04|281.21|1.34|0.0|11.63|16.0|12.81|2070.0|
|master|AB|8|100|4|[.mar](https://torchserve.pytorch.org/mar_files/vgg16_scripted.mar)|100|10000|0|301.07|324|367|407|332.147|0.0|100.49|109.71|136.18|102.69|102.63|223.7|2.59|0.0|0.0|0.0|0.0|0.0|
