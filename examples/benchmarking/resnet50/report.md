
TorchServe Benchmark on gpu
===========================

# Date: 2023-07-18 00:56:21

# TorchServe Version: 0.8.1

## eager_mode_resnet50

|version|Benchmark|Batch size|Batch delay|Workers|Model|Concurrency|Input|Requests|TS failed requests|TS throughput|TS latency P50|TS latency P90|TS latency P99|TS latency mean|TS error rate|Model_p50|Model_p90|Model_p99|handler_time_mean|predict_mean|waiting_time_mean|worker_thread_mean|cpu_percentage_mean|memory_percentage_mean|gpu_percentage_mean|gpu_memory_percentage_mean|gpu_memory_used_mean|backend_preprocess_mean|backend_inference_mean|backend_postprocess_mean|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.8.1|AB|1|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|0|97.91|998|1259|1420|1021.319|0.0|19.04|24.84|26.19|38.7|38.83|974.83|0.42|50.0|35.11|18.0|28.51|6566.0|29.85|7.32|0.27|
|0.8.1|AB|16|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|9999|102.3|949|1380|1750|977.526|99.99|436.65|556.46|576.54|610.41|610.67|349.69|10.15|100.0|36.51|8.0|30.13|6938.0|585.39|19.59|0.26|
|0.8.1|AB|2|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|9998|94.47|1042|1258|1488|1058.591|99.98|38.15|56.68|62.99|81.12|81.36|966.6|2.16|0.0|34.35|10.5|27.11|6242.0|70.01|8.01|0.24|
|0.8.1|AB|32|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|9119|132.2|735|1157|1501|756.404|91.19|110.47|194.85|225.53|513.62|513.85|16.01|6.62|100.0|38.35|2.0|33.7|7760.0|484.35|24.86|0.26|
|0.8.1|AB|4|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|3|100.3|975|1213|1453|997.042|0.03|82.37|124.11|133.04|154.0|154.24|829.77|3.7|100.0|34.59|12.0|27.38|6304.0|141.65|8.01|0.29|
|0.8.1|AB|64|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|9382|213.37|465|552|641|468.677|93.82|260.14|295.21|305.11|366.7|366.78|59.21|14.16|0.0|48.69|15.0|52.48|12084.0|312.83|52.37|0.34|
|0.8.1|AB|8|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|9999|100.38|969|1263|1571|996.255|99.99|199.61|268.68|285.87|309.21|309.48|671.79|6.71|100.0|34.99|14.0|28.18|6490.0|293.13|10.72|0.28|
