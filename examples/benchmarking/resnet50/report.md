
TorchServe Benchmark on gpu
===========================

# Date: 2023-07-18 04:55:29

# TorchServe Version: 0.8.1

## eager_mode_resnet50

|version|Benchmark|Batch size|Batch delay|Workers|Model|Concurrency|Input|Requests|TS failed requests|TS throughput|TS latency P50|TS latency P90|TS latency P99|TS latency mean|TS error rate|Model_p50|Model_p90|Model_p99|predict_mean|handler_time_mean|waiting_time_mean|worker_thread_mean|cpu_percentage_mean|memory_percentage_mean|gpu_percentage_mean|gpu_memory_percentage_mean|gpu_memory_used_mean|handler_preprocess_mean|handler_inference_mean|handler_postprocess_mean|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|0.8.1|AB|1|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|0|99.93|982|1205|1463|1000.695|0.0|18.84|24.62|25.97|37.98|37.85|953.51|0.43|50.0|34.18|25.5|28.51|6566.0|28.92|7.39|0.29|
|0.8.1|AB|16|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|9999|104.86|920|1326|1622|953.641|99.99|448.82|553.22|578.83|596.05|595.73|341.17|9.92|100.0|35.41|0.0|30.13|6938.0|570.62|19.7|0.29|
|0.8.1|AB|2|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|9998|93.53|1051|1300|1507|1069.126|99.98|37.2|56.11|62.15|82.27|82.05|975.93|2.07|50.0|33.63|10.5|27.11|6242.0|70.94|7.97|0.23|
|0.8.1|AB|32|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|9123|134.14|735|1091|1469|745.498|91.23|104.72|198.32|218.35|504.1|503.73|16.7|6.86|100.0|37.7|36.0|33.7|7760.0|474.67|24.98|0.31|
|0.8.1|AB|4|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|3|97.82|1009|1247|1455|1022.334|0.03|82.93|131.6|139.87|158.27|158.02|851.26|3.71|100.0|33.76|11.0|27.38|6304.0|145.68|8.16|0.25|
|0.8.1|AB|64|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|9710|216.49|458|549|654|461.913|97.1|259.07|273.75|282.77|362.22|362.15|54.17|13.63|0.0|45.38|18.0|47.95|11042.0|307.99|52.71|0.34|
|0.8.1|AB|8|100|4|[.mar](file:///home/ubuntu/serve/model_store/resnet-50.mar)|100|[input](./examples/image_classifier/kitten.jpg)|10000|9999|102.11|961|1249|1562|979.354|99.99|192.75|263.79|281.48|304.86|304.59|661.39|6.1|100.0|34.29|4.0|28.17|6488.0|288.49|11.0|0.28|
