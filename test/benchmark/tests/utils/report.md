### Benchmark report

```
vgg16 | eager_mode | p3.8xlarge | batch size 1
```
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg16 | 100 | 1000 | 0 | 208.55 | 442 | 480 | 735 | 479.495 | 0.0 | 15.35 | 15.46 | 15.46 | 17.26 | 17.22 | 411.22 | 0.49 | 

```
vgg16 | eager_mode | p3.8xlarge | batch size 2
```
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg16 | 100 | 1000 | 0 | 208.17 | 444 | 495 | 755 | 480.386 | 0.0 | 30.23 | 30.39 | 30.39 | 34.56 | 34.51 | 392.56 | 1.13 | 

```
vgg16 | eager_mode | p3.8xlarge | batch size 4
```
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg16 | 100 | 1000 | 0 | 218.78 | 430 | 467 | 696 | 457.074 | 0.0 | 61.58 | 61.94 | 61.94 | 67.03 | 66.98 | 345.57 | 2.15 | 

```
vgg16 | eager_mode | p3.8xlarge | batch size 8
```
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg16 | 100 | 1000 | 0 | 221.51 | 411 | 513 | 683 | 451.455 | 0.0 | 123.02 | 124.18 | 124.18 | 130.94 | 130.88 | 272.18 | 4.09 | 

```
vgg16 | scripted_mode | p3.8xlarge | batch size 1
```
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg16 | 100 | 1000 | 0 | 201.51 | 455 | 499 | 831 | 496.260 | 0.0 | 15.22 | 15.39 | 15.39 | 17.98 | 17.94 | 428.86 | 0.49 | 

```
vgg16 | scripted_mode | p3.8xlarge | batch size 2
```
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg16 | 100 | 1000 | 0 | 207.14 | 442 | 479 | 825 | 482.774 | 0.0 | 30.43 | 30.73 | 30.73 | 35.07 | 35.02 | 398.25 | 1.15 | 

```
vgg16 | scripted_mode | p3.8xlarge | batch size 4
```
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg16 | 100 | 1000 | 0 | 206.83 | 438 | 511 | 887 | 483.486 | 0.0 | 61.24 | 61.56 | 61.56 | 69.59 | 69.54 | 359.6 | 2.11 | 

```
vgg16 | scripted_mode | p3.8xlarge | batch size 8
```
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg16 | 100 | 1000 | 0 | 211.91 | 416 | 534 | 844 | 471.905 | 0.0 | 124.8 | 125.38 | 125.38 | 135.9 | 135.84 | 282.59 | 4.02 | 
