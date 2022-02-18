# Torchserve Model Server Benchmarking

The benchmarks measure the performance of TorchServe on various models and benchmarks. It supports either a number of built-in models or a custom model passed in as a path or URL to the .mar file. It also runs various benchmarks using these models (see benchmarks section below). The benchmarks are executed in the user machine through a shell script in case of apache benchmark. TorchServe is run on the same machine in a docker instance to avoid network latencies. The benchmark must be run from within the context of the full TorchServe repo(i.e. the benchmark tests reside inside serve/benchmarks folder).

We currently support benchmarking with Apache Bench. You can also profile backend code with snakeviz.

* [Benchmarking with Apache Bench](#benchmarking-with-apache-bench)
* [AutoBenchmarking Apachage Bench on AWS](#benchmarking-apache-bench-aws)
* [Profiling](#profiling)
# Benchmarking with Apache Bench

## Installation

It assumes that you have followed quick start/installation section and have required pre-requisites i.e. python3, java and docker [if needed]. If not then please refer [quick start](../README.md) for setup.

### pip dependencies

`pip install -r requirements-ab.txt`

### install apache2-utils

* Ubuntu

```
apt-get install apache2-utils
```

* macOS

Apache Bench is installed in Mac by default. You can test by running ```ab -h```

* Windows
    - Download apache binaries from [Apache Lounge](https://www.apachelounge.com/download/)
    - Extract and place the contents at some location eg: `C:\Program Files\`
    - Add this path `C:\Program Files\Apache24\bin`to the environment variable PATH.
    NOTE - You may need to  install Visual C++ Redistributable for Visual Studio 2015-2019.

## Benchmark
### Run benchmark
This command will run the AB benchmark with default parameters. It will start a Torchserve instance locally, register Resnet-18 model, and run 100 inference requests with a concurrency of 10.
Refer [parameters section](#benchmark-parameters) for more details on configurable parameters.

`python benchmark-ab.py`

### Run benchmark with a test plan
The benchmark comes with pre-configured test plans which can be used directly to set parameters. Refer available [test plans](#test-plans) for more details.
`python benchmark-ab.py <test plan>`

### Run benchmark with a customized test plan
This command will run Torchserve locally and perform benchmarking on the VGG11 model with test plan `soak` test plan soak has been configured with default Resnet-18 model, here we override it by providing extra parameters. Similarly, all parameters can be customized with a Test plan

`python benchmark-ab.py soak --url https://torchserve.pytorch.org/mar_files/vgg11.mar`

### Run benchmark in docker
This command will run Torchserve inside a docker container and perform benchmarking with default parameters. The docker image used here is the latest CPU based torchserve image available on the docker hub. The custom image can also be used using the `--image` parameter.
`python benchmark-ab.py --exec_env docker`

### Run benchmark in GPU docker
This command will run Torchserve inside a docker container with 4 GPUs and perform benchmarking with default parameters. The docker image used here is the latest GPU based torchserve image available on the docker hub. The custom image can also be used using the `--image` parameter.
`python benchmark-ab.py --exec_env docker --gpus 4`

### Run benchmark using a config file
The config parameters can be provided using cmd line args and a config json file as well.
This command will use all the configuration parameters given in config.json file.
`python benchmark-ab.py --config config.json`. The other parameters like config.properties, inference_model_url can also be added in the config.json.

### Sample config file
```json
{
  "url":"https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar",
  "requests": 1000,
  "concurrency": 10,
  "input": "../examples/image_classifier/kitten.jpg",
  "exec_env": "docker",
  "gpus": "2"
}
```
### Benchmark parameters
The following parameters can be used to run the AB benchmark suite.
- url: Input model URL. Default: `https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar`
- device: Execution device type. Default: cpu
- exec_env: Execution environment. Default: docker
- concurrency: Concurrency of requests. Default: 10
- requests: Number of requests. Default: 100
- batch_size: The batch size of the model. Default: 1
- batch_delay: Max batch delay of the model. Default:200
- workers: Number of worker thread(s) for model
- input: Input file for model
- content_type: Input file content type.
- image: Custom docker image to run Torchserve on. Default: Standard public Torchserve image
- docker_runtime: Specify docker runtime if required
- ts: Use Already running Torchserve instance. Default: False
- gpus: Number of gpus to run docker container with. By default it runs the docker container on CPU.
- backend_profiling: Enable backend profiling using CProfile. Default: False
- config_properties: Path to config.properties file. Default: config.properties in the benchmark directory
- inference_model_url: Inference function url - can be either for predictions or explanations. Default: predictions/benchmark.
- config: All the above params can be set using a config JSON file. When this flag is used, all other cmd line params are ignored.


### Examples

* TORCHSERVE SERVING PREDICTIONS

```
python benchmark-ab.py --url https://torchserve.pytorch.org/mar_files/mnist.mar --content_type application/png --config_properties config.properties --inference_model_url predictions/benchmark --input ../examples/image_classifier/mnist/test_data/0.png
```

* TORCHSERVE SERVING EXPLANATIONS

```
python benchmark-ab.py --url https://torchserve.pytorch.org/mar_files/mnist.mar --content_type application/png --config_properties config.properties --inference_model_url explanations/benchmark --input ../examples/image_classifier/mnist/test_data/0.png
```

* KSERVE SERVING PREDICTIONS

```
python benchmark-ab.py --url https://torchserve.pytorch.org/mar_files/mnist.mar --content_type application/json --config_properties config_kf.properties --inference_model_url v1/models/benchmark:predict --input ../kubernetes/kserve/kf_request_json/mnist.json
```

- KSERVE SERVING EXPLANATIONS

```
python benchmark-ab.py --url https://torchserve.pytorch.org/mar_files/mnist.mar --content_type application/json --config_properties config_kf.properties --inference_model_url v1/models/benchmark:explain --input ../kubernetes/kserve/kf_request_json/mnist.json
```

* TORCHSERVE SERVING PREDICTIONS WITH DOCKER

```
python benchmark-ab.py --url https://torchserve.pytorch.org/mar_files/mnist.mar --content_type application/png --config_properties config.properties --inference_model_url predictions/benchmark --input ../examples/image_classifier/mnist/test_data/0.png --exec_env docker 
```

### Test plans
Benchmark supports pre-defined, pre-configured params that can be selected based on the use case.
1. soak: default model url with requests =100000 and concurrency=10
2. vgg11_1000r_10c: vgg11 model with requests =1000 and concurrency=10
3. vgg11_10000r_100c: vgg11 model with requests =10000 and concurrency=100
4. resnet152_batch: Resnet-152 model with batch size = 4, requests =1000 and concurrency=10
5. resnet152_batch_docker: Resnet-152 model with batch size = 4, requests =1000, concurrency=10 and execution env = docker 

Note: These pre-defined parameters in test plan can be overwritten by cmd line args.

### Benchmark reports
The reports are generated at location "/tmp/benchmark/"
- CSV report: /tmp/benchmark/ab_report.csv
- latency graph: /tmp/benchmark/predict_latency.png
- torchserve logs: /tmp/benchmark/logs/model_metrics.log
- raw ab output: /tmp/benchmark/result.txt

### Sample output CSV
| Benchmark | Model | Concurrency | Requests | TS failed requests | TS throughput | TS latency P50 | TS latency P90| TS latency P90 | TS latency mean | TS error rate | Model_p50 | Model_p90 | Model_p99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---| ---|
| AB | [squeezenet1_1](https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar) | 10 | 100 | 0 | 15.66 | 512 | 1191 | 2024 | 638.695 | 0 | 196.57 | 270.9 | 106.53|

### Sample latency graph
![](predict_latency.png)

# Benchmarking Apache Bench AWS
If you're making a large change to TorchServe it's best to run an [automated benchmarking suite on AWS](https://github.com/pytorch/serve/tree/master/test/benchmark) so that you can test multiple CUDA versions and EC2 hardware configurations easily.

# Profiling
## PyTorch
To profile your PyTorch model you can use TorchServe as is but enable an environment variable first.

```
export ENABLE_TORCH_PROFILER=TRUE
```

To learn more about the PyTorch profiler https://github.com/pytorch/kineto/tree/main/tb_plugin
## Frontend

The benchmarks can be used in conjunction with standard profiling tools such as JProfiler to analyze the system performance. JProfiler can be downloaded from their [website](https://www.ej-technologies.com/products/jprofiler/overview.html).  Once downloaded, open up JProfiler and follow these steps:

1. Run TorchServe directly through gradle (do not use docker). This can be done either on your machine or on a remote machine accessible through SSH.
2. In JProfiler, select "Attach" from the ribbon and attach to the ModelServer. The process name in the attach window should be "com.amazonaws.ml.ts.ModelServer". If it is on a remote machine, select "On another computer" in the attach window and enter the SSH details.  For the session startup settings, you can leave it with the defaults.  At this point, you should see live CPU and Memory Usage data on JProfiler's Telemetries section.
3. Select Start Recordings in JProfiler's ribbon
4. Run the Benchmark script targeting your running TorchServe instance. It might run something like `./benchmark.py throughput --ts https://127.0.0.1:8443`. It can be run on either your local machine or a remote machine (if you are running remote), but we recommend running the benchmark on the same machine as the model server to avoid confounding network latencies.
5. Once the benchmark script has finished running, select Stop Recordings in JProfiler's ribbon

Once you have stopped recording, you should be able to analyze the data. One useful section to examine is CPU views > Call Tree and CPU views > Hot Spots to see where the processor time is going.

## Backend
The benchmarks can also be used to analyze the backend performance using cProfile. To benchmark a backend code, 

1. Install Torchserve

    Using local TorchServe instance:

    * Install TorchServe using the [install guide](../README.md#install-torchserve-and-torch-model-archiver)
    
    By using external docker container for TorchServe:

    * Create a [docker container for TorchServe](../docker/README.md).

2. Set environment variable and start Torchserve

    If using local TorchServe instance:
    ```bash
    export TS_BENCHMARK=TRUE
    torchserve --start --model-store <path_to_your_model_store>
    ```
    If using external docker container for TorchServe:
    * start docker with /tmp directory mapped to local /tmp and set `TS_BENCHMARK` to True.
    ```
        docker run --rm -it -e TS_BENCHMARK=True -v /tmp:/tmp -p 8080:8080 -p 8081:8081 pytorch/torchserve:latest
    ```

3. Register a model & perform inference to collect profiling data. This can be done with the benchmark script described in the previous section.
    ```
    python benchmark.py throughput --ts http://127.0.0.1:8080
    ```

4. Visualize SnakeViz results.
 
    To visualize the profiling data using `snakeviz` use following commands:

    ```bash
    pip install snakeviz
    snakeviz /tmp/tsPythonProfile.prof
    ```
    ![](snake_viz.png)

    It should start up a web server on your machine and automatically open the page. Note that tha above command will fail if executed on a server where no browser is installed. The backend profiling should generate a visualization similar to the pic shown above. 
