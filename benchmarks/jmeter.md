# Benchmarking with JMeter

## Installation

It assumes that you have followed quick start/installation section and have required pre-requisites i.e. python3, java and docker [if needed]. If not then please refer [quick start](../README.md) for setup.

### Ubuntu

We have provided an `install_dependencies.sh` script to install everything needed to execute the benchmark on user's Ubuntu environment. First clone the TorchServe repository:

```bash
git clone https://github.com/pytorch/serve.git
```
Now execute this script as below.
On CPU based instance, use `./install_dependencies.sh`.
On GPU based instance, use `./install_dependencies.sh True`.

### MacOS

For mac, you should have python3 and java installed. If you wish to run the default benchmarks featuring a docker-based instance of TorchServe, you will need to install docker as well. Finally, you will need to install jmeter with plugins which can be accomplished by running `mac_install_dependencies.sh`.

The benchmarking script requires the following to run:
- python3
- JDK or OpenJDK
- jmeter installed through homebrew or linuxbrew with the plugin manager and the following plugins: jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults=2.1,jpgc-cmd=2.1,jpgc-perfmon=2.1
- nvidia-docker

### Windows

For Windows, you should have python3 and java(OpenJDK-17) installed. You will need to install jmeter with plugins which can be accomplished by running `python windows_install_dependencies.py <Path to install jmeter>`. For example:
```bash
python3 windows_install_dependencies.py "C:\\Program Files"
```

## Models

The pre-trained models for the benchmark can be mostly found in the [TorchServe model zoo](../docs/model_zoo.md). We currently support the following:
- [resnet: ResNet-18 (Default)](https://torchserve.pytorch.org/mar_files/resnet-18.mar)
- [squeezenet: SqueezeNet V1.1](https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar)

## Benchmarks

We support several basic benchmarks:
- throughput: Run inference with enough threads to occupy all workers and ensure full saturation of resources to find the throughput. The number of threads defaults to 100.
- latency: Run inference with a single thread to determine the latency
- ping: Test the throughput of pinging against the frontend
- load: Loads the same model many times in parallel. The number of loads is given by the "count" option and defaults to 16.
- repeated_scale_calls: Will scale the model up to "scale_up_workers"=16 then down to "scale_down_workers"=1 then up and down repeatedly.
- multiple_models: Loads and scales up three models (1. squeeze-net and 2. resnet), at the same time, runs inferences on them, and then scales them down. Use the options "urlN", "modelN_name", "dataN" to specify the model url, model name, and the data to pass to the model respectively. data1 and data2 are of the format "&apos;Some garbage data being passed here&apos;" and data3 is the filesystem path to a file to upload.

We also support compound benchmarks:
- concurrent_inference: Runs the basic benchmark with different numbers of threads

#### Using pre-build docker image

* You can specify, docker image using --docker option. You must create docker by following steps given [here](../docker/README.md).

```bash
cd serve/benchmarks
./benchmark.py latency -l 1 --docker pytorch/torchserve:0.1.1-cpu
```

* If you don't specify --ts or --docker then it will use latest image for torchserve on dockerhub and start container by the name of 'ts_benchmark_gpu' or 'ts_benchmark_cpu' depending on whether you have selected --gpus or not

```bash
cd serve/benchmarks
./benchmark.py latency -l 1
```

NOTE - '--docker' and '--ts' are mutually exclusive options

#### Using local TorchServe instance:

* Install TorchServe using the [install guide](../README.md#install-torchserve-and-torch-model-archiver)
* Start TorchServe using following command :

```bash
torchserve --start --model-store <path_to_your_model_store>
```
* To start benchmarking execute following commands

```bash
cd serve/benchmarks
python benchmark.py throughput --ts http://127.0.0.1:8080
```

#### By using external docker container for TorchServe:

* Create and start a [docker container for TorchServe](../docker/README.md).
* To start benchmarking execute following commands

```bash
cd serve/benchmarks
python benchmark.py throughput --ts http://127.0.0.1:8080
```

Note:
1) Refer the examples below to run different benchmarking suites on TorchServe.

## Accessing benchmark reports :

The benchmark reports are available at /tmp/TSBenchmark/

## Examples

Run basic latency test on default resnet-18 model\
```./benchmark.py latency```


Run basic throughput test on default resnet-18 model.\
```./benchmark.py throughput```


Run all benchmarks\
```./benchmark.py --all```


Run using the squeeze-net model\
```./benchmark.py latency -m squeezenet1_1```


Run on GPU (4 gpus)\
```./benchmark.py latency -g 4```


Run with a custom image\
```./benchmark.py latency -i {imageFilePath}```


Run with a custom model (works only for CNN based models, which accept image as an input for now. We will add support for more input types in future to this command. )\
```./benchmark.py latency -c {modelUrl} -i {imageFilePath}```


Run with custom options\
```./benchmark.py repeated_scale_calls --options scale_up_workers 100 scale_down_workers 10```


Run against an already running instance of TorchServe\
```./benchmark.py latency --ts 127.0.0.1``` (defaults to http, port 80, management port = port + 1)\
```./benchmark.py latency --ts 127.0.0.1:8080 --management-port 8081```


Run with multiple models \
```./benchmark.py multiple_models```

Run verbose with only a single loop\
```./benchmark.py latency -v -l 1```

## Known Issues(Running with SSL):
Using ```https``` instead of ```http``` as the choice of protocol might not work properly currently. This is not a tested option.
```./benchmark.py latency --ts https://127.0.0.1:8443```


## Benchmark options

The full list of options can be found by running with the -h or --help flags.

## Adding test plans
Refer [adding a new jmeter](add_jmeter_test.md) test plan for torchserve.

# Profiling

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
