

# Torchserve Model Server Benchmarking

The benchmarks measure the performance of TorchServe on various models and benchmarks.  It supports either a number of built-in models or a custom model passed in as a path or URL to the .mar file.  It also runs various benchmarks using these models (see benchmarks section below).  The benchmarks are executed in the user machine through a python3 script in case of jmeter and a shell script in case of apache benchmark.  TorchServe is run on the same machine in a docker instance to avoid network latencies.  The benchmark must be run from within the context of the full TorchServe repo(i.e. the benchmark tests reside inside serve/benchmarks folder).

We currently support benchmarking with JMeter & Apache Bench. One can also profile backend code with snakeviz.

* [Benchmarking with JMeter](#benchmarking-with-jmeter)
* [Benchmarking with Apache Bench](#benchmarking-with-apache-bench)
* [Profiling](#profiling)

# Benchmarking with JMeter


## Installation

It assumes that you have followed quick start/installation section and have required pre-requisites i.e. python3, java and docker [if needed]. If not then please refer [quick start](https://github.com/pytorch/serve/blob/master/README.md) for setup.

### Ubuntu

We have provided an `install_dependencies.sh` script to install everything needed to execute the benchmark on user's Ubuntu environment. First clone the TorchServe repository:

```bash
git clone https://github.com/pytorch/serve.git
```
Now execute this script as below.
On CPU based instance, use `./install_dependencies.sh`.
On GPU based instance, use `./install_dependencies.sh True`.

### MacOS

For mac, you should have python3 and java installed.  If you wish to run the default benchmarks featuring a docker-based instance of TorchServe, you will need to install docker as well.  Finally, you will need to install jmeter with plugins which can be accomplished by running `mac_install_dependencies.sh`.

The benchmarking script requires the following to run:
- python3
- JDK or OpenJDK
- jmeter installed through homebrew or linuxbrew with the plugin manager and the following plugins: jpgc-synthesis=2.1,jpgc-filterresults=2.1,jpgc-mergeresults=2.1,jpgc-cmd=2.1,jpgc-perfmon=2.1
- nvidia-docker

## Models

The pre-trained models for the benchmark can be mostly found in the [TorchServe model zoo](https://github.com/pytorch/serve/blob/master/docs/model_zoo.md). We currently support the following:
- [resnet: ResNet-18 (Default)](https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar)
- [squeezenet: SqueezeNet V1.1](https://torchserve.s3.amazonaws.com/mar_files/squeezenet1_1.mar)

## Benchmarks

We support several basic benchmarks:
- throughput: Run inference with enough threads to occupy all workers and ensure full saturation of resources to find the throughput.  The number of threads defaults to 100.
- latency: Run inference with a single thread to determine the latency
- ping: Test the throughput of pinging against the frontend
- load: Loads the same model many times in parallel.  The number of loads is given by the "count" option and defaults to 16.

Following benchmarks will be available upon merging PR #255 to master
- repeated_scale_calls: Will scale the model up to "scale_up_workers"=16 then down to "scale_down_workers"=1 then up and down repeatedly.
- multiple_models: Loads and scales up three models (1. squeeze-net and 2. resnet), at the same time, runs inferences on them, and then scales them down.  Use the options "urlN", "modelN_name", "dataN" to specify the model url, model name, and the data to pass to the model respectively.  data1 and data2 are of the format "&apos;Some garbage data being passed here&apos;" and data3 is the filesystem path to a file to upload.

We also support compound benchmarks:
- concurrent_inference: Runs the basic benchmark with different numbers of threads

#### Using pre-build docker image

* You can specify, docker image using --docker option. You must create docker by following steps given [here](https://github.com/pytorch/serve/tree/master/docker).

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

* Install TorchServe using the [install guide](../README.md#install-torchserve)
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


Run verbose with only a single loop\
```./benchmark.py latency -v -l 1```

## Known Issues(Running with SSL):
Using ```https``` instead of  ```http``` as the choice of protocol might not work properly currently. This is not a tested option.
```./benchmark.py latency --ts https://127.0.0.1:8443```


## Benchmark options

The full list of options can be found by running with the -h or --help flags.

# Benchmarking with Apache Bench

Apache Bench can also be used in torchserve for Benchmarking performance of inference API's. The ApacheBench tool (ab) can load test servers by sending an arbitrary number of concurrent requests.

The benchmarks measure the performance of torchserve on inference API for various models. It supports passed in a URL to the .mar file. It also runs various benchmarks using these models (see benchmarks section below).

## Installation

It assumes that you have followed quick start/installation section and have required pre-requisites i.e. python3, java and docker [if needed]. If not then please refer [quick start](https://github.com/pytorch/serve/blob/master/README.md) for setup.

### Ubuntu

First check if you already have Apache Bench(ab) installed on your Ubuntu box(local/EC2 instance). The following code will help verify the installation −

#### ab -V

If ab is not installed on your machine, then please follow the following steps:

Refresh the package database.

```bash
apt-get update
```
Install the apache2-utils package to get access to ApacheBench.

```bash
apt-get install apache2-utils
```

### macOS

For mac, you will be using 'ab' which is by default installed in macOS. You can test by running ```ab -h```.

Apart from ab, You will also need to have following:

- bc: for metric percentile calculation
- nvidia-docker for gpu machine

## Models

The pre-trained models for the benchmark can be mostly found in the [TorchServe model zoo](https://github.com/pytorch/serve/blob/master/docs/model_zoo.md)

### Benchmarks
We support several basic benchmarks:

- MMS throughput
- MMS latency P50
- MMS latency P90
- MMS latency P99
- MMS latency mean
- MMS HTTP error rate
- Model latency P50
- Model latency P90
- Model latency P99

## Accessing benchmark reports :

The benchmark reports are available at /tmp/benchmark/

## Usage & Examples:

Run benchmark test on resnet-18 model.
It use kitten.jpg image as input from: https://s3.amazonaws.com/model-server/inputs/kitten.jpg 
```bash
./benchmark-ab.sh -u https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar
```

By default, the script will use 1 concurrency and run 1000 requests.  You can change those parameters like this below:
```bash
./benchmark-ab.sh -c 200 -n 2000 -u https://torchserve.s3.amazonaws.com/mar_files/vgg11.mar
```

You can pass `-s` parameter to upload results to S3:
```bash
./benchmark-ab.sh -s -u https://torchserve.s3.amazonaws.com/mar_files/vgg11.mar
```

You can also choose your local docker image to run benchmark
```bash
./benchmark-ab.sh -d pytorch/torchserve:0.1.1-cpu -n 100 -u https://torchserve.s3.amazonaws.com/mar_files/vgg11.mar
```

For batch registration test, first register a model with batch related parameters like this:
```bash
curl -v -X POST "http://localhost:8081/models?initial_workers=1&batch_size=2&max_batch_delay=200&synchronous=true&url=https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar"
```

Then you can choose the exact same values for batch size and batch delay parameters for batch inferencing benchmark as shown below:
```bash
./benchmark-ab.sh --bsize 2 --batch_delay 200 -u https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar
```

The benchmarking script will choose to run on CPU or GPU instance based on presence of -g or --gpu flag.
```bash
./benchmark-ab.sh -g -u https://torchserve.s3.amazonaws.com/mar_files/resnet-18.mar
```
## Troubleshooting Note:
Make sure that you do not have any existing torchserve already running on 8080/8081 ports.



# Profiling

## Frontend

The benchmarks can be used in conjunction with standard profiling tools such as JProfiler to analyze the system performance.  JProfiler can be downloaded from their [website](https://www.ej-technologies.com/products/jprofiler/overview.html).  Once downloaded, open up JProfiler and follow these steps:

1. Run TorchServe directly through gradle (do not use docker).  This can be done either on your machine or on a remote machine accessible through SSH.
2. In JProfiler, select "Attach" from the ribbon and attach to the ModelServer.  The process name in the attach window should be "com.amazonaws.ml.ts.ModelServer".  If it is on a remote machine, select "On another computer" in the attach window and enter the SSH details.  For the session startup settings, you can leave it with the defaults.  At this point, you should see live CPU and Memory Usage data on JProfiler's Telemetries section.
3. Select Start Recordings in JProfiler's ribbon
4. Run the Benchmark script targeting your running TorchServe instance.  It might run something like `./benchmark.py throughput --ts https://127.0.0.1:8443`.  It can be run on either your local machine or a remote machine (if you are running remote), but we recommend running the benchmark on the same machine as the model server to avoid confounding network latencies.
5. Once the benchmark script has finished running, select Stop Recordings in JProfiler's ribbon

Once you have stopped recording, you should be able to analyze the data.  One useful section to examine is CPU views > Call Tree and CPU views > Hot Spots to see where the processor time is going.

## Backend

The benchmarks can also be used to analyze the backend performance using cProfile. To benchmark a backend code, 

1. Enable Benchmarks in TorchServe code with a boolean flag.
2. Install TorchServe with the updated flag & start torchserve.
3. Register a model & perform inference to collect profiling data. This can be done with the benchmark script described in the previous section.
4. Visualize SnakeViz results. 

#### Enable Benchmarks in TorchServe code with a boolean flag

In the file `ts/model_service_worker.py`, set the constant BENCHMARK to true at the top to enable benchmarking.

If running inside docker,

```
    cd docker
    git clone https://github.com/pytorch/serve.git
    cd serve
    ## set BENCHMARK flag to true
    vim ts/model_service_worker.py
    cd ..
```

#### Install TorchServe with the updated flag & Start Torchserve

```
    pip install .
```

If running inside docker

```
    DOCKER_BUILDKIT=1 docker build --file Dockerfile_dev.cpu -t torchserve:dev .
```
then start docker with /tmp directory mapped to local /tmp
    
#### Register a model & perform inference to collect profiling data.

```
python benchmark.py throughput --ts http://127.0.0.1:8080
```

#### Visualize SnakeViz results
 
To visualize the profiling data using `snakeviz` use following commands:

```bash
pip install snakeviz
snakeviz tsPythonProfile.prof
```
![](snake_viz.png)

It should start up a web server on your machine and automatically open the page. Note that tha above command will fail if executed on a server where no browser is installed. The backend profiling should generate a visualization similar to the pic shown above. 
