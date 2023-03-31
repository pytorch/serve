# Micro Batching
Accelerators like GPUs can be used most cost efficiently for inference if they are steadily fed with incoming data.
TorchServe currently allows a single batch to be processed per backend worker.
In each worker the three computation steps (preprocess, inference, postprocess) are executed sequentially.
Because pre- and postprocessing are often carried out on the CPU the GPU sits idle until the batch worker receives a new batch.
The following example will show how to make better use of an accelerator in high load scenarios.

For this we are going to assume that there are a lot of incoming requests and we can potentially fill a bigger batch size within the batch delay timeframe where the frontend collects requests for the next batch.
Given this precondition we are going to increase the batch size the background worker receives and subsequently split the big batch up into smaller *micro* batches to perform the processing.
We can then perform the computation on the micro batches in parallel as more than one batches are available to the worker.

## Implementation
This example implements this design using a custom handler which overwrites the handle method with a MicroBatching object defined in [micro_batching.py](micro_batching.py).
```python
class MicroBatchingHandler(ImageClassifier):
    def __init__(self):
        mb_handle = MicroBatching(self)
        self.handle = mb_handle
```
The MicroBatching object takes the custom handler as an input and spins up a number of threads.
Each thread will work on one of the processing steps (preprocess, inference, postprocess) of the custom handler while multiple threads can be assigned to process micro batches in parallel.
The number of threads as well as the size of the micro batch size is configurable through a json file of the following format:
```json
{
    "micro_batch_size": 4,
    "parallelism": {
        "preprocess": 2,
        "inference": 1,
        "postprocess": 2,
    },
}
```
Each number in the *parallelism* sub-dictionary represents the number of threads created for the respective step on initialization.

## Example
The following example will take a ResNet18 to run the pre- and postprocessing in parallel which includes resizing and cropping the image.

First, we need to download the model weights:
```bash
$ cd <TorchServe main folder>
$ wget https://download.pytorch.org/models/resnet18-f37072fd.pth
```
Second, we create the MAR file while including the necessary source and config files as additional files:
```bash
$ torch-model-archiver --model-name resnet-18_mb --version 1.0 --model-file ./examples/image_classifier/resnet_18/model.py --serialized-file resnet18-f37072fd.pth --handler examples/micro_batching/micro_batching_handler.py --extra-files ./examples/image_classifier/index_to_name.json,examples/micro_batching/micro_batching.py
```
Our MicroBatchingHandler defined in [micro_batching_handler.py](micro_batching_handler.py) inherits from ImageClassifier which already defined the necessary pre- and postprocessing.

Third, we move the MAR file to our model_store and start TorchServe.
```bash
$ mkdir model_store
$ mv resnet-18_mb.mar model_store/
$ torchserve --start --ncs --model_store model_store --models resnet-18_mb
```

Finally, we test the registered model with a request:
```bash
$ <curl send image to model>
```
In the next section we will have a look at how the throughput and latency of the model behave by benchmarking it with TorchServe's benchmark tool.

## Results
For the following benchmark we use [benchmark-ab.py](../../benchmarks/benchmark-ab.py) and a ResNet50 instead of the smaller ResNet18.
We ran this benchmark on an AWS g4dn.4xlarge instance with a single T4 GPU.
After creating the MAR file as described above we extract it into the model_store so we do not need to upload the file.
```bash
$ unzip -d model_store/resnet-50_mb model_store/resnet-50_mb.mar
```
Subsequently, we can run the benchmark with:
```bash
$ python3 benchmarks/benchmark-ab.py --config benchmarks/config.json
```
The config.json for the benchmark has the following content.
```json
{
    "url":"/home/ubuntu/serve/model_store/resnet-50_mb/",
    "requests": 50000,
    "concurrency": 200,
    "input": "/home/ubuntu/serve/examples/image_classifier/kitten.jpg",
    "workers": "1",
    "batch_size": 64
}
```
This will run the model with a batch size of 64 and a micro batch size of 4 as configured in the micro_batching.json.
For this section we ran the benchmark with different batch sizes and micro batch sized (marked with "MBS=X") as well as different number of threads to create the following diagrams.
As a baseline we also ran the vanilla ImageClassifier handler without micro batching which is marked as "NO MB".
![](assets/throughput_latency.png)
In the diagrams we see the throughput and P99 latency plotted over the batch size (as configured through TorchServe API).
Each curve represents a different micro batch size as configured through [micro_batching.json](micro_batching.json).
We can see that the throughput stays flat for the vanilla ImageClassifier (NO MB) which suggests that the inference is preprocessing bound and the GPU is underutilized which can be confirmed with a look at the nvidia-smi output.
By interleaving the three compute steps and using two threads for pre- and prostprocessing we see that the micro batched variant (MBS=4-16) achieves a higher throughput and even a lower batch latency as the GPU is utilized much better through the introduction of micro batches.
For this particular model we can achieve a throughput of up to 250 QPS by increasing the number of preprocessing threads to 4 and chosing 128 and 8 as batch size and micro batch size, respectively.
The actual achieved speedup will depend on the model as well as the intensity of the pre- and postprocessing steps.
Image scaling and decompression is usually more compute intense than text preprocessing.

## Summary
In summary we can see that micro batching can help to increase the throughput of a model while decreasing its latency.
This is especially true for workloads with compute intense pre- or postprocessing as well as smaller models.
The microbatching approach can also be used to save memory in a CPU use case by scaling the number if inference threads to >1 which allows multiple instances of the model which share the underlying weights.
This in contrast to running multiple TorchServe worker which each create their own model instance which can not share their weights as they reside in different processes.
