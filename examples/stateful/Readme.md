# Stateful Inference

A stateful model possesses the ability to discern interdependencies between successive inference requests. This type of model maintains a persistent state across inference requests, thereby establishing a linkage between the outcomes of prior inquiries and those that follow. Notable illustrations of stateful models encompass online speech recognition systems, such as the Long Short-Term Memory (LSTM) model. Employing stateful inference mandates that the model server adheres to the sequential order of inference requests, ensuring predictions build upon the previous outcomes.

Within this context, TorchServe offers a mechanism known as sequence batching. This approach involves the retrieval of an individual inference request from a particular sequence, followed by the amalgamation of multiple requests originating from diverse sequences into a unified batch. Each request is associated with a unique sequence ID, which can be extracted using the "get_sequence_id" function of context.py. This sequence ID serves as a key employed by custom handlers to store and retrieve values within the backend cache store, fostering efficient management of stateful inference processes.

This example serves as a practical showcase of employing stateful inference. Underneath the surface, the backend leverages an [LRU dictionary](https://github.com/amitdev/lru-dict), functioning as a caching layer.



### Step 1: Implement handler

stateful_handler.py is an example of stateful handler. It creates a cache by calling `[LRU](https://github.com/amitdev/lru-dict)`.

```python
    def initialize(self, ctx: Context):
        """
        Loads the model and Initializes the necessary artifacts
        """

        super().initialize(ctx)
        self.context = ctx
        if self.context.model_yaml_config["handler"] is not None:
            self.cache = LRU(int(self.context.model_yaml_config["handler"]["cache"]["capacity"]))


```

### Step 2: Model configuration

Stateful inference has three parameters.
* sequenceMaxIdleMSec: the max idle in milliseconds of a sequence inference request of this stateful model. The default value is 0 (ie. this is not a stateful model.)
* maxNumSequence: the max number of sequence inference requests of this stateful model. The default value is minWorkers * batchSize.
* maxSequenceJobQueueSize: the job queue size of an inference sequence of this stateful model. The default value is 1.


```yaml
#cat model-config.yaml

minWorkers: 2
maxWorkers: 2
batchSize: 4
sequenceMaxIdleMSec: 60000
maxNumSequence: 4
maxSequenceJobQueueSize: 10

handler:
  cache:
    capacity: 4
```

### Step 3: Generate mar or tgz file

```bash
torch-model-archiver --model-name stateful --version 1.0 --handler stateful_handler.py -r requirements.txt --config-file model-config.yaml
```

### Step 4: Start torchserve

```bash
torchserve --start --ncs --model-store model_store --models stateful.mar
```

### Step 6: Build GRPC Client
The details can be found at [here](https://github.com/pytorch/serve/blob/master/docs/grpc_api.md).
* Install gRPC python dependencies
* Generate python gRPC client stub using the proto files

### Step 7: Run inference
* Start TorchServe

```bash
torchserve --ncs --start --model-store models --model stateful.mar --ts-config config.properties
```

* Run sequence infernce
```bash
cd ../../
python ts_scripts/torchserve_grpc_client.py  infer_stream2 stateful seq_0 examples/stateful/sample/sample1.txt,examples/stateful/sample/sample2.txt,examples/stateful/sample/sample3.txt
```
