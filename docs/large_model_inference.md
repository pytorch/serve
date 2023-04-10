# Serving large models with Torchserve

This document explain how Torchserve supports large model serving, here large model refers to the models that are not able to fit into one gpu so they need be splitted in multiple partitions over multiple gpus.

## PiPPY (PyTorch Native solution for large model inference)

PiPPy provides pipeline parallelsim for serving large models that would not fit into one gpu. It takes your model and splits it into equal sizes (stages) partitioned over the number devices you specify. Then uses micro batching to run your batched input for inference ( its is more optimal for batch sizes >1).


## How to use PiPPY in Torchserve

To use Pippy in Torchserve, we need to use a custom handler which inhertis from base_pippy_handler and put our setting in model-config.yaml.

Customer handler in Torchserve is simply a python script that defines model loading, preprocess, inference and postprocess logic specific to your workflow.

It would look like below:

Create `custom_handler.py` or any other descriptive name.

```python
#DO import the necessary packages along with following
from ts.torch_handler.distributed.base_pippy_handler import BasePippyHandler
from ts.handler_utils.distributed.pt_pippy import initialize_rpc_workers, get_pipline_driver
class ModelHandler(BasePippyHandler, ABC):
    def __init__(self):
        super(ModelHandler, self).__init__()
        self.initialized = False

    def initialize(self, ctx):
        model = # load your model from model_dir
        self.device = self.local_rank %  torch.cuda.device_count()# being used to move model inputs to (self.device)
        self.model = get_pipline_driver(model,self.world_size, ctx)

```

Here is what your `model-config.yaml` needs, this config file is very flexible, you can add setting related to frontend, backend and handler.

```bash
#frontend settings
minWorkers: 1
maxWorkers: 1
maxBatchDelay: 100
responseTimeout: 120
parallelLevel: 4
deviceType: "gpu"
parallelType: "pp" #options depending on the solution, pp(pipeline parallelism), tp(tensor parallelism), pptp ( pipeline and tensor parallelism)
                   # This will be used to route input to either rank0 or all ranks from fontend based on the solution (e.g. DeepSpeed support tp, PiPPy support pp)
#backend settings
pippy:
    chunks: 1 # This sets the microbatch sizes, microbatch = batch size/ chunks
    input_names: ['input_ids'] # input arg names to the model, this is required for FX tracing
    model_type: "HF" # set the model type to HF if you are using Huggingface model other wise leave it blank or any other model you use.
    rpc_timeout: 1800
    num_worker_threads: 512 #set number of threads for rpc worker init.

torchrun:
    nproc-per-node: 4 # specifies the number of processes torchrun starts to serve your model, set to world_size or number of
                       # gpus you wish to split your model
handler:
    max_length: 80 # max length of tokens for tokenizer in the handler
```

**How to access it in the handler?** here is an example:

```python
def initialize(self, ctx):
    model_type = ctx.model_yaml_config["pippy"]["model_type"]

```

The rest is as usual in Torchserve, basically packaging your model and starting the server.

Example of the command for packaging your model, make sure you pass model-config.yaml

```bash
torch-model-archiver --model-name bloom --version 1.0 --handler pippy_handler.py --extra-files --extra-files $MODEL_CHECKPOINTS_PATH -r requirements.txt --config-file model-config.yaml --archive-format tgz

```

Tensor Parallel support in progress and will be added as soon as ready.
