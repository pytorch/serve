
# TorchInductor Caching with TorchServe inference of densenet161 model

`torch.compile()` is a JIT compiler and JIT compilers generally have a startup cost. To handle this, `TorchInductor` already makes use of caching in `/tmp/torchinductor_USERID` of your machine

## TorchInductor FX Graph Cache
There is an experimental feature to cache FX Graph as well. This is not enabled by default and can be set with the following config

```
import os
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
```

This needs to be set before you `import torch`

or

```
import torch

torch._inductor.config.fx_graph_cache = True
```

To see the effect of caching on `torch.compile` execution times, we need to have a multi worker setup. In this example, we use 4 workers. Workers 2,3,4 will see the benefit of caching when they execute `torch.compile`

We show below how this can be used with TorchServe


### Pre-requisites

- `PyTorch >= 2.2`

Change directory to the examples directory
Ex:  `cd  examples/pt2/torch_inductor_caching`


### torch.compile config

`torch.compile` supports a variety of config and the performance you get can vary based on the config. You can find the various options [here](https://pytorch.org/docs/stable/generated/torch.compile.html)

In this example , we use the following config

```yaml
pt2 : {backend: inductor, mode: max-autotune}
```

### Create model archive

```
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
mkdir model_store
torch-model-archiver --model-name densenet161 --version 1.0 --model-file ../../image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --export-path model_store --extra-files ../../image_classifier/index_to_name.json --handler ./caching_handler.py --config-file model-config-fx-cache.yaml -f
```

#### Start TorchServe
```
torchserve --start --ncs --model-store model_store --models densenet161.mar
```

#### Run Inference

```
curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg && curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg && curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg && curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}
```

## TorchInductor Cache Directory
`TorchInductor` already makes use of caching in `/tmp/torchinductor_USERID` of your machine.

Since the default directory is in `/tmp`, the cache is deleted on restart

`torch.compile` provides a config to change the cache directory for `TorchInductor `

```
import os

os.environ["TORCHINDUCTOR_CACHE_DIR"] =  "/path/to/directory"  # replace with your desired path

```


We show below how this can be used with TorchServe


### Pre-requisites

- `PyTorch >= 2.2`

Change directory to the examples directory
Ex:  `cd  examples/pt2/torch_inductor_caching`


### torch.compile config

`torch.compile` supports a variety of config and the performance you get can vary based on the config. You can find the various options [here](https://pytorch.org/docs/stable/generated/torch.compile.html)

In this example , we use the following config

```yaml
pt2 : {backend: inductor, mode: max-autotune}
```

### Create model archive

```
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
mkdir model_store
torch-model-archiver --model-name densenet161 --version 1.0 --model-file ../../image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --export-path model_store --extra-files ../../image_classifier/index_to_name.json --handler ./caching_handler.py --config-file model-config-cache-dir.yaml -f
```

#### Start TorchServe
```
torchserve --start --ncs --model-store model_store --models densenet161.mar
```

#### Run Inference

```
curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg && curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg && curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg && curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}
```

## Additional links for improving `torch.compile` performance and debugging

- [Compile Threads](https://pytorch.org/blog/training-production-ai-models/#34-controlling-just-in-time-compilation-time)
- [Profiling torch.compile](https://pytorch.org/docs/stable/torch.compiler_profiling_torch_compile.html)
