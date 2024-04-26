## PyTorch 2.x integration

PyTorch 2.x brings more compiler options to PyTorch, for you that should mean better perf either in the form of lower latency or lower memory consumption.

We strongly recommend you leverage newer hardware so for GPUs that would be an Ampere architecture. You'll get even more benefits from using server GPU deployments like A10G and A100 vs consumer cards. But you should expect to see some speedups for any Volta or Ampere architecture.

## Get started

Install torchserve and ensure that you're using at least `torch>=2.0.0`

To use the latest nightlies, you can run the following commands
```sh
python ts_scripts/install_dependencies.py --cuda=cu121 --nightly_torch
pip install torchserve-nightly torch-model-archiver-nightly
```

## torch.compile

PyTorch 2.x supports several compiler backends and you pick which one you want by passing in an optional file `model_config.yaml` during your model packaging

```yaml
pt2: "inductor"
```

You can also pass a dictionary with compile options if you need more control over torch.compile:

```yaml
pt2 : {backend: inductor, mode: reduce-overhead}
```

An example of using `torch.compile` can be found [here](./torch_compile/README.md)

The exact same approach works with any other model, what's going on is the below

```python
# 1. Convert a regular module to an optimized module
opt_mod = torch.compile(mod)
# 2. Train the optimized module
# ....
# 3. Save the opt module state dict
torch.save(opt_model.state_dict(), "model.pt")

# 4. Reload the model
mod = torch.load(model)

# 5. Compile the module and then run inferences with it
opt_mod = torch.compile(mod)
```

torchserve takes care of 4 and 5 for you while the remaining steps are your responsibility. You can do the exact same thing on the vast majority of TIMM or HuggingFace models.

### Compiler Cache

`torch.compile()` is a JIT compiler and JIT compilers generally have a startup cost. To reduce the warm up time, `TorchInductor` already makes use of caching in `/tmp/torchinductor_USERID` of your machine

To persist this cache and /or to make use of additional experimental caching feature, set the following

```
import os

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/path/to/directory"  # replace with your desired path
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
```
An example of how to use these with TorchServe is shown [here](./torch_inductor_caching/)

## torch.export.export

Export your model from a training script, keep in mind that an exported model cannot have graph breaks.

```python
import io
import torch

class MyModule(torch.nn.Module):
    def forward(self, x):
        return x + 10

ep = torch.export.export(MyModule(), (torch.randn(5),))

# Save to file
# torch.export.save(ep, 'exported_program.pt2')
extra_files = {'foo.txt': b'bar'.decode('utf-8')}
torch.export.save(ep, 'exported_program.pt2', extra_files=extra_files)

# Save to io.BytesIO buffer
buffer = io.BytesIO()
torch.export.save(ep, buffer)
```

Serve your exported model from a custom handler

```python
# from initialize()
ep = torch.export.load('exported_program.pt2')

with open('exported_program.pt2', 'rb') as f:
    buffer = io.BytesIO(f.read())
buffer.seek(0)
ep = torch.export.load(buffer)

# Make sure everything looks good
print(ep)
print(extra_files['foo.txt'])

# from inference()
print(ep(torch.randn(5)))
```

## torch._export.aot_compile

Using `torch.compile` to wrap your existing eager PyTorch model can result in out of the box speedups. However, `torch.compile` is a JIT compiler. TorchServe has been supporting `torch.compile` since PyTorch 2.0 release. In a production setting, when you have multiple instances of TorchServe, each of of your instances would `torch.compile` the model on the first inference. TorchServe's model archiver is not able to truly guarantee reproducibility because its a JIT compiler.

In addition, the first inference request with `torch.compile` will be slow as the model needs to compile.

To solve this problem, `torch.export` has an experimental API `torch._export.aot_compile` which is able to `torch.export` a torch compilable model if it has no graph breaks and then run it with AOTInductor

You can find more details [here](https://pytorch.org/docs/main/torch.compiler_aot_inductor.html)


This is an experimental API and needs PyTorch 2.2 nightlies
To achieve this, add the following config in your `model-config.yaml`

```yaml
pt2 :
  export:
    aot_compile: true
```
You can find an example [here](./torch_export_aot_compile/README.md)

## torch.compile GenAI examples

### GPT Fast

GPT-Fast is a simple and efficient pytorch-native transformer text generation which uses `torch.compile`. This model is 10x faster than the baseline llama2 model.

The example can be found [here](../large_models/gpt_fast/README.md)

### Segment Anything Fast

Segment Anything Fast is the optimized version of [Segment Anything](https://github.com/facebookresearch/segment-anything) with 8x performance improvements compared to the original implementation. The improvements were achieved using native PyTorch, primarily `torch.compile`.

The example can be found [here](../large_models/segment_anything_fast/README.md)

### Diffusion Fast

Diffusion Fast is a simple and efficient pytorch-native way of optimizing Stable Diffusion XL (SDXL) with 3x performance improvements compared to the original implementation. This is using `torch.compile`

The example can be found [here](../large_models/diffusion_fast/README.md)

## C++ AOTInductor examples

AOTInductor is the Ahead-of-time-compiler, a specialized version of `TorchInductor`, designed to process exported PyTorch models, optimize them, and produce shared libraries as well as other relevant artifacts. These compiled artifacts are specifically crafted for deployment in non-Python environments. You can find the AOTInductor C++ examples [here](../cpp/aot_inductor)
