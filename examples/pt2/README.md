## PyTorch 2.x integration

PyTorch 2.0 brings more compiler options to PyTorch, for you that should mean better perf either in the form of lower latency or lower memory consumption. Integrating PyTorch 2.0 is fairly trivial but for now the support will be experimental until the PyTorch 1.14 release.

## Get started

Install torchserve with nightly torch binaries

```
python ts_scripts/dependencies.py --cuda=cu117 --nightly_torch
pip install torchserve torch-model-archiver
```

## Package your model

PyTorch 2.0 supports several compiler backends and you pick which one you want by passing in an optional file `compile.json` during your model packaging

`{"pt2" : "inductor"}`

As an example let's expand our getting started guidde with the only difference being passing in the extra `compile.json` file

```
mkdir model_store
torch-model-archiver --model-name densenet161 --version 1.0 --model-file ./serve/examples/image_classifier/densenet_161/model.py --export-path model_store --extra-files ./serve/examples/image_classifier/index_to_name.json,./serve/examples/image_classifier/compile.json --handler image_classifier
torchserve --start --ncs --model-store model_store --models densenet161.mar
```

The exact same approach works with any other mdoel, what's going on is the beelow

```python
# 1. Convert a regular module to an optimized module
opt_mod = torch.compile(mod)
# 2. Train the optimized module
# ....
# 3. Save the original module (weights are shared)
torch.save(model, "model.pt")  

# 4. Load the non optimized model
mod = torch.load(model)

# 5. Compile the module and then run inferences with it
opt_mod = torch.compile(mod)
```

torchserve takes care of 4 and 5 for you while the remaining steps are your responsibility. You can do the exact same thing on the vast majority of TIMM or HuggingFace models.

## Next steps

For now PyTorch 2.0 has mostly been focused on accelerating training so production grade applications should instead opt for TensorRT for accelerated inference performance which is also natively supported in torchserve. We just wanted to make it really easy for users to experiment with the PyTorch 2.0 stack.
