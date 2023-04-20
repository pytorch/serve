## PyTorch 2.x integration

PyTorch 2.0 brings more compiler options to PyTorch, for you that should mean better perf either in the form of lower latency or lower memory consumption. Integrating PyTorch 2.0 is fairly trivial but for now the support will be experimental given that most public benchmarks have focused on training instead of inference.

We strongly recommend you leverage newer hardware so for GPUs that would be an Ampere architecture. You'll get even more benefits from using server GPU deployments like A10G and A100 vs consumer cards. But you should expect to see some speedups for any Volta or Ampere architecture.

## Get started

Install torchserve and ensure that you're using at least `torch>=2.0.0`

```sh
python ts_scripts/install_dependencies.py --cuda=cu117
pip install torchserve torch-model-archiver
```

## Package your model

PyTorch 2.0 supports several compiler backends and you pick which one you want by passing in an optional file `model_config.yaml` during your model packaging

`pt2: "inductor"`

As an example let's expand our getting started guide with the only difference being passing in the extra `model_config.yaml` file

```
mkdir model_store
torch-model-archiver --model-name densenet161 --version 1.0 --model-file ./serve/examples/image_classifier/densenet_161/model.py --export-path model_store --extra-files ./serve/examples/image_classifier/index_to_name.json --handler image_classifier --config-file model_config.yaml
torchserve --start --ncs --model-store model_store --models densenet161.mar
```

The exact same approach works with any other model, what's going on is the below

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

For now PyTorch 2.0 has mostly been focused on accelerating training so production grade applications should instead opt for TensorRT for accelerated inference performance which is also natively supported in torchserve. We just wanted to make it really easy for users to experiment with the PyTorch 2.0 stack. You can learn more here https://github.com/pytorch/serve/blob/master/docs/performance_guide.md
