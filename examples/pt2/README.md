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

## Next steps

For now PyTorch 2.0 has mostly been focused on accelerating training so production grade applications should instead opt for TensorRT for accelerated inference performance which is also natively supported in torchserve. We just wanted to make it really easy for users to experiment with the PyTorch 2.0 stack.
