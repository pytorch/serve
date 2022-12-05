## PyTorch 2.x integration

PyTorch 2.0 brings more compiler options to PyTorch, for you that should mean better perf either in the form of lower latency or lower memory consumption. Integrating PyTorch 2.0 is fairly trivial but for now the support will be experimental until the PyTorch 1.14 release.

## Get started

Install torchserve

```
python ts_scripts/dependencies.py --cuda=cu117 --nightly
pip install torchserve
```

## Package your model
