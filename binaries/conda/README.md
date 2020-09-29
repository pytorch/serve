# Building conda packages

To build conda packages you must first produce wheels for the project, see [this readme](../pip/README.md) for more details on building `TorchServe` wheel.

After producing wheels use the following command to build conda packages:

```
python build_packages.py python build_packages.py --ts-wheel=/path/to/torchserve.whl --ma-wheel=/path/to/torch_model_archiver_wheel
```

Produced conda packages are then stored in the `output` directory
