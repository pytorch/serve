# Building TorchServe wheel

To build `TorchServe` wheel use the following command

```
./build_wheels.sh
```

Produced `torchserve`, `torch-model-archiver` and `torch-workflow-archiver` wheel is then stored in the `output` directory

## Retag

A retag lets you easily rename a wheel package from for example a nightly build to an official release without having to rebuild anything.

It's convenient to do something like

`wget https://files.pythonhosted.org/packages/86/f3/18c7796335e1c7a1920662ad65aa36b7331330c32a14747576c2d0fd698f/torchserve_nightly-2022.5.15-py3-none-any.whl`

And then rename `torchserve_nightly-2022.5.15` to `torchserve_0.6.0`

To retag pypi binaries run the these 2 commands:

1. `sudo apt install zip`

2. `NEW_VERSION="my_cool_new_version" ./retag_pypi_binary.sh <path_to_whl_file> <path_to_multiple_whl_files>`
