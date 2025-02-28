# ⚠️ Notice: Limited Maintenance

This project is no longer actively maintained. While existing releases remain available, there are no planned updates, bug fixes, new features, or security patches. Users should be aware that vulnerabilities may not be addressed.

# AMD Support

TorchServe can be run on any combination of operating system and device that is
[supported by ROCm](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility.html).

## Supported Versions of ROCm

The current stable `major.patch` version of ROCm and the previous path version will be supported. For example version `N.2` and `N.1` where `N` is the current major version.

## Installation

  - Make sure you have **python >= 3.8 installed** on your system.
  - clone the repo
    ```bash
    git clone git@github.com:pytorch/serve.git
    ```

  - cd into the cloned folder

    ```bash
    cd serve
    ```

  - create a virtual environment for python

    ```bash
    python -m venv venv
    ```

  - activate the virtual environment. If you use another shell (fish, csh, powershell) use the relevant option in from `/venv/bin/`
    ```bash
    source venv/bin/activate
    ```

  - install the dependencies needed for ROCm support.

    ```bash
    python ./ts_scripts/install_dependencies.py --rocm=rocm61
    python ./ts_scripts/install_from_src.py
    ```
  - enable amd-smi in the python virtual environment
    ```bash
    sudo chown -R $USER:$USER /opt/rocm/share/amd_smi/
    pip install -e /opt/rocm/share/amd_smi/
    ```

### Selecting Accelerators Using `HIP_VISIBLE_DEVICES`

If you have multiple accelerators on the system where you are running TorchServe you can select which accelerators should be visible to TorchServe
by setting the environment variable `HIP_VISIBLE_DEVICES` to a string of 0-indexed comma-separated integers representing the ids of the accelerators.

If you have 8 accelerators but only want TorchServe to see the last four of them do `export HIP_VISIBLE_DEVICES=4,5,6,7`.

>ℹ️  **Not setting** `HIP_VISIBLE_DEVICES` will cause TorchServe to use all available accelerators on the system it is running on.

> ⚠️  You can run into trouble if you set `HIP_VISIBLE_DEVICES` to an empty string.
> eg. `export HIP_VISIBLE_DEVICES=` or `export HIP_VISIBLE_DEVICES=""`
> use `unset HIP_VISIBLE_DEVICES` if you want to remove its effect.

> ⚠️  Setting both `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` may cause unintended behaviour and should be avoided.
> Doing so may cause an exception in the future.

## Docker

**In Development**

`Dockerfile.rocm` provides preliminary ROCm support for TorchServe.

Building and running `dev-image`:

```bash
docker build --file docker/Dockerfile.rocm --target dev-image -t torch-serve-dev-image-rocm --build-arg USE_ROCM_VERSION=rocm62 --build-arg BUILD_FROM_SRC=true .

docker run -it --rm --device=/dev/kfd --device=/dev/dri torch-serve-dev-image-rocm bash
```

## Example Usage

After installing TorchServe with the required dependencies for ROCm you should be ready to serve your model.

For a simple example, refer to `serve/examples/image_classifier/mnist/`.
