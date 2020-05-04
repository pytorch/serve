# TorchServe on AWS

The following guide will help you run TorchServe on AWS. You can install TorchServe on Ubuntu 16.04 or 18.04 EC2 instances. You can choose instances with GPUs (p or g-types are recommended) or only use CPUs (c or r-types are recommended). To test TorchServe, you can use a free-tier CPU-only instance type, like t2.medium.

There a few ways you can use TorchServe on AWS:

* The [AWS Deep Learning AMI (DLAMI)](https://docs.aws.amazon.com/dlami/latest/devguide/launch-config.html) on EC2
* If you want fully managed inference, model monitoring, and autoscaling, use [Amazon SageMaker endpoints](https://aws.amazon.com/blogs/machine-learning/deploying-pytorch-models-for-inference-at-scale-using-torchserve/).
* If you prefer Docker or Kubernetes, you can use [AWS Deep Learning Containers](https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/setup.html).

**Note:** Model serving and inference for very large models can be memory intensive, so choose your instance type with enough memory. If you use Amazon SageMaker endpoints, you can scale up or down, out or in, as needed.

## Setup TorchServe on an AWS Deep Learning AMI

1. Spin up an EC2 instance with the latest Deep Learning AMI (DLAMI). This should only take a few minutes.
1. Log in to your instance.
1. Install Java 11
    ```bash
    sudo apt-get install openjdk-11-jdk
    ```
1. Clone the TorchServe repository
    ```bash
    git clone https://github.com/pytorch/serve.git
    cd serve
    ```
1. Choose a Python environment. DLAMI ships with PyTorch already installed in Python 2.7 or 3.6. You can upgrade either of those environments or create a new one. Skip the following create step if you want to upgrade an existing environment.
1. Create a Python environment that you will use for TorchServe.
    - The following names the environment `torchserve_p38`, but you can name it whatever you wish.
    - You can choose a different Python version if you wish.
    ```bash
    conda create -n torchserve_p38 python=3.8.*
    ```
    **Note:** When you run `conda` you may get a notice that `conda` can be upgraded. It is recommended that you upgrade so that you can avoid any issues. You may get a warning that your shell is not configured to use `conda`. Follow the instructions the warning provides on how to fix this issue.
1. Update the environment with TorchServe and its dependencies. Change the name according to the environment you wish to update. This example uses `torchserve_p38`.
    * For a CPU instance
    ```bash
    conda env update -n torchserve_p38 -f environment_cpu.yml
    ```
    * For a GPU instance
    ```bash
    conda env update -n torchserve_p38 -f environment_gpu.yml
    ```
1. Activate the environment
    ```bash
    conda activate torchserve_p38
    ```
1. Test your installation
    * Run  `torchserve`
    ```bash
    torchserve --help
    ```
    * Run `torch-model-archiver`
    ```bash
    torch-model-archiver --help
    ```
Now you are ready to try out some TorchServe examples. You can find image classification, image segmentation, object detection, and text classification examples in the TorchServe repository's `/examples` folder.
