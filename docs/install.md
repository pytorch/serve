
# Install TorchServe

## Prerequisites

* **Python**: Required. Model Server for PyTorch (TorchServe) works with Python 2 or 3.  When installing TorchServe, we recommend that you use a Python and Conda environment to avoid conflicts with your other Torch installations.

* **java 8**: Required. TorchServe use java to serve HTTP requests. You must install java 8 (or later) and make sure java is on available in $PATH environment variable *before* installing TorchServe. If you have multiple java installed, you can use $JAVA_HOME environment variable to control which java to use.

For ubuntu:
```bash
sudo apt-get install openjdk-8-jre-headless
```

For centos
```bash
sudo yum install java-1.8.0-openjdk
```

For Mac:
```bash
brew tap caskroom/versions
brew update
brew cask install java8
```

You can also download and install [Oracle JDK](https://www.oracle.com/technetwork/java/javase/overview/index.html) manually if you have trouble with above commands.

* **Torch**: Recommended. TorchServe won't install `torch` by default. Torch is required for most of examples in this project. TorchServe won't install torch engine by default. And you can also choose specific version of torch if you want.

```bash
pip install torch torchvision
```

* **Curl**: Optional. Curl is used in all of the examples. Install it with your preferred package manager.

* **Unzip**: Optional. Unzip allows you to easily extract model files and inspect their content. If you choose to use it, associate it with `.mar` extensions.


## Install TorchServe from Source Code

If you prefer, you can clone TorchServe from source code. First, run the following command:

```bash
git clone https://github.com/pytorch/serve.git
cd serve
python setup.py install
```

## Install TorchServe for Development

If you plan to develop with TorchServe and change some of the source code, install it from source code and make your changes executable with this command:

```bash
pip install -e .
```

To upgrade TorchServe from source code and make changes executable, run:


```bash
pip install -U -e .
```

## Troubleshooting Installation


| Issue | Solution |
|---|---|
|java not found, please make sure JAVA_HOME is set properly. | Make sure java is installed. java is on the $PATH or $JAVA_HOME is set properly. |
|Your PYTHONPATH points to a site-packages dir for Python 3.x but you are running Python 2.x! | You do one of following: <ul><li>use virtualenv</li><li>unset PYTHONPATH</li><li>set PYTHONPATH properly</li></ul> |
