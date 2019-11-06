Project Description
===================

TorchServe (PyTorch mdoel server) is a flexible and easy to use tool for
serving deep learning models exported from `PyTorch <http://pytorch.org/>`__.

Use the TorchServe CLI, or the pre-configured Docker images, to start a
service that sets up HTTP endpoints to handle model inference requests.

Prerequisites
-------------

* **java 8**: Required. TorchServe use java to serve HTTP requests. You must install java 8 (or later) and make sure java is on available in $PATH environment variable *before* installing torchserve. If you have multiple java installed, you can use $JAVA_HOME environment vairable to control which java to use.
* **PyTorch**: Required. Latest version of PyTorch will be installed as a part of TorchServe installation.

For ubuntu:
::

    sudo apt-get install openjdk-8-jre-headless


For centos
::

    sudo yum install java-1.8.0-openjdk


For Mac:
::

    brew tap caskroom/versions
    brew update
    brew cask install java8


Install PyTorch:
::

    pip install torch


Installation
------------

::

    pip install torchserve


Source code
-----------

You can check the latest source code as follows:

::

    git clone https://github.com/pytorch/serve.git

Citation
--------

If you use torchserve in a publication or project, please cite torchserve:
https://github.com/pytorch/serve
