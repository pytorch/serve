.. role:: raw-html-m2r(raw)
   :format: html

.. _install:


Install TorchServe
==================

Prerequisites
-------------


*
  **Python**\ : Required. Model Server for PyTorch (TorchServe) works with Python 3.  When installing TorchServe, we recommend that you use a Python and Conda environment to avoid conflicts with your other Torch installations.

*
  **java 8**\ : Required. TorchServe use java to serve HTTP requests. You must install java 8 (or later) and make sure java is on available in $PATH environment variable *before* installing TorchServe. If you have multiple java installed, you can use $JAVA_HOME environment variable to control which java to use.

For Ubuntu:

.. code-block:: bash

   sudo apt-get install openjdk-11-jdk

For CentOS:

.. code-block:: bash

   openjdk-11-jdk
   sudo yum install java-11-openjdk

For macOS:

.. code-block:: bash

   brew tap AdoptOpenJDK/openjdk
   brew cask install adoptopenjdk11

You can also download and install `Oracle JDK <https://www.oracle.com/technetwork/java/javase/overview/index.html>`_ manually if you have trouble with above commands.


*
  **Torch**\ : Recommended. TorchServe won't install ``torch`` by default. Torch is required for most of examples in this project. TorchServe won't install torch engine by default. And you can also choose specific version of torch if you want.

*
  For virtualenv

.. code-block:: bash

   #For CPU/GPU
   pip install torch torchvision torchtext


* For conda

.. code-block:: bash

   #For CPU
   conda install psutil pytorch torchvision torchtext -c pytorch

.. code-block:: bash

   #For GPU
   conda install future psutil pytorch torchvision cudatoolkit=10.1 torchtext -c pytorch


*
  **Curl**\ : Optional. Curl is used in all of the examples. Install it with your preferred package manager.

*
  **Unzip**\ : Optional. Unzip allows you to easily extract model files and inspect their content. If you choose to use it, associate it with ``.mar`` extensions.

Installation with pip
-----------------------------------

**From PyTorch official repositories**

.. code-block:: bash

   pip install -f https://download.pytorch.org/torch_stable.html torchserve

Installation with conda
-----------------------------------

.. code-block:: bash

   conda install -c pytorch torchserve


Running with Docker
-----------------------------------
CPU only
~~~~~~~~
For specific versions use:

.. code-block:: bash

   docker run --rm -it pytorch/torchserve:0.1-cpu

For the latest version you can use the `latest` tag:

.. code-block:: bash

   docker run --rm -it pytorch/torchserve:latest

GPU
~~~
For the latest version you can use the `gpu-latest` tag:

.. code-block:: bash

   docker run --rm -it --gpus all pytorch/torchserve:latest-gpu

For specific versions use:

.. code-block:: bash

   docker run --rm -it --gpus all pytorch/torchserve:0.1-cuda10.1-cudnn7-runtime


Install TorchServe from Source Code
-----------------------------------

If you prefer, you can clone TorchServe from source code. First, run the following command:

.. code-block:: bash

   git clone https://github.com/pytorch/serve.git
   cd serve
   pip install .

**Notes:**


* In case ``pip install .`` step fails, try using ``python setup.py install`` and install the following python packages using ``pip install`` : Pillow, psutil, future

Install TorchServe for Development
----------------------------------

If you plan to develop with TorchServe and change some of the source code, install it from source code and make your changes executable with this command:

.. code-block:: bash

   pip install -e .

To upgrade TorchServe from source code and make changes executable, run:

.. code-block:: bash

   pip install -U -e .

Troubleshooting Installation
----------------------------

.. list-table::
   :header-rows: 1

   * - Issue
     - Solution
   * - java not found, please make sure JAVA_HOME is set properly.
     - Make sure java is installed. java is on the $PATH or $JAVA_HOME is set properly.
   * - Your PYTHONPATH points to a site-packages dir for Python 3.x but you are running Python 2.x!
     - You do one of following: :raw-html-m2r:`<ul><li>use virtualenv</li><li>unset PYTHONPATH</li><li>set PYTHONPATH properly</li></ul>`
