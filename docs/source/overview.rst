TorchServe
==========

TorchServe is a flexible and easy to use tool for serving PyTorch
models.

A quick overview and examples for both serving and packaging are
provided below. Detailed documentation and examples are provided in the
`docs folder <docs/README.md>`__.

Contents of this Document
-------------------------

-  `Quick Start <#quick-start>`__
-  `Serve a Model <#serve-a-model>`__
-  `Other Features <#other-features>`__
-  `Contributing <#contributing>`__

Quick Start
-----------

Prerequisites
~~~~~~~~~~~~~

Before proceeding further with this document, make sure you have the
following prerequisites. 1. Ubuntu, CentOS, or macOS. Windows support is
experimental. The following instructions will focus on Linux and macOS
only. 1. Python - TorchServe requires python to run the workers. 1. pip
- Pip is a python package management system. 1. Java 11 - TorchServe
requires Java 11 to start. You have the following options for installing
Java 11:

::

    For Ubuntu:
    ```bash
    sudo apt-get install openjdk-11-jdk
    ```

    For CentOS:
    ```bash
    openjdk-11-jdk
    sudo yum install java-11-openjdk
    ```

    For macOS:
    ```bash
    brew tap AdoptOpenJDK/openjdk
    brew cask install adoptopenjdk11
    ```

Installing TorchServe with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setup
^^^^^

**Step 1:** Setup a Virtual Environment

We recommend installing and running TorchServe in a virtual environment.
It's a good practice to run and install all of the Python dependencies
in virtual environments. This will provide isolation of the dependencies
and ease dependency management.

-  **Use Virtualenv** : This is used to create virtual Python
   environments. You may install and activate a virtualenv for Python
   3.7 as follows:

.. code:: bash

    pip install virtualenv

Then create a virtual environment:

.. code:: bash

    # Assuming we want to run python3.7 in /usr/local/bin/python3.7
    virtualenv -p /usr/local/bin/python3.7 /tmp/pyenv3
    # Enter this virtual environment as follows
    source /tmp/pyenv3/bin/activate

Refer to the `Virtualenv
documentation <https://virtualenv.pypa.io/en/stable/>`__ for further
information.

-  **Use Anaconda** : This is package, dependency and environment
   manager. You may download and install Anaconda as follows : `Download
   anaconda
   distribution <https://www.anaconda.com/distribution/#download-section>`__

Then create a virtual environment using conda.

.. code:: bash

    conda create -n myenv
    source activate myenv

**Step 2:** Install torch TorchServe won't install the PyTorch engine by
default. If it isn't already installed in your virtual environment, you
must install the PyTorch pip packages.

-  For virtualenv

.. code:: bash

    #For CPU/GPU
    pip install torch torchvision torchtext

-  For conda

The ``torchtext`` package has a dependency on ``sentencepiece``, which
is not available via Anaconda. You can install it via ``pip``:

::

    pip install sentencepiece

.. code:: bash

    #For CPU
    conda install psutil pytorch torchvision torchtext -c pytorch

.. code:: bash

    #For GPU
    conda install future psutil pytorch torchvision cudatoolkit=10.1 torchtext -c pytorch

**Step 3:** Install TorchServe as follows:

.. code:: bash

    git clone https://github.com/pytorch/serve.git
    cd serve
    pip install .

**Notes:** \* In case ``pip install .`` step fails, try using
``python setup.py install`` and install the following python packages
using ``pip install`` : Pillow, psutil, future \* See the `advanced
installation <docs/install.md>`__ page for more options and
troubleshooting.

Installing torch-model-archiver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

\*Install torch-model-archiver as follows:

.. code:: bash

    cd serve/model-archiver
    pip install .

**Note** \* Once torch-model-arvchiver is available in Python Package
Index (PyPi), it will be a part of dependency in TorchServe
installation. \* See the `detailed
documentation <model-archiver/README.md>`__ page for more options and
troubleshooting. ### Serve a Model

Once installed, you can get TorchServe model server up and running very
quickly. Try out ``--help`` to see all the CLI options available.

.. code:: bash

    torchserve --help

For this quick start, we'll skip over most of the features, but be sure
to take a look at the `full server docs <docs/server.md>`__ when you're
ready.

Here is an easy example for serving an object classification model:

.. code:: bash

    wget https://download.pytorch.org/models/densenet161-8d451a50.pth
    torch-model-archiver --model-name densenet161 --version 1.0 --model-file examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
    mkdir model_store
    mv densenet161.mar model_store/
    torchserve --start --model-store model_store --models densenet161=densenet161.mar

With the command above executed, you have TorchServe running on your
host, listening for inference requests. **Please note, that if you
specify model(s) during TorchServe start - it will automatically scale
backend workers to the number equal to available vCPUs (if you run on
CPU instance) or to the number of available GPUs (if you run on GPU
instance). In case of powerful hosts with a lot of compute resoures
(vCPUs or GPUs) this start up and autoscaling process might take
considerable time. If you would like to minimize TorchServe start up
time you can try to avoid registering and scaling up model during start
up time and move that to a later point by using corresponding
`Management API <docs/management_api.md#register-a-model>`__ calls (this
allows finer grain control to how much resources are allocated for any
particular model).**

To test it out, you can open a new terminal window next to the one
running TorchServe. Then you can use ``curl`` to download one of these
`cute pictures of a
kitten <https://www.google.com/search?q=cute+kitten&tbm=isch&hl=en&cr=&safe=images>`__
and curl's ``-o`` flag will name it ``kitten.jpg`` for you. Then you
will ``curl`` a ``POST`` to the TorchServe predict endpoint with the
kitten's image.

.. figure:: docs/images/kitten_small.jpg
   :alt: kitten

   kitten
In the example below, we provide a shortcut for these steps.

.. code:: bash

    curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
    curl -X POST http://127.0.0.1:8080/predictions/densenet161 -T kitten.jpg

The predict endpoint will return a prediction response in JSON. It will
look something like the following result:

.. code:: json

    [
      {
        "tiger_cat": 0.46933549642562866
      },
      {
        "tabby": 0.4633878469467163
      },
      {
        "Egyptian_cat": 0.06456148624420166
      },
      {
        "lynx": 0.0012828214094042778
      },
      {
        "plastic_bag": 0.00023323034110944718
      }
    ]

You will see this result in the response to your ``curl`` call to the
predict endpoint, and in the server logs in the terminal window running
TorchServe. It's also being `logged locally with
metrics <docs/metrics.md>`__.

Now you've seen how easy it can be to serve a deep learning model with
TorchServe! `Would you like to know more? <docs/server.md>`__

Stopping the running TorchServe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To stop the current running TorchServe instance, run the following
command:

.. code:: bash

    $ torchserve --stop

You would see output specifying that TorchServe has stopped.

Create a Model Archive
~~~~~~~~~~~~~~~~~~~~~~

TorchServe enables you to package up all of your model artifacts into a
single model archive. This makes it easy to share and deploy your
models. To package a model, check out `model archiver
documentation <model-archiver/README.md>`__

Recommended production deployments
----------------------------------

-  TorchServe doesn't provide authentication. You have to have your own
   authentication proxy in front of TorchServe.
-  TorchServe doesn't provide throttling, it's vulnerable to DDoS
   attack. It's recommended to running TorchServe behind a firewall.
-  TorchServe only allows localhost access by default, see `Network
   configuration <docs/configuration.md#configure-ts-listening-port>`__
   for detail.
-  SSL is not enabled by default, see `Enable
   SSL <docs/configuration.md#enable-ssl>`__ for detail.
-  TorchServe use a config.properties file to configure TorchServe's
   behavior, see `Manage TorchServe <docs/configuration.md>`__ page for
   detail of how to configure TorchServe.

Other Features
--------------

Browse over to the `Docs readme <docs/README.md>`__ for the full index
of documentation. This includes more examples, how to customize the API
service, API endpoint details, and more.

Contributing
------------

We welcome all contributions!

To file a bug or request a feature, please file a GitHub issue. Pull
requests are welcome.

Experimental Release Roadmap
----------------------------

Below, in order, is a prioritized list of tasks for this repository.

v0.1 Plan
~~~~~~~~~

-  [ ] CI (initially AWS CodeBuild)
-  [x] Default handler

   -  [x] Handle eager-mode and TorchScript (tracing and scripting)
   -  [x] Add zero-code pre and post-processing for Image Classification

-  [x] Basic examples

   -  [x] Eager-mode image classifier
   -  [x] TorchScript image classifier
   -  [x] Custom neural network

-  [x] Basic docs (install, serve a model and use it for inference)

v0.2 Plan
~~~~~~~~~

-  [ ] Basic unit tests
-  [ ] Versioning

