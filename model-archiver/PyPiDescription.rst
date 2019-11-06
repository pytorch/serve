Project Description
===================

Model Archiver is a tool used for creating archives of trained neural
net models that can be consumed by MXNet-Model-Server inference.

Use the Model Archiver CLI to start create a ``.mar`` file.

Model Archiver is part of `MMS <https://pypi.org/project/mxnet-model-server/>`__.
However,you ca install Model Archiver stand alone.

Detailed documentation and examples are provided in the `README
<https://github.com/awslabs/mxnet-model-server/model-archiver/README.md>`__.

Prerequisites
-------------

ONNX support is optional in `model-archiver` tool. It's not installed
by default with `model-archiver`.

If you wish to package a ONNX model, you will need to first install a
``protobuf`` compiler, ``onnx`` and ``mxnet`` manually.

`Instructions for installing Model Archiver with
ONNX <https://github.com/awslabs/mxnet-model-server/blob/master/model-archiver/docs/convert_from_onnx.md#install-model-archiver-with-onnx-support>`__.



Installation
------------

::

    pip install model-archiver

Development
-----------

We welcome new contributors of all experience levels. For information on
how to install MMS for development, refer to the `MMS
docs <https://github.com/awslabs/mxnet-model-server/blob/master/docs/install.md>`__.

Important links
---------------

-  `Official source code
   repo <https://github.com/awslabs/mxnet-model-server>`__
-  `Download
   releases <https://pypi.org/project/mxnet-model-server/#files>`__
-  `Issue
   tracker <https://github.com/awslabs/mxnet-model-server/issues>`__

Source code
-----------

You can check the latest source code as follows:

::

    git clone https://github.com/awslabs/mxnet-model-server.git

Testing
-------

After installation, try out the MMS Quickstart for `Create a
model archive <https://github.com/awslabs/mxnet-model-server/blob/master/README.md#model-archive>`__
and `Serving a
Model <https://github.com/awslabs/mxnet-model-server/blob/master/README.md#serve-a-model>`__.


Help and Support
----------------

-  `Documentation <https://github.com/awslabs/mxnet-model-server/blob/master/docs/README.md>`__
-  `Forum <https://discuss.mxnet.io/latest>`__

Citation
--------

If you use MMS in a publication or project, please cite MMS:
https://github.com/awslabs/mxnet-model-server
