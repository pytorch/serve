TorchServe
==================================
TorchServe is a flexible and easy to use tool for serving PyTorch models.

A quick overview and examples for both serving and packaging are provided below. Detailed documentation and examples are provided in the `docs folder <https://github.com/pytorch/serve/blob/master/docs/README.md>`_.

.. warning ::
     TorchServe is experimental and subject to change.

:ref:`install`

.. toctree::
   :maxdepth: 1
   :caption: Core APIs

   inference_api
   default_handlers
   logging
   management_api
   metrics
   rest_api
   configuration

.. toctree::
   :maxdepth: 1
   :caption: TorchServe Usage

   server
   batch_inference_with_ts
   custom_service
