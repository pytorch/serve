## Examples of Loading Large Models on Multiple GPUS Using DeepSpeed

There are some examples that illustrate how to load large models on multiple GPUs using the DeepSpeed library. The common structure of these examples consists of

* TorchServe model config yaml file: model-config.yaml
* DeepSpeed config jspn file: ds-config.json
* Customer handler based on BaseDeepSpeedHandler: customer_handler.py
* Install DeepSpeed:
  * Method1: requirements.txt
  * Method2: pre-install via command (Recommended to speed up model loading)
    ```
    # See https://www.deepspeed.ai/tutorials/advanced-install/
    DS_BUILD_OPS=1 pip install deepspeed
    ```
