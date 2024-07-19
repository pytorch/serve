#### TorchServe inference with torch.compile of Resnet152 batch image classifier:
Run the commands given in following steps from the root directory of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

First, we create a configuration file where we enable `torch.compile`. You can find more information about the available configuration options [here](https://pytorch.org/docs/stable/generated/torch.compile.html).

```bash
echo "pt2:
  compile:
    enable: True" > examples/image_classifier/resnet_152_batch/model-config.yaml
```

Sample commands to create a Resnet152 torch.compile model archive for batch inputs, register it on TorchServe and run image prediction:

```bash
wget https://download.pytorch.org/models/resnet152-394f9c45.pth
torch-model-archiver --model-name resnet-152-batch --version 1.0 --model-file examples/image_classifier/resnet_152_batch/model.py --serialized-file resnet152-394f9c45.pth --handler image_classifier --extra-files examples/image_classifier/index_to_name.json --config-file examples/image_classifier/resnet_152_batch/model-config.yaml
mkdir model-store
mv resnet-152-batch.mar model-store/
torchserve --start --model-store model-store --disable-token-auth --enable-model-api
curl -X POST "localhost:8081/models?model_name=resnet152&url=resnet-152-batch.mar&batch_size=4&max_batch_delay=5000&initial_workers=3&synchronous=true"
```

And run batched inference:
```bash
curl http://localhost:8080/predictions/resnet152 -T examples/image_classifier/resnet_152_batch/images/croco.jpg &
curl http://localhost:8080/predictions/resnet152 -T examples/image_classifier/resnet_152_batch/images/dog.jpg &
curl http://localhost:8080/predictions/resnet152 -T examples/image_classifier/resnet_152_batch/images/kitten.jpg &
```

#### Sample commands to create a resnet-152 eager mode model archive for batch inputs, register it on TorchServe and run image prediction
Run the commands given in following steps from the root directory of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

```bash
wget https://download.pytorch.org/models/resnet152-394f9c45.pth
torch-model-archiver --model-name resnet-152-batch --version 1.0 --model-file examples/image_classifier/resnet_152_batch/model.py --serialized-file resnet152-394f9c45.pth --handler image_classifier --extra-files examples/image_classifier/index_to_name.json
mkdir model-store
mv resnet-152-batch.mar model-store/
torchserve --start --model-store model-store --disable-token-auth --enable-model-api
curl -X POST "localhost:8081/models?model_name=resnet152&url=resnet-152-batch.mar&batch_size=4&max_batch_delay=5000&initial_workers=3&synchronous=true"
```

The above commands will create the mar file and register the resnet152 model with torchserve with following configuration :

 - model_name : resnet152
 - batch_size : 4
 - max_batch_delay : 5000 ms
 - workers : 3

To test batch inference execute the following commands within the specified max_batch_delay time :

```bash
curl http://127.0.0.1:8080/predictions/resnet152 -T examples/image_classifier/resnet_152_batch/images/croco.jpg &
curl http://127.0.0.1:8080/predictions/resnet152 -T examples/image_classifier/resnet_152_batch/images/dog.jpg &
curl http://127.0.0.1:8080/predictions/resnet152 -T examples/image_classifier/resnet_152_batch/images/kitten.jpg &
```

#### TorchScript example using Resnet152 batch image classifier:

* Save the Resnet152-batch model in as an executable script module or a traced script:

1. Save model using scripting
   ```python
   #scripted mode
   from torchvision import models
   import torch
   model = models.resnet152(pretrained=True)
   sm = torch.jit.script(model)
   sm.save("resnet-152-batch.pt")
   ```

2. Save model using tracing
   ```python
   #traced mode
   from torchvision import models
   import torch
   model = models.resnet152(pretrained=True)
   model.eval()
   example_input = torch.rand(1, 3, 224, 224)
   traced_script_module = torch.jit.trace(model, example_input)
   traced_script_module.save("resnet-152-batch.pt")
   ```

* For batch inference you need to set the batch size while registering the model. This can be done either through the management API or if using Torchserve 0.4.1 and above, it can be set through config.properties as well.  Here is how to register Resnet152-batch torchscript with batch size setting with management API and through config.properties. You can read more on batch inference in Torchserve [here](https://github.com/pytorch/serve/tree/master/docs/batch_inference_with_ts.md).

    * Management API

            ```bash

            torch-model-archiver --model-name resnet-152-batch --version 1.0  --serialized-file resnet-152-batch.pt --extra-files examples/image_classifier/index_to_name.json  --handler image_classifier
            mkdir model_store
            mv resnet-152-batch.mar model_store/
            torchserve --start --model-store model_store --models resnet_152=resnet-152-batch.mar --disable-token-auth  --enable-model-api

            curl -X POST "localhost:8081/models?model_name=resnet152&url=resnet-152-batch.mar&batch_size=4&max_batch_delay=5000&initial_workers=3&synchronous=true"
            ```
    * Config.properties
        ```text
        load_models=resnet-152-batch_v2.mar
        models={\
          "resnet-152-batch_v2": {\
            "2.0": {\
                "defaultVersion": true,\
                "marName": "resnet-152-batch_v2.mar",\
                "minWorkers": 1,\
                "maxWorkers": 1,\
                "batchSize": 3,\
                "maxBatchDelay": 5000,\
                "responseTimeout": 120\
            }\
          }\
        }
        ```
        ```bash
        torchserve --start --model-store model_store  --ts-config config.properties --disable-token-auth  --enable-model-api
        ```
* To test batch inference execute the following commands within the specified max_batch_delay time :

```bash

curl -X POST http://127.0.0.1:8080/predictions/resnet152 -T examples/image_classifier/resnet_152_batch/images/croco.jpg &
curl -X POST http://127.0.0.1:8080/predictions/resnet152 -T examples/image_classifier/resnet_152_batch/images/dog.jpg &
curl -X POST http://127.0.0.1:8080/predictions/resnet152 -T examples/image_classifier/resnet_152_batch/images/kitten.jpg &

```
