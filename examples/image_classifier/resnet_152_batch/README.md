#### Sample commands to create a resnet-152 eager mode model archive for batch inputs, register it on TorchServe and run image prediction
Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path

```bash
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth
torch-model-archiver --model-name resnet-152-batch --version 1.0 --model-file ./serve/examples/image_classifier/resnet_152_batch/model.py --serialized-file resnet152-b121ed2d.pth --handler ./serve/examples/image_classifier/resnet_152_batch/resnet152_handler.py --extra-files ./serve/examples/image_classifier/index_to_name.json
mkdir model-store
mv resnet-152-batch.mar model-store/
torchserve --start --model-store model-store
curl -X POST "localhost:8081/models?model_name=resnet152&url=resnet-152-batch.mar&batch_size=4&max_batch_delay=5000&initial_workers=3&synchronous=true"
```

The above commands will create the mar file and register the resnet152 model with torchserve with following configuration :

 - model_name : resnet152
 - batch_size : 4
 - max_batch_delay : 5000 ms
 - workers : 3

To test batch inference execute the following commands within the specified max_batch_delay time :

```bash
curl http://127.0.0.1:8080/predictions/resnet152 -T ./serve/examples/image_classifier/resnet_152_batch/images/croco.jpg &
curl http://127.0.0.1:8080/predictions/resnet152 -T ./serve/examples/image_classifier/resnet_152_batch/images/dog.jpg &
curl http://127.0.0.1:8080/predictions/resnet152 -T ./serve/examples/image_classifier/resnet_152_batch/images/kitten.jpg &
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

* Use following commands to register Resnet152-batch torchscript model on TorchServe and run image prediction

    ```bash

    torch-model-archiver --model-name resnet-152-batch --version 1.0  --serialized-file resnet-152-batch.pt --extra-files serve/examples/image_classifier/index_to_name.json  --handler serve/examples/image_classifier/resnet_152_batch/resnet152_handler.py
    mkdir model_store
    mv resnet-152-batch.mar model_store/
    torchserve --start --model-store model_store --models resnet_152=resnet-152-batch.mar

    curl -X POST "localhost:8081/models?model_name=resnet152&url=resnet-152-batch.mar&batch_size=4&max_batch_delay=5000&initial_workers=3&synchronous=true"
    ```

* To test batch inference execute the following commands within the specified max_batch_delay time :

```bash

curl -X POST http://127.0.0.1:8080/predictions/resnet152 -T serve/examples/image_classifier/resnet_152_batch/images/croco.jpg &
curl -X POST http://127.0.0.1:8080/predictions/resnet152 -T serve/examples/image_classifier/resnet_152_batch/images/dog.jpg &
curl -X POST http://127.0.0.1:8080/predictions/resnet152 -T serve/examples/image_classifier/resnet_152_batch/images/kitten.jpg &

```
