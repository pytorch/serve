#### Sample commands to create a resnet-152 eager mode model archive for batch inputs, register it on TorchServe and run image prediction

```bash
wget https://download.pytorch.org/models/resnet152-b121ed2d.pth
torch-model-archiver --model-name resnet-152-batch --version 1.0 --model-file serve/examples/image_classifier/resnet_152_batch/model.py --serialized-file resnet152-b121ed2d.pth --handler serve/examples/image_classifier/resnet_152_batch/resnet152_handler.py --extra-files serve/examples/image_classifier/index_to_name.json
mkdir model-store
mv resnet-152-batch.mar model-store/
torchserve --start --model-store model-store --models resnet-152-batch=resnet-152-batch.mar
curl -X POST http://127.0.0.1:8080/predictions/resnet-152-batch -T serve/examples/image_classifier/resnet_152_batch/images
```

#### TorchScript example using Resnet152-batch image classifier:

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
   example_input = torch.rand(1, 3, 224, 224)
   traced_script_module = torch.jit.trace(model, example_input)
   traced_script_module.save("resnet-152-batch.pt")
   ```  
 
* Use following commands to register Resnet152-batch torchscript model on TorchServe and run image prediction

    ```bash
    torch-model-archiver --model-name resnet-152-batch --version 1.0  --serialized-file resnet-152-batch.pt --extra-files serve/examples/image_classifier/index_to_name.json --handler image_classifier
    mkdir model-store
    mv resnet-152-batch.mar model-store/
    torchserve --start --model-store model-store --models resnet-152-batch=resnet-152-batch.mar
    curl -X POST http://127.0.0.1:8080/predictions/resnet-152-batch -T serve/examples/image_classifier/kitten.jpg
    ```