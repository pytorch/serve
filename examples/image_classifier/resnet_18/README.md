#### Sample commands to create a resnet-18 eager mode model archive, register it on TorchServe and run image prediction

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path

```bash
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
torch-model-archiver --model-name resnet-18 --version 1.0 --model-file ./serve/examples/image_classifier/resnet_18/model.py --serialized-file resnet18-5c106cde.pth --handler image_classifier --extra-files ./serve/examples/image_classifier/index_to_name.json
mkdir model_store
mv resnet-18.mar model_store/
torchserve --start --model-store model_store --models resnet-18=resnet-18.mar
curl http://127.0.0.1:8080/predictions/resnet-18 -T ./serve/examples/image_classifier/kitten.jpg
```

#### TorchScript example using Resnet18 image classifier:

* Save the Resnet18 model in as an executable script module or a traced script:

1. Save model using scripting
   ```python
   #scripted mode
   from torchvision import models
   import torch
   model = models.resnet18(pretrained=True)
   sm = torch.jit.script(model)
   sm.save("resnet-18.pt")
   ```

2. Save model using tracing
   ```python
   #traced mode
   from torchvision import models
   import torch
   model = models.resnet18(pretrained=True)
   model.eval()
   example_input = torch.rand(1, 3, 224, 224)
   traced_script_module = torch.jit.trace(model, example_input)
   traced_script_module.save("resnet-18.pt")
   ```  
 
* Use following commands to register Resnet18 torchscript model on TorchServe and run image prediction

    ```bash
    torch-model-archiver --model-name resnet-18 --version 1.0  --serialized-file resnet-18.pt --extra-files ./serve/examples/image_classifier/index_to_name.json --handler image_classifier
    mkdir model_store
    mv resnet-18.mar model_store/
    torchserve --start --model-store model_store --models resnet-18=resnet-18.mar
    curl http://127.0.0.1:8080/predictions/resnet-18 -T ./serve/examples/image_classifier/kitten.jpg
    ```
