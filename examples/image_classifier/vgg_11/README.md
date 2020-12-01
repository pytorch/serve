#### Sample commands to create a vgg-11 eager mode model archive, register it on TorchServe and run image prediction

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

```bash
wget https://download.pytorch.org/models/vgg11-bbd30ac9.pth
torch-model-archiver --model-name vgg11 --version 1.0 --model-file ./serve/examples/image_classifier/vgg_11/model.py --serialized-file vgg11-bbd30ac9.pth --handler ./serve/examples/image_classifier/vgg_11/vgg_handler.py --extra-files ./serve/examples/image_classifier/index_to_name.json
mkdir model_store
mv vgg11.mar model_store/
torchserve --start --model-store model_store --models vgg11=vgg11.mar
curl http://127.0.0.1:8080/predictions/vgg11 -T ./serve/examples/image_classifier/kitten.jpg
```

#### TorchScript example using VGG11 image classifier:

* Save the VGG11 model in as an executable script module or a traced script:

1. Save model using scripting
   ```python
   #scripted mode
   from torchvision import models
   import torch
   model = models.vgg11(pretrained=True)
   sm = torch.jit.script(model)
   sm.save("vgg11.pt")
   ```

2. Save model using tracing
   ```python
   #traced mode
   from torchvision import models
   import torch
   model = models.vgg11(pretrained=True)
   model.eval()
   example_input = torch.rand(1, 3, 224, 224)
   traced_script_module = torch.jit.trace(model, example_input)
   traced_script_module.save("vgg11.pt")
   ```  
 
* Use following commands to register vgg11 torchscript model on TorchServe and run image prediction

    ```bash
    torch-model-archiver --model-name vgg11 --version 1.0  --serialized-file vgg11.pt --extra-files ./serve/examples/image_classifier/index_to_name.json --handler ./serve/examples/image_classifier/vgg_11/vgg_handler.py
    mkdir model_store
    mv vgg11.mar model_store/
    torchserve --start --model-store model_store --models vgg11=vgg11.mar
    curl http://127.0.0.1:8080/predictions/vgg11 -T ./serve/examples/image_classifier/kitten.jpg
    ```
