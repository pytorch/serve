#### Sample commands to create a vgg-16 eager mode model archive, register it on TorchServe and run image prediction

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

```bash
wget https://download.pytorch.org/models/vgg16-397923af.pth
torch-model-archiver --model-name vgg16 --version 1.0 --model-file ./examples/image_classifier/vgg_16/model.py --serialized-file vgg16-397923af.pth --handler ./examples/image_classifier/vgg_16/vgg_handler.py --extra-files ./examples/image_classifier/index_to_name.json
mkdir model_store
mv vgg16.mar model_store/
torchserve --start --model-store model_store --models vgg16=vgg16.mar
curl http://127.0.0.1:8080/predictions/vgg16 -T ./examples/image_classifier/kitten.jpg
```

#### TorchScript example using VGG16 image classifier:

* Save the VGG16 model in as an executable script module or a traced script:

1. Save model using scripting
   ```python
   #scripted mode
   from torchvision import models
   import torch
   model = models.vgg16(pretrained=True)
   sm = torch.jit.script(model)
   sm.save("vgg16.pt")
   ```

2. Save model using tracing
   ```python
   #traced mode
   from torchvision import models
   import torch
   model = models.vgg16(pretrained=True)
   model.eval()
   example_input = torch.rand(1, 3, 224, 224)
   traced_script_module = torch.jit.trace(model, example_input)
   traced_script_module.save("vgg16.pt")
   ```  
 
* Use following commands to register vgg16 torchscript model on TorchServe and run image prediction

    ```bash
    torch-model-archiver --model-name vgg16 --version 1.0  --serialized-file vgg16.pt --extra-files ./examples/image_classifier/index_to_name.json --handler ./examples/image_classifier/vgg_16/vgg_handler.py
    mkdir model_store
    mv vgg16.mar model_store/
    torchserve --start --model-store model_store --models vgg16=vgg16.mar
    curl http://127.0.0.1:8080/predictions/vgg16 -T ./serve/examples/image_classifier/kitten.jpg
    ```
