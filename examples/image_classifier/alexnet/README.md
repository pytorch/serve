#### Sample commands to create a alexnet eager mode model archive, register it on TorchServe and run image prediction

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

```bash
wget https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
torch-model-archiver --model-name alexnet --version 1.0 --model-file ./serve/examples/image_classifier/alexnet/model.py --serialized-file alexnet-owt-4df8aa71.pth --handler image_classifier --extra-files ./serve/examples/image_classifier/index_to_name.json
mkdir model_store
mv alexnet model_store/
torchserve --start --model-store model_store --models alexnet=alexnet.mar
curl http://127.0.0.1:8080/predictions/alexnet -T ./serve/examples/image_classifier/kitten.jpg
```

#### TorchScript example using alexnet image classifier:

* Save the alexnet model in as an executable script module or a traced script:

1. Save model using scripting
   ```python
   #scripted mode
   from torchvision import models
   import torch
   model = models.alexnet(pretrained=True)
   sm = torch.jit.script(model)
   sm.save("alexnet.pt")
   ```

2. Save model using tracing
   ```python
   #traced mode
   from torchvision import models
   import torch
   model = models.alexnet(pretrained=True)
   model.eval()
   example_input = torch.rand(1, 3, 224, 224)
   traced_script_module = torch.jit.trace(model, example_input)
   traced_script_module.save("alexnet.pt")
   ```  
 
* Use following commands to register alexnet torchscript model on TorchServe and run image prediction

    ```bash
    torch-model-archiver --model-name alexnet --version 1.0  --serialized-file alexnet.pt --extra-files ./serve/examples/image_classifier/index_to_name.json --handler image_classifier
    mkdir model_store
    mv alexnet.mar model_store/
    torchserve --start --model-store model_store --models alexnet=alexnet.mar
    curl http://127.0.0.1:8080/predictions/alexnet -T ./serve/examples/image_classifier/kitten.jpg
    ```
