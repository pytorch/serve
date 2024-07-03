#### TorchServe inference with torch.compile of vgg-16 model
This example shows how to take eager model of `vgg-16`, configure TorchServe to use `torch.compile` and run inference using `torch.compile`

Change directory to the root directory of this project.

`torch.compile` supports a variety of config and the performance you get can vary based on the config. You can find the various options [here](https://pytorch.org/docs/stable/generated/torch.compile.html).

Sample command to start torchserve with torch.compile:

```bash
wget https://download.pytorch.org/models/vgg16-397923af.pth
torch-model-archiver --model-name vgg16 --version 1.0 --model-file ./examples/image_classifier/vgg_16/model.py --serialized-file vgg16-397923af.pth --handler ./examples/image_classifier/vgg_16/vgg_handler.py --extra-files ./examples/image_classifier/index_to_name.json --config-file ./examples/image_classifier/vgg_16/model-config.yaml -f
mkdir model_store
mv vgg16.mar model_store/vgg16_compiled.mar
torchserve --start --model-store model_store --models vgg16=vgg16_compiled.mar --disable-token-auth  --enable-model-api
```

Now in another terminal, run

```bash
curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg
```

which produces the output

```
{
  "tiger_cat": 0.44697248935699463,
  "tabby": 0.4408799707889557,
  "Egyptian_cat": 0.05904558673501015,
  "tiger": 0.02059638872742653,
  "lynx": 0.009934586472809315
}
```

#### Sample commands to create a vgg-16 eager mode model archive, register it on TorchServe and run image prediction

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

```bash
wget https://download.pytorch.org/models/vgg16-397923af.pth
torch-model-archiver --model-name vgg16 --version 1.0 --model-file ./examples/image_classifier/vgg_16/model.py --serialized-file vgg16-397923af.pth --handler ./examples/image_classifier/vgg_16/vgg_handler.py --extra-files ./examples/image_classifier/index_to_name.json
mkdir model_store
mv vgg16.mar model_store/
torchserve --start --model-store model_store --models vgg16=vgg16.mar --disable-token-auth  --enable-model-api
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
    torchserve --start --model-store model_store --models vgg16=vgg16.mar --disable-token-auth  --enable-model-api
    curl http://127.0.0.1:8080/predictions/vgg16 -T ./serve/examples/image_classifier/kitten.jpg
    ```
