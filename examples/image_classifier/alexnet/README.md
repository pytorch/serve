### Sample commands to create a alexnet model archive, register it on TorchServe and run image prediction

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

```bash
wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
torch-model-archiver --model-name alexnet --version 1.0 --model-file ./serve/examples/image_classifier/alexnet/model.py --serialized-file alexnet-owt-7be5be79.pth --handler image_classifier --extra-files ./serve/examples/image_classifier/index_to_name.json
mkdir model_store
mv alexnet.mar model_store/
torchserve --start --model-store model_store --models alexnet=alexnet.mar --disable-token-auth  --enable-model-api
curl http://127.0.0.1:8080/predictions/alexnet -T ./serve/examples/image_classifier/kitten.jpg
```

### Serving with Torch Compile
`torch.compile` allows for potential performance improvements when serving the model. It supports a variety of configs and the performance you get can vary based on the config. You can find the various options [here](https://pytorch.org/docs/stable/generated/torch.compile.html).

Use the command below to create a `model-config.yaml` file that will be used in this example:

```
echo "pt2:
  compile:
    enable: True
    backend: inductor
    mode: reduce-overhead" > model-config.yaml
```

##### Create archive and serve model
Ensure your current directory is `examples/image_classifier/alexnet`, then run:

```
wget https://download.pytorch.org/models/alexnet-owt-7be5be79.pth
mkdir model_store
torch-model-archiver --model-name alexnet --version 1.0 --model-file model.py --serialized-file alexnet-owt-7be5be79.pth --handler image_classifier --extra-files ../index_to_name.json --config-file model-config.yaml
mv alexnet.mar model_store/
torchserve --start --model-store model_store --models alexnet=alexnet.mar --disable-token-auth --enable-model-api
```

##### Run Inference
```
curl http://127.0.0.1:8080/predictions/alexnet -T ../kitten.jpg
```
This should output:
```
{
  "tabby": 0.40966343879699707,
  "tiger_cat": 0.346704363822937,
  "Egyptian_cat": 0.13002890348434448,
  "lynx": 0.023919545114040375,
  "bucket": 0.011532172560691833
}
```
If you would like to measure performance, run:
```
echo "handler:
  profile: true" > model-config.yaml
```
to add the relevant config to your model-config.yaml. This will add inference performance metrics (marked as `[METRICS]`) to the output.

### TorchScript example using alexnet image classifier:

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
    torchserve --start --model-store model_store --models alexnet=alexnet.mar --disable-token-auth  --enable-model-api
    curl http://127.0.0.1:8080/predictions/alexnet -T ./serve/examples/image_classifier/kitten.jpg
    ```
