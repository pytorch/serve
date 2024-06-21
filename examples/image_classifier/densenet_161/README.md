#### TorchServe inference with torch.compile of densenet161 model
This example shows how to take eager model of `densenet161`, configure TorchServe to use `torch.compile` and run inference using `torch.compile`

Change directory to the examples directory
`cd  examples/image_classifier/densenet_161`

`torch.compile` supports a variety of config and the performance you get can vary based on the config. You can find the various options [here](https://pytorch.org/docs/stable/generated/torch.compile.html).

Sample command to start torchserve with torch.compile:

```bash
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
mkdir model_store
torch-model-archiver --model-name densenet161 --version 1.0 --model-file model.py --serialized-file densenet161-8d451a50.pth --export-path model_store --extra-files ../../image_classifier/index_to_name.json --handler image_classifier --config-file model-config.yaml -f
torchserve --start --ncs --model-store model_store --models densenet161.mar
curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}
```


#### Sample commands to create a densenet eager mode model archive, register it on TorchServe and run image prediction

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

```bash
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
torch-model-archiver --model-name densenet161 --version 1.0 --model-file examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --handler image_classifier --extra-files examples/image_classifier/index_to_name.json
mkdir model_store
mv densenet161.mar model_store/
torchserve --start --model-store model_store --models densenet161=densenet161.mar
curl http://127.0.0.1:8080/predictions/densenet161 -T examples/image_classifier/kitten.jpg
```

#### TorchScript example using densenet161 image classifier:

* Save the Densenet161 model in as an executable script module or a traced script:

  * Save model using scripting
```python
#scripted mode
from torchvision import models
import torch
model = models.densenet161(pretrained=True)
sm = torch.jit.script(model)
sm.save("densenet161.pt")
```

  * Save model using tracing
```python
#traced mode
from torchvision import models
import torch
model = models.densenet161(pretrained=True)
model.eval()
example_input = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("densenet161.pt")
```

* Use following commands to register Densenet161 torchscript model on TorchServe and run image prediction

```bash
torch-model-archiver --model-name densenet161_ts --version 1.0  --serialized-file densenet161.pt --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
mkdir model_store
mv densenet161_ts.mar model_store/
torchserve --start --model-store model_store --models densenet161=densenet161_ts.mar
curl http://127.0.0.1:8080/predictions/densenet161 -T examples/image_classifier/kitten.jpg
```
