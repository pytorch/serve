#### Sample commands to create a squeezenet eager mode model archive, register it on TorchServe and run image prediction

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

```bash
wget https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth
torch-model-archiver --model-name squeezenet1_1 --version 1.0 --model-file examples/image_classifier/squeezenet/model.py --serialized-file squeezenet1_1-f364aa15.pth --handler image_classifier --extra-files examples/image_classifier/index_to_name.json
mkdir model_store
mv squeezenet1_1 model_store/
torchserve --start --model-store model_store --models squeezenet1_1=squeezenet1_1.mar
curl http://127.0.0.1:8080/predictions/squeezenet1_1 -T examples/image_classifier/kitten.jpg
```

#### TorchScript example using alexnet image classifier:

* Save the Squeezenet1_1 model as an executable script module or a traced script:

  * Save model using scripting
    ```python
    #scripted mode
    from torchvision import models
    import torch
    model = models.squeezenet1_1(pretrained=True)
    sm = torch.jit.script(model)
    sm.save("squeezenet1_1.pt")
    ```

  * Save model using tracing
    ```python
    #traced mode
    from torchvision import models
    import torch
    model = models.squeezenet1_1(pretrained=True)
    model.eval()
    example_input = torch.rand(1, 3, 224, 224)
    traced_script_module = torch.jit.trace(model, example_input)
    traced_script_module.save("squeezenet1_1.pt")
    ```

* Use following commands to register Squeezenet1_1 torchscript model on TorchServe and run image prediction

```bash
torch-model-archiver --model-name squeezenet1_1 --version 1.0  --serialized-file squeezenet1_1.pt --extra-files examples/image_classifier/index_to_name.json --handler image_classifier
mkdir model_store
mv squeezenet1_1.mar model_store/
torchserve --start --model-store model_store --models squeezenet1_1=squeezenet1_1.mar
curl http://127.0.0.1:8080/predictions/squeezenet1_1 -T examples/image_classifier/kitten.jpg
```
