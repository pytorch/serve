#### Eager Mode example using torchvision image classifiers:

* TorchVision Image Classification Models : Download a pre-trained model state_dict for computer vision model that classifies images from the following :

  * [Image Classification with AlexNet](alexnet) - https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
  * [Image Classification with DenseNet161](densenet_161) - https://download.pytorch.org/models/densenet161-8d451a50.pth
  * [Image Classification with ResNet18](resnet_18) - https://download.pytorch.org/models/resnet18-5c106cde.pth
  * [Image Classification with SqueezeNet 1_1](squeezenet) - https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth
  * [Image Classification with VGG11](vgg_11) - https://download.pytorch.org/models/vgg11-bbd30ac9.pth

* [Create a model archive file](../../model-archiver/README.md) based on selected model or use the sample provided with above examples.

* You can use the [index_to_name.json](index_to_name.json) file for mapping predicted index to class or use a custom one.

* Create a torch model archive file using the above provided syntax command.

#### Sample commands to create a DenseNet161 eager mode model archive, register it on TorchServe and run image prediction

    ```bash
    wget https://download.pytorch.org/models/densenet161-8d451a50.pth
    torch-model-archiver --model-name densenet161 --version 1.0 --model-file examples/image_classifier/densenet_161/model.py --serialized-file densenet161-8d451a50.pth --handler image_classifier --extra-files examples/image_classifier/index_to_name.json
    mkdir model_store
    mv densenet161.mar model_store/
    torchserve --start --model-store model_store --models densenet161=densenet161.mar
    curl http://127.0.0.1:8080/predictions/densenet161 -T examples/image_classifier/kitten.jpg


#### TorchScript example using DenseNet161 image classifier:

* Save the Densenet161 model in as an executable script module or a traced script:

1. Save model using scripting
   ```python
   #scripted mode
   from torchvision import models
   import torch
   model = models.densenet161(pretrained=True)
   sm = torch.jit.script(model)
   sm.save("densenet161.pt")
   ```

2. Save model using tracing
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
#### TorchScript example using custom model and custom handler:

Following example demonstrates how to create and serve a custom NN model with custom handler archives in TorchServe :

* [Digit recognition with MNIST](mnist)
