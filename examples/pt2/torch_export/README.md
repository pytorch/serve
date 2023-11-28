# TorchServe inference with torch export program

This example shows how to run TorchServe a Torch exported program

### Pre-requisites

- `PyTorch >= 2.1.0`

Change directory to the root of `serve`
Ex: if `serve` is under `/home/ubuntu`, change directory to `/home/ubuntu/serve`


### Create a Torch exported program

The model is saved with `.pt2` extension

```
python examples/pt2/torch_export/resnet18_torch_export.py
```

### Create model archive

```
torch-model-archiver --model-name res18-pt2 --handler image_classifier --version 1.0 --serialized-file resnet18.pt2 --extra-files ./examples/image_classifier/index_to_name.json
mkdir model_store
mv res18-pt2.mar model_store/.
```

#### Start TorchServe
```
torchserve --start --model-store model_store --models res18-pt2=res18-pt2.mar --ncs
```

#### Run Inference

```
curl http://127.0.0.1:8080/predictions/res18-pt2 -T ./examples/image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.40966305136680603,
  "tiger_cat": 0.34670504927635193,
  "Egyptian_cat": 0.1300286501646042,
  "lynx": 0.023919589817523956,
  "bucket": 0.011532178148627281
}
```
