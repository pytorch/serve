
# TorchServe Inference with torch.compile with HPU backend of Resnet50 model

This guide provides steps on how to optimize a ResNet50 model using `torch.compile` with [HPU backend](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Getting_Started_with_Inference.html), aiming to enhance inference performance when deployed through TorchServe. `torch.compile` allows for ahead-of-time compilation of PyTorch models.

### Prerequisites
- `Intel® Gaudi® AI accelerator software for PyTorch` - Go to [Installation_Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) which covers installation procedures, including software verification and subsequent steps for software installation and management.

## Workflow
1. Configure torch.compile.
2. Create model archive.
3. Start TorchServe.
4. Run Inference.
5. Stop TorchServe.

First, navigate to `examples/pt2/torch_compile_hpu`
```bash
cd examples/pt2/torch_compile_hpu
```

### 1. Configure torch.compile

`torch.compile` allows various configurations that can influence performance outcomes. Explore different options in the [official PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.compile.html)


In this example, we use the following config that is provided in `model-config.yaml` file:

```yaml
minWorkers: 1
maxWorkers: 1
pt2: {backend: "hpu_backend"}
```
`pt2: {backend: "hpu_backend"}` - this line enables compile mode, if you remove it from the config file, the model will run in eager mode.

### 2. Create model archive

Download the pre-trained model and prepare the model archive:
```bash
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth
mkdir model_store
PT_HPU_LAZY_MODE=0 torch-model-archiver --model-name resnet-50 --version 1.0 --model-file model.py \
  --serialized-file resnet50-11ad3fa6.pth --export-path model_store \
  --extra-files ../../image_classifier/index_to_name.json --handler hpu_image_classifier.py \
  --config-file model-config.yaml
```

### 3. Start TorchServe

Start the TorchServe server using the following command:
```bash
PT_HPU_LAZY_MODE=0 torchserve --start --ncs --model-store model_store --models resnet-50.mar
```

### 4. Run Inference

**Note:** `torch.compile` requires a warm-up phase to reach optimal performance. Ensure you run at least as many inferences as the `maxWorkers` specified before measuring performance.

```bash
# Open a new terminal
cd  examples/pt2/torch_compile_hpu
curl http://127.0.0.1:8080/predictions/resnet-50 -T ../../image_classifier/kitten.jpg
```

The expected output will be JSON-formatted classification probabilities, such as:

```json
{
  "tabby": 0.2724992632865906,
  "tiger_cat": 0.1374046504497528,
  "Egyptian_cat": 0.046274710446596146,
  "lynx": 0.003206699388101697,
  "lens_cap": 0.002257900545373559
}
```

### 5. Stop the server
Stop TorchServe with the following command:

```bash
torchserve --stop
```
