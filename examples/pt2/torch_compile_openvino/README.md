
# TorchServe Inference with torch.compile with OpenVINO backend of Resnet50 model

This guide provides steps on how to optimize a ResNet50 model using `torch.compile` with [OpenVINO backend](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html), aiming to enhance inference performance when deployed through TorchServe. `torch.compile` allows for ahead-of-time compilation of PyTorch models, and when combined with OpenVINO, it leverages hardware optimizations that are particularly beneficial for deployment in production environments.

### Prerequisites
- `PyTorch >= 2.1.0`
- `OpenVINO >= 2024.1.0` . Install the latest version as shown below:

```bash
# Install OpenVINO
cd  examples/pt2/torch_compile_openvino
pip install -r requirements.txt
```

## Workflow
1. Configure torch.compile.
1. Create Model Archive.
1. Start TorchServe.
1. Run Inference.
1. Stop TorchServe.
1. Measure and Compare Performance with different backends.

First, navigate to `examples/pt2/torch_compile_openvino`
```bash
cd examples/pt2/torch_compile_openvino
```

### 1. Configure torch.compile

`torch.compile` allows various configurations that can influence performance outcomes. Explore different options in the [official PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.compile.html) and the [OpenVINO backend documentation](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html).


In this example, we use the following config:

```bash
echo "minWorkers: 1
maxWorkers: 2
pt2: {backend: openvino}" > model-config.yaml
```

If you want to measure the handler `preprocess`, `inference`, `postprocess` times, use the following config:

```bash
echo "minWorkers: 1
maxWorkers: 2
pt2: {backend: openvino}
handler:
  profile: true" > model-config.yaml
```

### 2. Create model archive

Download the pre-trained model and prepare the model archive:
```bash
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth
mkdir model_store
torch-model-archiver --model-name resnet-50 --version 1.0 --model-file model.py \
  --serialized-file resnet50-11ad3fa6.pth --export-path model_store \
  --extra-files ../../image_classifier/index_to_name.json --handler image_classifier \
  --config-file model-config.yaml
```

### 3. Start TorchServe

Start the TorchServe server using the following command:
```bash
torchserve --start --ncs --model-store model_store --models resnet-50.mar
```

### 4. Run Inference

**Note:** `torch.compile` requires a warm-up phase to reach optimal performance. Ensure you run at least as many inferences as the `maxWorkers` specified before measuring performance.

```bash
# Open a new terminal
cd  examples/pt2/torch_compile_openvino
curl http://127.0.0.1:8080/predictions/resnet-50 -T ../../image_classifier/kitten.jpg
```

The expected output will be JSON-formatted classification probabilities, such as:

```bash
{
  "tabby": 0.27249985933303833,
  "tiger_cat": 0.13740447163581848,
  "Egyptian_cat": 0.04627467691898346,
  "lynx": 0.0032067003194242716,
  "lens_cap": 0.002257897751405835
}
```

### 5. Stop the server
Stop TorchServe with the following command:

```bash
torchserve --stop
```

### 6. Measure and Compare Performance with different backends

Following the steps outlined in the previous section, you can compare the inference times for Eager mode, Inductor backend, and OpenVINO backend:

1. Update model-config.yaml by adding `profile: true` under the `handler` section.
1. Create a new model archive using torch-model-archiver with the updated configuration.
1. Start TorchServe and run inference.
1. Analyze the TorchServe logs for metrics like `ts_handler_preprocess.Milliseconds`, `ts_handler_inference.Milliseconds`, and `ts_handler_postprocess.Milliseconds`. These metrics represent the time taken for pre-processing, inference, and post-processing steps, respectively, for each inference request.

#### 6.1. Measure inference time with Pytorch Eager mode

Update the `model-config.yaml` file to use Pytorch Eager mode:

```bash
echo "minWorkers: 1
maxWorkers: 2
handler:
  profile: true" > model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.

```bash
2024-05-01T10:29:29,586 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:5.254030227661133|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559369,fd3743e0-9c89-41b2-9972-c1f403872113, pattern=[METRICS]
2024-05-01T10:29:29,609 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:22.122859954833984|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559369,fd3743e0-9c89-41b2-9972-c1f403872113, pattern=[METRICS]
2024-05-01T10:29:29,609 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.057220458984375|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559369,fd3743e0-9c89-41b2-9972-c1f403872113, pattern=[METRICS]
```

#### 6.2. Measure inference time with using `torch.compile` with backend Inductor

Update the model-config.yaml file to specify the Inductor backend:

```bash
echo "minWorkers: 1
maxWorkers: 2
pt2: {backend: inductor, mode: reduce-overhead}
handler:
  profile: true" > model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` requires a warm-up phase to reach optimal performance. Ensure you run at least as many inferences as the `maxWorkers` specified before measuring performance.

```bash
2024-05-01T10:32:05,808 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:5.209445953369141|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559525,9f84ea11-7b77-40e3-bf2c-926746db9c6f, pattern=[METRICS]
2024-05-01T10:32:05,821 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:12.910842895507812|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559525,9f84ea11-7b77-40e3-bf2c-926746db9c6f, pattern=[METRICS]
2024-05-01T10:32:05,822 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.06079673767089844|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714559525,9f84ea11-7b77-40e3-bf2c-926746db9c6f, pattern=[METRICS]
```

#### 6.3. Measure inference time with using `torch.compile` with backend OpenVINO

Update the model-config.yaml file to specify the OpenVINO backend:

```bash
echo "minWorkers: 1
maxWorkers: 2
pt2: {backend: openvino}
handler:
  profile: true" > model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` requires a warm-up phase to reach optimal performance. Ensure you run at least as many inferences as the `maxWorkers` specified before measuring performance.

```bash
2024-05-01T10:40:45,031 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:5.637407302856445|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714560045,7fffdb96-7022-495d-95bb-8dd0b17bf30a, pattern=[METRICS]
2024-05-01T10:40:45,036 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:5.518198013305664|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714560045,7fffdb96-7022-495d-95bb-8dd0b17bf30a, pattern=[METRICS]
2024-05-01T10:40:45,037 [INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.06508827209472656|#ModelName:resnet-50,Level:Model|#type:GAUGE|#hostname:MDSATSM002ARC,1714560045,7fffdb96-7022-495d-95bb-8dd0b17bf30a, pattern=[METRICS]
```

### Conclusion

- Using `torch.compile` with the OpenVINO backend, inference times are reduced to approximately 5.5 ms, a significant improvement from 22 ms with the Eager backend and 13 ms with the Inductor backend. This configuration has been tested on an Intel Xeon Platinum 8469 CPU, showing substantial enhancements in processing speed.

- The actual performance gains may vary depending on your hardware, model complexity, and workload. Consider exploring more advanced `torch.compile` [configurations](https://docs.openvino.ai/2024/openvino-workflow/torch-compile.html) for further optimization based on your specific use case.

- Try out [Stable Diffusion](./stable_diffusion/) example !
