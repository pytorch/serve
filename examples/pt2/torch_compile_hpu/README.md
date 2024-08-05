
# TorchServe Inference with torch.compile with HPU backend of Resnet50 model

This guide provides steps on how to optimize a ResNet50 model using `torch.compile` with [HPU backend](https://docs.habana.ai/en/latest/PyTorch/Inference_on_PyTorch/Getting_Started_with_Inference.html), aiming to enhance inference performance when deployed through TorchServe. `torch.compile` allows for JIT compilation of Python code into optimized kernels with a simple API.

### Prerequisites and installation
First install `Intel® Gaudi® AI accelerator software for PyTorch` - Go to [Installation_Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) which covers installation procedures, including software verification and subsequent steps for software installation and management.

Then install the dependencies with the `--skip_torch_install` flag so as not to overwrite habana torch, which you should already have installed. Then install torch-model-archiver, torch-workflow-archiver and torchserve as in the example below.

```bash
git clone https://github.com/pytorch/serve.git
cd serve
python ./ts_scripts/install_dependencies.py --skip_torch_install
pip install torch-model-archiver torch-workflow-archiver
```
Then install torchserve:

- Latest release
``` bash
pip install torchserve
```
- Build from source
``` bash
python ./ts_scripts/install_dependencies.py --skip_torch_install --environment=dev
python ./ts_scripts/install_from_src.py
```


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

In this example, we use the following config that is provided in `model-config.yaml` file:

```bash
echo "minWorkers: 1
maxWorkers: 1
pt2:
  compile:
    enable: True
    backend: hpu_backend" > model-config.yaml
```
Using this configuration will activate the compile mode. Eager mode can be enabled by setting `enable: False` or removing the whole `pt2:` section.

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

`PT_HPU_LAZY_MODE=0` selects `eager+torch.compile` mode. Gaudi integration with PyTorch supports officially 2 modes of operation: `lazy` and `eager+torch.compile (beta state)`. Currently the first one is default, therefore it is necessary to use this flag for compile mode until `eager+torch.compile` mode is set as default. [More information](https://docs.habana.ai/en/latest/PyTorch/Reference/Runtime_Flags.html#pytorch-runtime-flags)

### 3. Start TorchServe

Start the TorchServe server using the following command:
```bash
PT_HPU_LAZY_MODE=0 torchserve --start --ncs --model-store model_store --models resnet-50.mar --disable-token-auth --enable-model-api
```
`--disable-token` - this is an option that disables token authorization. This option is used here only for example purposes. Please refer to the torchserve [documentation](https://github.com/pytorch/serve/blob/master/docs/token_authorization_api.md), which describes the process of serving the model using tokens.

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

### 6. Performance improvement from using `torch.compile`

To measure the handler `preprocess`, `inference`, `postprocess` times, run the following

#### Measure inference time with PyTorch eager

```bash
echo "minWorkers: 1
maxWorkers: 1
handler:
  profile: true" > model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
After a few iterations of warmup, we see the following

```bash
[INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:6.921529769897461|#ModelName:resnet-50,Level:Model|#type:GAUGE|###,1718265363,fe1dcea2-854d-4847-848e-a05e922d456c, pattern=[METRICS]
[INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:5.218982696533203|#ModelName:resnet-50,Level:Model|#type:GAUGE|###,1718265363,fe1dcea2-854d-4847-848e-a05e922d456c, pattern=[METRICS]
[INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:8.724212646484375|#ModelName:resnet-50,Level:Model|#type:GAUGE|###,1718265363,fe1dcea2-854d-4847-848e-a05e922d456c, pattern=[METRICS]
```

#### Measure inference time with `torch.compile`

```bash
echo "minWorkers: 1
maxWorkers: 1
pt2:
  compile:
    enable: True
    backend: hpu_backend
handler:
  profile: true" > model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` needs a few inferences to warmup. Once warmed up, we see the following
```bash
[INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:6.833314895629883|#ModelName:resnet-50,Level:Model|#type:GAUGE|###,1718265582,53da9032-4ad3-49df-8cd4-2d499eea7691, pattern=[METRICS]
[INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:0.7846355438232422|#ModelName:resnet-50,Level:Model|#type:GAUGE|###,1718265582,53da9032-4ad3-49df-8cd4-2d499eea7691, pattern=[METRICS]
[INFO ] W-9000-resnet-50_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:1.9681453704833984|#ModelName:resnet-50,Level:Model|#type:GAUGE|###,1718265582,53da9032-4ad3-49df-8cd4-2d499eea7691, pattern=[METRICS]
```

### Conclusion

`torch.compile` reduces the inference time from 5.22ms to 0.78ms
