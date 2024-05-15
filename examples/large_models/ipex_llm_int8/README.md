# Serving IPEX Optimized Models
This example provides an example of serving IPEX-optimized LLMs e.g. ```meta-llama/llama2-7b-hf``` on huggingface. For setting up the Python environment for this example, please refer here: https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/llm/README.md#3-environment-setup


1. Run the model archiver
```
torch-model-archiver --model-name llama2-7b --version 1.0 --handler llm_handler.py --config-file llama2-7b-int8-woq-config.yaml --archive-format no-archive
```

2. Move the model inside model_store
```
mkdir model_store
mv llama2-7b ./model_store
```

3. Start the torch server
```
torchserve --ncs --start --model-store model_store models llama2-7b
```

5. Test the model status
```
curl http://localhost:8081/models/llama2-7b
```

6. Send the request
```
curl http://localhost:8080/predictions/llama2-7b -T ./sample_text_0.txt
```
## Model Config
In addition to usual torchserve configurations, you need to enable ipex specific optimization arguments.

In order to enable IPEX, ```ipex_enable=true``` in the ```config.parameters``` file. If not enabled it will run with default PyTorch with ```auto_mixed_precision``` if enabled. In order to enable ```auto_mixed_precision```, you need to set ```auto_mixed_precision: true``` in model-config file.

You can choose either Weight-only Quantization or Smoothquant path for quantizing the model to ```INT8```. If the ```quant_with_amp``` flag is set to ```true```, it'll use a mix of ```INT8``` and ```bfloat16``` precisions, otherwise, it'll use ```INT8``` and ```FP32``` combination. If neither approaches are enabled, the model runs on ```bfloat16``` precision by default as long as ```quant_with_amp``` or ```auto_mixed_precision``` is set to ```true```.

There are 3 different example config files; ```model-config-llama2-7b-int8-sq.yaml``` for quantizing with smooth-quant,  ```model-config-llama2-7b-int8-woq.yaml``` for quantizing with weight only quantization, and  ```model-config-llama2-7b-bf16.yaml``` for running the text generation on bfloat16 precision.

### IPEX Weight Only Quantization
<ul>
    <li> weight_type: weight data type for weight only quantization. Options: INT8 or INT4.  
    <li> lowp_mode: low precision mode for weight only quantization. It indicates data type for computation.
</ul>

### IPEX Smooth Quantization

<ul>
    <li> calibration_dataset, and calibration split: dataset and split to be used for calibrating the model quantization
    <li> num_calibration_iters: number of calibration iterations
    <li> alpha: a floating point number between 0.0 and 1.0. For more complex smoothquant config, explore IPEX quantization recipes ( https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/llm/single_instance/run_quantization.py )
</ul>

Set ```greedy``` to true if you want to perform greedy search decoding. If set false, beam search of size 4 is performed by default.
