This example provides an example of serving IPEX-optimized LLMs e.g. ```meta-llama/llama2-7b-hf``` on huggingface. For setting up the Python environment for this example, please refer here: https://github.com/intel/intel-extension-for-pytorch/blob/main/examples/cpu/inference/python/llm/README.md#3-environment-setup 

You can choose either Weight-only Quantization or Smoothquant path for quantizing the model to ```INT8```. If the ```quant_with_amp``` flag is set to ```true```, it'll use a mix of ```INT8``` and ```bfloat16``` precisions, otherwise, it'll use ```INT8``` and ```FP32``` combination. If neither approaches are enabled, the model runs on ```bfloat16``` precision by default as long as ```quant_with_amp``` is set to ```true```. 
There are 3 different example config files; ```model-config-llama2-7b-int8-sq.yaml``` for quantizing with smooth-quant,  ```model-config-llama2-7b-int8-woq.yaml``` for quantizing with weight only quantization, and  ```model-config-llama2-7b-bf16.yaml``` for running the text generation on bfloat16 precision.

1. Zip everything using the model archiver
```
torch-model-archiver --model-name llama2-7b --version 1.0 --handler llm_handler.py --config-file model-config-llama2-7b-int8-woq.yaml 
```

2. Move archive to model_store
```
mkdir model_store
mv llama2-7b.mar ./model_store
```

3. Start the torch server 
```
torchserve --ncs --start --model-store model_store
```

4. From the client, set up batching parameters. I couldn't make it work by putting the max_batch_size and max_batch_delay in config.properties. 
```
curl -X POST "localhost:8081/models?url=llama2-7b.mar&batch_size=4&max_batch_delay=100"
```

5. Test the model status 
```
curl http://localhost:8081/models/llama2-7b
```

6. Send the request
```
curl http://localhost:8080/predictions/llama2-7b -T ./sample_text_0.txt
```
