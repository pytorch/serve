This example demonstrates how to run code generation model e.g., Salesforce/codegen25-7b-multi. We are using IPEX Weight Only Quantization to convert the model to INT8.

For setting the conda environment for IPEX WoQ check out the documentation here:
 
https://github.com/intel/intel-extension-for-pytorch/tree/main/examples/cpu/inference/python/llm


1. Zip everything using model archiver
```
 torch-model-archiver --model-name codegen25 --version 1.0 --handler codegen_handler.py --config-file model-config.yaml 
```

2. Move archive to model_store
```
mkdir model_store
mv codegen25.mar ./model_store
```
3. Start the torch server 
```
torchserve --ncs --start --model-store model_store
```

4. From the client, set up batching parameters. I couldn't make it work by putting the max_batch_size and max_batch_delay in config.properties. 
```
curl -X POST "localhost:8081/models?url=codegen25.mar&batch_size=4&max_batch_delay=500"
```

5. Test the model status 
```
curl http://localhost:8081/models/codegen25
```

6. Send the request
```
curl http://localhost:8080/predictions/codegen25 -T ./sample_text_0.txt
```

7. Batching the requests
```
bash benchmark.sh _batch_size
 e.g., bash benchmark.sh 4
```



