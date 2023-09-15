# Serving CodeGen using TorchServe

### Pre-requisite
1. Setup anaconda virtual environment
```
conda create -n llm_torchserve python=3.9 -y
conda activate llm_torchserve
```

2. Install dependencies
```
conda install -c conda-forge libstdcxx-ng=12
python -m pip install packaging
```

3. Set environment variable
```
export LD_PRELOAD=${CONDA_PREFIX}/lib/libstdc++.so.6
```

4. Install PyTorch 2.2 and IPEX 2.2
```
python -m pip install torch==2.2.0.dev20230911+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
python -m pip install intel_extension_for_pytorch-2.2.0+xxxxxxxx.whl
```
If building from source, please refer to {public llm_feature_branch}. 
	
5. Clone and install TorchServe
```
git clone https://github.com/pytorch/serve.git -b minjean/llm_codegen 
pip install torchserve torch-model-archiver torch-workflow-archiver
```

### Serve CodeGen using TorchServe
1. Install dependencies
```
pip install -r requirements.txt
```

2.  Download model
```
python ../utils/Download_model.py --model_name Salesforce/codegen-2B-multi
```
The script prints the path where the model is downloaded as below. This is an example and in your workload you want to use your actual trained model checkpoints.
```
model/models--Salesforce--codegen-2B-multi/snapshots/c33da754a6605cb4eda7cf7e2b30a6d8bbcd9385/
```

3. Create a `model-config.yaml`
```
minWorkers: 1
maxWorkers: 1

handler:
    model_name: "Salesforce/codegen-2B-multi"
    model_path: "{path/to/torchserve}/serve/examples/large_models/CodeGen/model/models--Salesforce--codegen-2B-multi/snapshots/c33da754a6605cb4eda7cf7e2b30a6d8bbcd9385" # the path to the checkpoints, please change to your model path.
    max_length: 128
    batch_size: 1
```

4. Generate `MAR` file
```
torch-model-archiver --model-name codegen --version 1.0 --handler codegen_handler.py --config-file model-config.yaml
```

5. Add the `MAR` file to model store
```
mkdir model_store
mv codegen.mar model_store
```

6. [Optional] Enable Intel® Extension for PyTorch* optimizations through `config.properties`

If there is a `config.properties` in the working directory, TorchServe loads the `config.properties` file from the current working directory.

Add the following lines in `config.properties`:
```
ipex_enable=true
cpu_launcher_enable=true
# cpu_launcher_args=--node_id 0
```

7. Start TorchServe
```
torchserve --ncs --start --model-store model_store --models codegen.mar
```

8. Run Inference
```
curl http://localhost:8080/predictions/codegen -T ./sample_text_0.txt
```
Sample output:
```
$ curl http://localhost:8080/predictions/codegen -T ./sample_text_0.txt
def hello_world():
    print "Hello World!"

def hello_world_with_args(name):
    print "Hello %s!" % name

def hello_world_with_kwargs(name, age):
    print "Hello %s, %s!" % (name, age)

def hello_world_with_args_and_kwargs(name, age):
    print "Hello %s, %s!" % (name, age)
```

```
curl http://localhost:8080/predictions/codegen -T ./sample_text_1.txt
```
Sample output:  
```
$ curl http://localhost:8080/predictions/codegen -T ./sample_text_1.txt
def random_number_generation():
    return random.randint(0, 100)

def random_string(length):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(length))

def random_string_with_special_chars(length):
return ''.join(random.choice(string.ascii_uppercase + string.digits + string.punctuation) for _ in range(length))
```

## Benchmark with TorchServe 
Additionally, TorchServe provides native [benchmark](https://github.com/pytorch/serve/tree/master/benchmarks) tool to measure performance of TorchServe with various models. 

1. Install dependencies
```
cd {path/to/torchserve}
cd ./benchmarks
pip install -r requirements-ab.txt
sudo apt-get install apache2-utils
```

2. Update `config.properties`

There is a `config.properties` in the benchmark directory that TorchServe loads when running the benchmark. Update to fill your host public IP.
```
inference_address=http://127.0.0.1:8080  
management_address=http://127.0.0.1:8081 
```

3. [Optional] Enable Intel® Extension for PyTorch* optimizations through `config.properties`
```
ipex_enable=true
cpu_launcher_enable=true
# cpu_launcher_args=--node_id 0
```

4. Run benchmark
```
python benchmark-ab.py --url "file:///{path/to/torchserve}/serve/examples/large_models/CodeGen/model_store/codegen.mar" --input ../examples/large_models/CodeGen/sample_text_0.txt
```
