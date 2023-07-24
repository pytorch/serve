
# Benchmark ResNet50 and profile the detailed split of PredictionTime

This example shows how to run the benchmark ab tool on ResNet50 and identify the time spent on preprocess, inference and postprocess

Change directory to the root of `serve`
Ex: if `serve` is under `/home/ubuntu`, change directory to `/home/ubuntu/serve`


## Download the weights

```
wget https://download.pytorch.org/models/resnet50-11ad3fa6.pth
```

### Create model archive

To enable profiling of TorchServe Handler, add the following config in model-config.yaml
```
handler:
  profile: true
```

```
torch-model-archiver --model-name resnet-50 --version 1.0 --model-file ./examples/benchmarking/resnet50/model.py --serialized-file resnet50-11ad3fa6.pth --handler image_classifier  --extra-files ./examples/image_classifier/index_to_name.json --config-file ./examples/benchmarking/resnet50/model-config.yaml

mkdir model_store
mv resnet-50.mar model_store/.
```

### Install dependencies for benchmark tool

```
sudo apt-get update -y
sudo apt-get install -y apache2-utils
pip install -r benchmarks/requirements-ab.txt
```

### Run ab tool for benchmarking

```
python benchmarks/auto_benchmark.py --input examples/benchmarking/resnet50/benchmark_profile.yaml --skip true
```

This generates the report under `/tmp/ts_benchmarking/report.md`
