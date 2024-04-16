This example uses AOTInductor to compile the Resnet50 into an so file which is then executed using libtorch.
The handler C++ source code for this examples can be found [here](src).

**Note**: Please note that due to an issue in Pytorch 2.2.1 the AOTInductor model can not be placed on a specific GPU through the API. This issue is resolved in the PT 2.3 nightlies. Please install the nightlies if you want to run multiple model worker on different GPUs.

### Setup
1. Follow the instructions in [README.md](../../../../cpp/README.md) to build the TorchServe C++ backend.

The build process will create the necessary artifact for this example.
To recreate these by hand you can follow the the [CMakeLists.txt](./CMakeLists.txt) file.
We will need the handler .so file as well as the resnet50_pt2.so file containing the model and weights.

2. Create a [model-config.yaml](model-config.yaml)

```yaml
minWorkers: 1
maxWorkers: 1
batchSize: 2

handler:
  model_so_path: "resnet50_pt2.so"
  mapping: "index_to_name.json"
```

### Generate Model Artifacts Folder

```bash
torch-model-archiver --model-name resnetcppaot --version 1.0 --handler ../../../../cpp/build/test/resources/examples/aot_inductor/resnet_handler/libresnet_handler:ResnetCppHandler --runtime LSP --extra-files index_to_name.json,../../../../cpp/build/test/resources/examples/aot_inductor/resnet_handler/resnet50_pt2.so --config-file model-config.yaml --archive-format no-archive
```

Create model store directory and move the folder `resnetcppaot`

```
mkdir model_store
mv resnetcppaot model_store/
```

### Inference

Start torchserve using the following command

```
export LD_LIBRARY_PATH=`python -c "import torch;from pathlib import Path;p=Path(torch.__file__);print(f\"{(p.parent / 'lib').as_posix()}:{(p.parents[1] / 'nvidia/nccl/lib').as_posix()}\")"`:$LD_LIBRARY_PATH
torchserve --ncs --model-store model_store/ --models resnetcppaot
```


Infer the model using the following command

```
curl http://localhost:8080/predictions/resnetcppaot -T ../../../../cpp/test/resources/examples/aot_inductor/resnet_handler/0_png.pt
{
  "lens_cap": 0.0022578993812203407,
  "lynx": 0.0032067005522549152,
  "Egyptian_cat": 0.046274684369564056,
  "tiger_cat": 0.13740436732769012,
  "tabby": 0.2724998891353607
}
```
