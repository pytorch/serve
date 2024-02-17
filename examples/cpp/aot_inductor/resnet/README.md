This example uses AOTInductor to compile the Resnet50 into an so file which is then executed using libtorch.
The handler C++ source code for this examples can be found [here](src).

### Setup
1. Follow the instructions in [README.md](../../../../cpp/README.md) to build the TorchServe C++ backend.

```
cd serve/cpp
./builld.sh
```

The build script will create the necessary artifact for this example.
To recreate these by hand you can follow the prepare_test_files function of the [build.sh](../../../../cpp/build.sh) script.
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

### Generate MAR file

Now lets generate the mar file

```bash
torch-model-archiver --model-name resnetcppaot --version 1.0 --handler ../../../../cpp/_build/test/resources/examples/aot_inductor/resnet_handler/libresnet_handler:ResnetCppHandler --runtime LSP --extra-files index_to_name.json,../../../../cpp/_build/test/resources/examples/aot_inductor/resnet_handler/resnet50_pt2.so --config-file model-config.yaml --archive-format no-archive
```

Create model store directory and move the mar file

```
mkdir model_store
mv resnetcppaot model_store/
```

### Inference

Start torchserve using the following command

```
torchserve --ncs --model-store model_store/
```

Register the model using the following command

```
curl -v -X POST "http://localhost:8081/models?initial_workers=1&url=resnetcppaot&batch_size=2&max_batch_delay=5000"

{
  "status": "Model \"resnetcppaot\" Version: 1.0 registered with 1 initial workers"
}
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
