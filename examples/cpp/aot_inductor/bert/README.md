This example uses AOTInductor to compile the [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) into an so file (see script [aot_compile_export.py](aot_compile_export.py)). In PyTorch 2.2, the supported `MAX_SEQ_LENGTH` in this script is 511.

Then, this example loads model and runs prediction using libtorch. The handler C++ source code for this examples can be found [here](src).

**Note**: Please note that due to an issue in Pytorch 2.2.1 the AOTInductor model can not be placed on a specific GPU through the API. This issue is resolved in the PT 2.3 nightlies. Please install the nightlies if you want to run multiple model worker on different GPUs.

### Setup
1. Follow the instructions in [README.md](../../../../cpp/README.md) to build the TorchServe C++ backend.

```
cd serve/cpp
./builld.sh
```

The build script will create the necessary artifact for this example.
To recreate these by hand you can follow the prepare_test_files function of the [build.sh](../../../../cpp/build.sh) script.
We will need the handler .so file as well as the bert-seq.so and tokenizer.json.

2. Create a [model-config.yaml](model-config.yaml)

```yaml
minWorkers: 1
maxWorkers: 1
batchSize: 2

handler:
  model_so_path: "bert-seq.so"
  tokenizer_path: "tokenizer.json"
  mapping: "index_to_name.json"
  model_name: "bert-base-uncased"
  mode: "sequence_classification"
  do_lower_case: true
  num_labels: 2
  max_length: 150
```

### Generate Model Artifact Folder

```bash
torch-model-archiver --model-name bertcppaot --version 1.0 --handler ../../../../cpp/_build/test/resources/examples/aot_inductor/bert_handler/libbert_handler:BertCppHandler --runtime LSP --extra-files index_to_name.json,../../../../cpp/_build/test/resources/examples/aot_inductor/bert_handler/bert-seq.so,../../../../cpp/_build/test/resources/examples/aot_inductor/bert_handler/tokenizer.json  --config-file model-config.yaml --archive-format no-archive
```

Create model store directory and move the folder `bertcppaot`

```
mkdir model_store
mv bertcppaot model_store/
```

### Inference

Start torchserve using the following command

```
export LD_LIBRARY_PATH=`python -c "import torch;from pathlib import Path;p=Path(torch.__file__);print(f\"{(p.parent / 'lib').as_posix()}:{(p.parents[1] / 'nvidia/nccl/lib').as_posix()}\")"`:$LD_LIBRARY_PATH
torchserve --ncs --model-store model_store/ --models bertcppaot
```

Infer the model using the following command

```
curl http://localhost:8080/predictions/bertcppaot -T ../../../../cpp/test/resources/examples/aot_inductor/bert_handler/sample_text.txt
Not Accepted
```
