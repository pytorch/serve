## Llama.cpp example

This example used [llama.cpp](https://github.com/ggerganov/llama.cpp) to deploy a Llama-2-7B-Chat model using the TorchServe C++ backend.
The handler C++ source code for this examples can be found [here](./src/).

### Setup
1. Follow the instructions in [README.md](../../../cpp/README.md) to build the TorchServe C++ backend.

```bash
cd ~/serve/cpp
./builld.sh
```

2. Download the model

```bash
cd ~/serve/examples/cpp/llamacpp
curl -L https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_0.gguf?download=true -o llama-2-7b-chat.Q5_0.gguf
```

4. Create a [config.json](config.json) with the path of the downloaded model weights:

```bash
echo '{
"checkpoint_path" : "/home/ubuntu/serve/examples/cpp/llamacpp/llama-2-7b-chat.Q5_0.gguf"
}' > config.json
```

5. Copy handle .so file

While building the C++ backend the `libllamacpp_handler.so` file is generated in the [llamacpp_handler](../../../cpp/_build/test/resources/examples/llamacpp/llamacpp_handler) folder.

```bash
cp ../../../cpp/_build/test/resources/examples/llamacpp/llamacpp_handler/libllamacpp_handler.so ./
```

### Generate MAR file

Now lets generate the mar file

```bash
torch-model-archiver --model-name llm --version 1.0 --handler libllamacpp_handler:LlamaCppHandler --runtime LSP --extra-files config.json
```

Create model store directory and move the mar file

```
mkdir model_store
mv llm.mar model_store/
```

### Inference

Start torchserve using the following command

```
export LD_LIBRARY_PATH=`python -c "import torch;from pathlib import Path;p=Path(torch.__file__);print(f\"{(p.parent / 'lib').as_posix()}:{(p.parents[1] / 'nvidia/nccl/lib').as_posix()}\")"`:$LD_LIBRARY_PATH
torchserve --ncs --model-store model_store/
```

Register the model using the following command

```
curl -v -X POST "http://localhost:8081/models?initial_workers=1&url=llm.mar&batch_size=2&max_batch_delay=5000"
```

Infer the model using the following command

```
curl http://localhost:8080/predictions/llm -T prompt1.txt
```

This example supports batching. To run batch prediction, run the following command

```
curl http://localhost:8080/predictions/llm -T prompt1.txt & curl http://localhost:8080/predictions/llm -T prompt2.txt &
```

Sample Response

```
Hello my name is Daisy everybody loves me
 I am a sweet and loving person
 I have a big heart and I am always willing to help others
 I am a good
```
