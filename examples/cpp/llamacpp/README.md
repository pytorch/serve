This example used [llama.cpp](https://github.com/ggerganov/llama.cpp) to deploy a Llama-2-7B-Chat model using the TorchServe C++ backend.
The handler C++ source code for this examples can be found [here](../../../cpp/src/examples/llamacpp/).

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

While building the C++ backend the `libllamacpp_handler.so` file is generated in the [llamacpp_handler](../../../cpp/test/resources/examples/llamacpp/llamacpp_handler) folder.

```bash
cp ../../../cpp/test/resources/examples/llamacpp/llamacpp_handler/libllamacpp_handler.so ./
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
Hello my name is Daisy. Daisy is three years old. She loves to play with her toys.
One day, Daisy's mommy said, "Daisy, it's time to go to the store." Daisy was so excited! She ran to the store with her mommy.
At the store, Daisy saw a big, red balloon. She wanted it so badly! She asked her mommy, "Can I have the balloon, please?"
Mommy said, "No, Daisy. We don't have enough money for that balloon."
Daisy was sad. She wanted the balloon so much. She started to cry.
Mommy said, "Daisy, don't cry. We can get the balloon. We can buy it and take it home."
Daisy smiled. She was so happy. She hugged her mommy and said, "Thank you, mommy!"
```
