This example uses Bert Maher's [llama2.so](https://github.com/bertmaher/llama2.so/) which is a fork of Andrej Karpathy's [llama2.c](https://github.com/karpathy/llama2.c).
It uses AOTInductor to compile the model into an so file which is then executed using libtorch.
The handler C++ source code for this examples can be found [here](src/).

### Setup
1. Follow the instructions in [README.md](../../../../cpp/README.md) to build the TorchServe C++ backend.

```
cd serve/cpp
./builld.sh
```

The build script will already create the necessary artifact for this example.
To recreate these by hand you can follow the prepare_test_files function of the [build.sh](../../../../cpp/build.sh) script.
We will need the handler .so file as well as the stories15M.so file containing the model and weights.

2. Copy the handler file

```bash
cd ~/serve/examples/cpp/aot_inductor/llama2
cp ../../../../cpp/_build/test/resources/examples/aot_inductor/llama_handler/libllama_so_handler.so ./
```
We will leave the model .so file in place and just use its absolute path in the next step.

4. Create a [config.json](config.json) with the path of the downloaded model and tokenizer:

```bash
echo '{
"checkpoint_path" : "/home/ubuntu/serve/cpp/_build/test/resources/examples/aot_inductor/llama_handler/stories15M.so",
"tokenizer_path" : "/home/ubuntu/serve/cpp/_build/test/resources/examples/babyllama/babyllama_handler/tokenizer.bin"
}' > config.json
```

The tokenizer is the same we also use for the babyllama example so we can reuse the file from there.

### Generate MAR file

Now lets generate the mar file

```bash
torch-model-archiver --model-name llm --version 1.0 --handler libllama_so_handler:LlamaHandler --runtime LSP --extra-files config.json
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
