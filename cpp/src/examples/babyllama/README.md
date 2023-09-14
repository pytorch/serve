This example is adapted from https://github.com/karpathy/llama2.c 

### Setup

1. Follow the instructions from [README.md](../../../README.md) to build the cpp backend
2. Download the model and tokenizer using the following command

```
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
```
Download the [tokenizer.bin](https://github.com/karpathy/llama2.c/blob/master/tokenizer.bin) file from the [llama2.c](https://github.com/karpathy/llama2.c) repo

3. Update [config.json](config.json) with the path of the downloaded model and tokenizer.

For example

```
{
"checkpoint_path" : "/home/ubuntu/serve/cpp/stories15M.bin",
"tokenizer_path" : "/home/ubuntu/serve/cpp/src/examples/babyllama/tokenizer.bin"
}
```

4. Run the build

```
cd serve/cpp
./builld.sh
```

Once the build is completed, `libbabyllama_handler.so` file is generated in the [babyllama_handler](../../../test/resources/torchscript_model/babyllama/babyllama_handler) folder

### Generate MAR file

Move to [babyllama_handler](../../../test/resources/torchscript_model/babyllama/babyllama_handler) folder and run the following command to generate mar file

```
torch-model-archiver --model-name llm --version 1.0 --serialized-file dummy.pt --handler libbabyllama_handler:BabyLlamaHandler --runtime LSP --extra-files config.json
```

Create model store directory and move the mar file

```
mkdir model_store
mv llm.mar model_store/llm.mar
```

### Inference

Start torchserve using the following command

```
torchserve --start --ncs --ts-config config.properties --model-store model_store/
```

Register the model using the following command

```
curl -v -X POST "http://localhost:8081/models?initial_workers=1&url=llm.mar&batch_size=2&max_batch_delay=5000&initial_workers=3"
```

Infer the model using the following command

```
curl http://localhost:8080/predictions/llm -T prompt.txt
```

This example supports batching. To run batch prediction, run the following command 

```
curl http://localhost:8080/predictions/llm -T prompt.txt & curl http://localhost:8080/predictions/llm -T prompt1.txt &
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