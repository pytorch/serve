# English-to-French translation using Fairseq Transformer model
We have used the following Fairseq/Transformer model for this example: 
https://pytorch.org/hub/pytorch_fairseq_translation/

# Install pip dependencies using following commands

```bash
pip install fastBPE regex requests sacremoses subword_nmt
```
# Serve the English-to-French Translation model on TorchServe

* Generate the model archive for English-to-French translation model using following command

    ```bash
    ./create_mar.sh
    ```

* Register the model on TorchServe using the above model archive file

    ```bash
    torchserve --start --model-store model_store
    ```

    Register the model with one initial worker use the below curl command

    ```bash
    curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=TransformerEn2Fr.mar"
    ```
* To get the inference use the following curl command

    ```bash
    curl http://127.0.0.1:8080/predictions/TransformerEn2Fr -T sample.txt
    ```
