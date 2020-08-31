# English-to-German translation using Fairseq Transformer model

In this example, we have show how to serve a [English-to-German Translation](https://pytorch.org/hub/pytorch_fairseq_translation/#english-to-german-translation) model using TorchServe. We have used a custom handler, model_handler.py which enables us to use pre-trained [Transformer_WMT14_En-De](https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md) models from [fairseq](https://github.com/pytorch/fairseq). 

## Serve the English-to-German Translation model on TorchServe

* Generate the model archive (.mar) file for English-to-German translation model using following command

    ```bash
    ./create_mar.sh
    ```
    By execuing the above script file, the the model archive (.mar) file will be auto generated in "model_store" folder in the same working directory.


* Start the TorchServe using the model archive (.mar) file created in above step

    ```bash
    torchserve --start --model-store model_store --ts-config config.properties
    ```

* Use [Management API](https://github.com/pytorch/serve/blob/master/docs/management_api.md#management-api) to register the model with one initial worker

    ```bash
    curl -X POST "http://localhost:8081/models?initial_workers=1&synchronous=true&url=TransformerEn2De.mar"
    ```
    Note:- Our model works only for one worker.

* To get the inference use the following curl command

    ```bash
    curl http://127.0.0.1:8080/predictions/TransformerEn2De -T sample.txt
    ```
    Here sample.txt contains simple english sentences which are given as input to [Inference API](https://github.com/pytorch/serve/blob/master/docs/inference_api.md#predictions-api). The output of above curl command will be the german translation of sentences present in the sample.txt file.

## English-to-German translation using TorchServe with batch-supported model

* Start the model server. In this example, we are starting the model server with config.properties file

    ```bash
    torchserve --start --model-store model_store --ts-config config.properties
    ```

* Now let's launch English_to_German translation model, which we have built to handle batch inference. 
In this example, we are going to launch 1 worker which handles a `batch size` of 4 with a `max_batch_delay` of 10s.

    ```bash
    curl -X POST "http://localhost:8081/models?url=TransformerEn2De.mar&initial_workers=1&synchronous=true&batch_size=4&max_batch_delay=10000"
    ```

* Run batch inference command to test the model.

    ```bash
    curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2De -T ./model_input/sample1.txt& 
    curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2De -T ./model_input/sample2.txt& 
    curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2De -T ./model_input/sample3.txt& 
    curl -X POST http://127.0.0.1:8080/predictions/TransformerEn2De -T ./model_input/sample4.txt&
    ```
