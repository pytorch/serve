# A TorchServe handler for Instructor Embedding models

A simple handler that you can use to serve [Instructor Embedding models](https://instructor-embedding.github.io/) with TorchServe, supporting both single inference and batch inference.

## Setup:

**1.** [Download an Instructor model (i.e. Instructor-XL)](https://huggingface.co/hkunlp/instructor-xl/tree/main?clone=true) from HuggingFace into your model store directory of choosing. Copy the `instructor-embedding-handler.py` into the same directory as your newly downloaded directory containing all the model-related files.

**2.** Create the .MAR Model Archive using [`torch-model-archiver`](https://github.com/pytorch/serve/blob/master/model-archiver/README.md):

```bash
torch-model-archiver --model-name <YOUR_MODEL_NAME_OF_CHOOSING> --version 1.0 --handler PATH/TO/instructor-embedding-handler.py --extra-files <DOWNLOADED_MODEL_DIR> --serialized-file <DOWNLOADED_MODEL_DIR>/pytorch_model.bin --f
```

**3.** Use [TorchServe](https://pytorch.org/serve/server.html) to startup the server and deploy the Instruction Embedding model you downloaded. 

**Note:** Instructor Embedding models are around ~4 GB. By default, torchserve will autoscale workers (each with a loaded copy of the model). [At present](https://github.com/pytorch/serve/issues/2432), if you have memory concerns, you have to make use of the [Management API](https://pytorch.org/serve/management_api.html) to bring up the server and deploy your model.


## Performing Inference
To perform inference for an instruction and corresponding sentence, use the following format for the request body: 
```json
{
    "inputs": [INSTRUCTION, SENTENCE]
}
```

To perform batch inference, use the following format for the request body:
```json
{
    "inputs": [
        [INSTRUCTION_1, SENTENCE_1],
        [INSTRUCTION_2, SENTENCE_2],
        ...
    ]
}
```

## Example: Single Inference 
Request Endpoint: /predictions/<model_name>

Request Body:
```json
{
    "inputs": ["Represent the Science title:", "3D ActionSLAM: wearable person tracking in multi-floor environments"]
}
```

### Response:
```json
[
  0.010738617740571499,
  ...
  0.10961631685495377
]
```

## Example: Batch Inference 
Request Endpoint: /predictions/<model_name>

Request Body:
```json
{
    "inputs": [
        ["Represent the Science title:", "3D ActionSLAM: wearable person tracking in multi-floor environments"],
        ["Represent the Medicine sentence for retrieving a duplicate sentence:", "Recent studies have suggested that statins, an established drug group in the prevention of cardiovascular mortality, could delay or prevent breast cancer recurrence but the effect on disease-specific mortality remains unclear."]
    ]
}
```

### Response:
```json
[
  [
    0.010738617740571499,
    ...
    0.10961631685495377
  ],
  [
    0.014582153409719467,
    ...
    0.08006688207387924
  ]
]
```

**Note:** The above request example was for batch inference on 2 distinct instruction/sentence pairs.