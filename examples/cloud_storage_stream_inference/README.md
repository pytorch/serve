# Using fsspec to stream data from cloud storage providers for bacth inference

This example shows how to use fsspec to stream large amount of data from cloud storage like s3, google cloud storage, azure cloud etc. and use it to create requests to torchserve for large scale batch inference with a large batch size.

The example shows retrieval of data using s3 API and fsspec, but the same idea can be used to retrieve data from other cloud storage peovider like Google Cloud, Azure etc.
**Requires python >= 3.8**

## Example overview

  - Main objective of this example is to show the process of reading data from a cloud storage provider and send batch inference request to a model deployed using torchserve. We're not concerned with the accuracy of predictions.
  - In this example we're going to run pre-trained distilbert model with torchserve and do text classification by reading input data using S3 API and fsspec and making REST API call to torchserve for batch inference.
  - We're going to use Amazon reviews dataset which contains customer reviews along with title and label.
  - We'll be taking customer review text and predict if the review is positive or negative usign pre-trained distilbert model.
  - We'll be using minio https://min.io/ as AWS S3 proxy to store Amazon reviews dataset tar file.

## Steps

1) Install minio -  https://hub.docker.com/r/minio/minio/

2) Download Amazon reviews dataset tar file (amazon_review_polarity_csv.tar.gz) by going to the following URL in browser - https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM

3) Start minio server
```
minio server ~/minio_data
```

4) Log into minio administration web page http://127.0.0.1:54379/ with user id - minioadmin and password - minioadmin

5) Create a bucket with name **pytorch-data**

6) Change the bucket access policy from private to public by clicking on manage button for the bucket from http://127.0.0.1:54379/buckets/pytorch-data/admin/summary

7) Upload **amazon_review_polarity_csv.tar.gz** file that was downloaded in step 2 to **pytorch-data** bucket from http://127.0.0.1:54379/buckets/pytorch-data/browse

8) Clone pytorch/serve project from github - https://github.com/pytorch/serve. This is required to run this example.

9) Create python virtual env if you like.

10) Install torchserve - https://github.com/pytorch/serve. This is required to run distilbert model.

11) Install transformer package
```
pip install transformers==4.6.0
```
If you get an error regarding missing rust compiler while installing transformers, install rust compiler -
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

12) Download **distilbert-base-uncased** pretrained model. This will download model files in Transformer_model directory.
```
cd serve/examples/cloud_storage_stream_inference
python3 Download_Transformer_models.py
```

13) Create model archive eager mode
```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ./Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json,./index_to_name.json"
```

14) Register and start serving the model with torchserve
```
mkdir model_store
mv BERTSeqClassification.mar model_store/
torchserve --start --model-store model_store --models my_tc=BERTSeqClassification.mar --ts-config=config.properties --ncs
```

15) To check if the model is running
```
curl http://localhost:8081/models/
```
You should see
```
{
  "models": [
    {
      "modelName": "my_tc",
      "modelUrl": "BERTSeqClassification.mar"
    }
  ]
}
```

16) Install **fsspec**
```
pip install fsspec
```

17) Install **s3fs**
```
pip install s3fs
```
18) Run the example
```
python stream_inference.py
```
This example -
- Reads **amazon_review_polarity_csv.tar.gz** file from minio usign S3 API
- Reads **test.csv** file from the tar file
- Extracts customer review string
- Creates a torchserve batch inference REST API requests with customer review string and sends it to torchserve for inference
- Prints model prediction
