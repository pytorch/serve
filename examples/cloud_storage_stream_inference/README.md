# Using fsspec to stream data from cloud storage providers for batch inference

This example shows how to use fsspec to stream large amount of data from cloud storage like s3, google cloud storage, azure cloud etc. and use it to create requests to torchserve for large scale batch inference with a large batch size.

The example shows retrieval of data using s3 API and fsspec, but the same idea can be used to retrieve data from other cloud storage providers like Google Cloud, Azure etc.

Filesystem Spec (fsspec) is a project to provide a unified pythonic interface to local, remote and embedded file systems and bytes storage. https://filesystem-spec.readthedocs.io/en/latest/

**Requires python >= 3.8**

## Example overview

  - Main objective of this example is to show the process of reading data from a cloud storage provider and send batch inference request to a model deployed using torchserve. **We're not concerned with the accuracy of predictions.**
  - In this example we're going to run pre-trained distilbert model with torchserve and do text classification by reading input data using S3 API and fsspec and making REST API call to torchserve for batch inference.
  - We're going to use Amazon reviews dataset which contains customer reviews along with title and label.
  - We'll be taking customer review text and predict if the review is positive or negative using pre-trained distilbert model.
  - We'll be using MinIO https://min.io/ as AWS S3 proxy to store Amazon reviews dataset tar file. MinIO is a High Performance Object Storage released under GNU Affero General Public License v3.0. It is API compatible with Amazon S3 cloud storage service. It can handle unstructured data such as photos, videos, log files, backups, and container images with the maximum supported object size of 5TB.

## Steps

1) Install MinIO -  https://hub.docker.com/r/minio/minio/.
<!-- markdown-link-check-disable -->
2) Download Amazon reviews dataset tar file (amazon_review_polarity_csv.tar.gz) by going to the following URL in browser - https://drive.google.com/uc?export=download&id=0Bz8a_Dbh9QhbaW12WVVZS2drcnM

3) Start MinIO server
```
minio server ~/minio_data
```

4) Log into MinIO administration web page http://127.0.0.1:54379/ with user id - minioadmin and password - minioadmin

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
python3 ../Huggingface_Transformers/Download_Transformer_models.py ../cloud_storage_stream_inference/setup_config.json
```

13) Create model archive eager mode
```
torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file Transformer_model/pytorch_model.bin --handler ../Huggingface_transformers/Transformer_handler_generalized.py --extra-files "Transformer_model/config.json,./setup_config.json,./index_to_name.json"
```

14) Register and start serving the model with torchserve
```
mkdir model_store
mv BERTSeqClassification.mar model_store/
torchserve --start --model-store model_store --models my_tc=BERTSeqClassification.mar --ts-config=config.properties --ncs --disable-token-auth  --enable-model-api
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
- Reads **amazon_review_polarity_csv.tar.gz** file from MinIO using S3 API
- Reads **test.csv** file from the tar file
- Extracts customer review string
- Creates a torchserve batch inference REST API requests with customer review string and sends it to torchserve for inference
- Prints model prediction
```
2022-10-28 10:23:48.088857 - Calling model inference with data -  {"text":"A waste of time"}
2022-10-28 10:23:48.088916 - Calling model inference with data -  {"text":"One of the best films ever made"}
2022-10-28 10:23:48.088977 - Calling model inference with data -  {"text":"Gods and Monsters is a superb movie about the last days of gay film director James Whale -- who directed Frankenstein and Bride of Frankenstein"}
2022-10-28 10:23:48.089036 - Calling model inference with data -  {"text":"One of the few recent films that I anticipated to be good and it exceeded my expectations"}
2022-10-28 10:23:48.089094 - Calling model inference with data -  {"text":"The manufacturer packed this product so poorly that the plastic joints between sections of the tree were smashed"}
2022-10-28 10:23:48.089154 - Calling model inference with data -  {"text":"Of the 200+ baseball books I've read"}
2022-10-28 10:23:55.767787 - Model prediction:  Negative
2022-10-28 10:23:55.768008 - Model prediction:  Negative
2022-10-28 10:23:55.768460 - Model prediction:  Negative
2022-10-28 10:23:55.768516 - Model prediction:  Negative
2022-10-28 10:23:55.768573 - Model prediction:  Positive
```
