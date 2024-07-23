## Faster Transformer

Batch inferencing with Transformers faces two challenges

- Large batch sizes suffer from higher latency and small or medium-sized batches this will become kernel latency launch bound.
- Padding wastes a lot of compute, (batchsize, seq_length) requires to pad the sequence to (batchsize, max_length) where difference between avg_length and max_length results in a considerable waste of computation, increasing the batch size worsen this situation.

[Faster Transformers](https://github.com/NVIDIA/FasterTransformer/blob/main/examples/pytorch/bert/run_glue.py) (FT) from Nvidia along with [Efficient Transformers](https://github.com/bytedance/effective_transformer) (EFFT) that is built on top of FT address the above two challenges, by fusing the CUDA kernels and dynamically removing padding during computations. The current implementation from [Faster Transformers](https://github.com/NVIDIA/FasterTransformer/blob/main/examples/pytorch/bert/run_glue.py) support BERT like encoder and decoder layers. In this example, we show how to get a Torchscripted (traced) EFFT variant of Bert models from HuggingFace (HF) for sequence classification and question answering and serve it.


### How to get a Torchscripted (Traced) EFFT of HF Bert model and serving it

**Requirements**

Running Faster Transformer at this point is recommended through [NVIDIA docker and NGC container](https://github.com/NVIDIA/FasterTransformer#requirements), also it requires [Volta](https://www.nvidia.com/en-us/data-center/volta-gpu-architecture/) or [Turing](https://www.nvidia.com/en-us/geforce/turing/) or [Ampere](https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/) based GPU. For this example we have used a **g4dn.2xlarge** EC2 instance that has a T4 GPU.

**Setup the a GPU machine that meets the requirements and connect to it**.

```bash
### Sign up for NGC https://ngc.nvidia.com  and get API key###
docker login nvcr.io
Username: $oauthtoken
Password: API key

docker pull nvcr.io/nvidia/pytorch:20.12-py3

nvidia-docker run -ti --gpus all --rm nvcr.io/nvidia/pytorch:20.12-py3 bash

git clone https://github.com/NVIDIA/FasterTransformer.git

cd FasterTransformer

mkdir -p build

cd build

cmake -DSM=75 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON ..   # -DSM = 70 for V100 gpu ------- 60 (P40) or 61 (P4) or 70 (V100) or 75(T4) or 80 (A100),

make

pip install transformers==2.5.1

cd /workspace

# clone Torchserve to access examples
git clone https://github.com/pytorch/serve.git

# install torchserve
cd serve

pip install -r requirements/common.txt

pip install torchserve torch-model-archiver torch-workflow-archiver

cp ./examples/FasterTransformer_HuggingFace_Bert/Bert_FT_trace.py /workspace/FasterTransformer/build/pytorch


```

Now we are ready to make the Torchscripted file, as mentioned at the beginning two models are supported Bert for sequence classification and question answering. To do this step we need the download the model weights. We do this the same way we do in [HuggingFace example](https://github.com/pytorch/serve/tree/master/examples/Huggingface_Transformers).

#### Sequence classification EFFT Traced model and serving

```bash
# Sequence classification, make sure to comment out import set_seed in Download_Transformer_models.py as its not supported in Transformers=2.5.1
python ./examples/Huggingface_Transformers/Download_Transformer_models.py

# This will download the model weights in ../Huggingface_Transformers/Transfomer_model directory

cd /workspace/FasterTransformer/build/

# This will generate the Traced model "traced_model.pt"
# --data_type can be fp16 or fp32
# --model_type being specified to thsext will use the Faster Transformer fusions
# --remove_padding is chosen by default that will make use of efficient padding along with Faster Transformer

python pytorch/Bert_FT_trace.py --mode sequence_classification --model_name_or_path /workspace/serve/Transformer_model --tokenizer_name "bert-base-uncased" --batch_size 1 --data_type fp16 --model_type thsext

cd /workspace/serve/examples/FasterTransformer_HuggingFace_Bert

# make sure to change the ../Huggingface_Transformers/setup_config.json "save_mode":"torchscript" and "FasterTransformer":true

# change the ../Huggingface_Transformers/setup_config.json as below
{
 "model_name":"bert-base-uncased",
 "mode":"sequence_classification",
 "do_lower_case":true,
 "num_labels":"0",
 "save_mode":"torchscript",
 "max_length":"128",
 "captum_explanation":false,
 "embedding_name": "bert",
 "FasterTransformer":true
}

torch-model-archiver --model-name BERTSeqClassification --version 1.0 --serialized-file /workspace/FasterTransformer/build/traced_model.pt --handler ../Huggingface_Transformers/Transformer_handler_generalized.py --extra-files "../Huggingface_Transformers/setup_config.json,../Huggingface_Transformers/Seq_classification_artifacts/index_to_name.json,/workspace/FasterTransformer/build/lib/libpyt_fastertransformer.so"

mkdir model_store

mv BERTSeqClassification.mar model_store/

torchserve --start --model-store model_store --models my_tc=BERTSeqClassification.mar --ncs --disable-token-auth  --enable-model-api

curl -X POST http://127.0.0.1:8080/predictions/my_tc -T ../Huggingface_Transformers/Seq_classification_artifacts/sample_text_captum_input.txt

```

#### Question answering EFFT Traced model and serving

```bash
# Question answering

# change the ../Huggingface_Transformers/setup_config.json as below
{
 "model_name":"bert-base-uncased",
 "mode":"question_answering",
 "do_lower_case":true,
 "num_labels":"0",
 "save_mode":"torchscript",
 "max_length":"128",
 "captum_explanation":false,
 "embedding_name": "bert",
 "FasterTransformer":true
}
python ../Huggingface_Transformers/Download_Transformer_models.py

# This will download the model weights in ../Huggingface_Transformers/Transfomer_model directory

cd /workspace/FasterTransformer/build/

# This will generate the Traced model "traced_model.pt"
# --data_type can be fp16 or fp32
python pytorch/Bert_FT_trace.py --mode question_answering --model_name_or_path "/workspace/serve/Transformer_model" --tokenizer_name "bert-base-uncased" --batch_size 1 --data_type fp16 --model_type thsext

cd -

# make sure to change the ../Huggingface_Transformers/setup_config.json "save_mode":"torchscript"

torch-model-archiver --model-name BERTQA --version 1.0 --serialized-file /workspace/FasterTransformer/build/traced_model.pt --handler ../Huggingface_Transformers/Transformer_handler_generalized.py --extra-files "./examples/Huggingface_Transformers/setup_config.json,/workspace/FasterTransformer/build/lib/libpyt_fastertransformer.so"

mkdir model_store

mv BERTQA.mar model_store/

torchserve --start --model-store model_store --models my_tc=BERTQA.mar --ncs --disable-token-auth  --enable-model-api

curl -X POST http://127.0.0.1:8080/predictions/my_tc -T ../Huggingface_Transformers/QA_artifacts/sample_text_captum_input.txt

```

####
