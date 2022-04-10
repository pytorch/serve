#From Tesnorrt Repo https://github.com/NVIDIA/TensorRT/tree/main
import torch
import tensorrt as trt
import os
import sys
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

import torch
import tensorrt as trt
from T5.trt import T5TRTEncoder, T5TRTDecoder

# huggingface
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
)
from collections import namedtuple

from T5.T5ModelConfig import T5ModelTRTConfig
from T5.measurements import decoder_inference, encoder_inference, full_inference_greedy
from T5.export import T5EncoderTorchFile, T5DecoderTorchFile
from NNDF.networks import TimingProfile
# from NNDF.networks import NetworkMetadata, Precision
from T5.export import T5DecoderONNXFile, T5EncoderONNXFile

def create_path(NEW_DIR):
    try:
        os.mkdir(NEW_DIR)
    except OSError:
        print ("Creation of directory %s failed" % NEW_DIR)
    else:
        print ("Successfully created directory %s " % NEW_DIR)

"""Precision(fp16: Bool)"""
Precision = namedtuple("Precision", ["fp16"])

"""NetworkMetadata(variant: str, precision: Precision, other: Union[namedtuple, None])"""
NetworkMetadata = namedtuple("NetworkMetadata", ["variant", "precision", "other"])

T5_VARIANT = 't5-small' # choices: t5-small | t5-base | t5-large

t5_model = T5ForConditionalGeneration.from_pretrained(T5_VARIANT)
tokenizer = T5Tokenizer.from_pretrained(T5_VARIANT)
config = T5Config(T5_VARIANT)

pytorch_model_dir = "./saved_model"
create_path(pytorch_model_dir)
t5_model.save_pretrained(pytorch_model_dir)
inputs = tokenizer("translate English to German: That is good.", return_tensors="pt")

# inference on a single example
t5_model.eval()
with torch.no_grad():
    outputs = t5_model(**inputs, labels=inputs["input_ids"])

logits = outputs.logits

# Generate sequence for an input
outputs = t5_model.to('cuda:0').generate(inputs.input_ids.to('cuda:0'))
print(tokenizer.decode(outputs[0], skip_special_tokens=True))



t5_torch_encoder = T5EncoderTorchFile.TorchModule(t5_model.encoder)
t5_torch_decoder = T5DecoderTorchFile.TorchModule(
    t5_model.decoder, t5_model.lm_head, t5_model.config
)
input_ids = inputs.input_ids
print(input_ids.type())
encoder_last_hidden_state, encoder_e2e_median_time = encoder_inference(
    t5_torch_encoder, input_ids, TimingProfile(iterations=10, number=1, warmup=1)
)

# convert to ONNX
onnx_model_path = "onnx_models"
metadata=NetworkMetadata(T5_VARIANT, Precision('fp16'), None)

encoder_onnx_model_fpath = T5_VARIANT + "-encoder.onnx"
decoder_onnx_model_fpath = T5_VARIANT + "-decoder-with-lm-head.onnx"

t5_encoder = T5EncoderTorchFile(t5_model.to('cpu'), metadata)
t5_decoder = T5DecoderTorchFile(t5_model.to('cpu'), metadata)

onnx_t5_encoder = t5_encoder.as_onnx_model(
    os.path.join(onnx_model_path, encoder_onnx_model_fpath), force_overwrite=False
)
onnx_t5_decoder = t5_decoder.as_onnx_model(
    os.path.join(onnx_model_path, decoder_onnx_model_fpath), force_overwrite=False
)


tensorrt_model_path = './tensort_models'
create_path(tensorrt_model_path)

t5_trt_encoder_engine = T5EncoderONNXFile(
                os.path.join(onnx_model_path, encoder_onnx_model_fpath), metadata
            ).as_trt_engine(os.path.join(tensorrt_model_path, encoder_onnx_model_fpath) + ".engine")

t5_trt_decoder_engine = T5DecoderONNXFile(
                os.path.join(onnx_model_path, decoder_onnx_model_fpath), metadata
            ).as_trt_engine(os.path.join(tensorrt_model_path, decoder_onnx_model_fpath) + ".engine")

#convert to TRT


tfm_config = T5Config(
    use_cache=True,
    num_layers=T5ModelTRTConfig.NUMBER_OF_LAYERS[T5_VARIANT],
)
    
t5_trt_encoder = T5TRTEncoder(
                t5_trt_encoder_engine, metadata, tfm_config
            )
t5_trt_decoder = T5TRTDecoder(
                t5_trt_decoder_engine, metadata, tfm_config
            )
from transformers.generation_stopping_criteria import (
    MaxLengthCriteria,
    StoppingCriteriaList,
)


decoder_output_greedy, full_e2e_median_runtime = full_inference_greedy(
    t5_trt_encoder,
    t5_trt_decoder,
    input_ids,
    tokenizer,
     TimingProfile(10,1,1),
    max_length=T5ModelTRTConfig.MAX_SEQUENCE_LENGTH[metadata.variant],
    use_cuda=False,
)

print(full_e2e_median_runtime, decoder_output_greedy)