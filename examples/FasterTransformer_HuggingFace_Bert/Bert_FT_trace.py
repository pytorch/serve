# This script is adopted from the run-glue example of Nvidia-FasterTransformer,https://github.com/NVIDIA/FasterTransformer/blob/main/sample/pytorch/run_glue.py

import argparse
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm, trange

from transformers import (
    BertConfig,
    BertTokenizer,
)
from utils.modeling_bert import BertForSequenceClassification, BertForQuestionAnswering
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--mode", default= "sequence_classification", help=" Set the model for sequence classification or question answering")
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--batch_size", default=8, type=int, help="Batch size for tracing.",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    # parser.add_arument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--model_type", type=str, help="ori, ths, thsext")
    parser.add_argument("--data_type", type=str, help="fp32, fp16")
    parser.add_argument('--ths_path', type=str, default='./lib/libpyt_fastertransformer.so',
                        help='path of the pyt_fastertransformer dynamic lib file')
    parser.add_argument('--remove_padding', action='store_false',
                        help='Remove the padding of sentences of encoder.')
    parser.add_argument('--allow_gemm_test', action='store_false',
                        help='per-channel quantization.')

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.device else logging.WARN,
    )

    # Set seed
    set_seed(args)

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    logger.info("Parameters %s", args)

    checkpoints = [args.model_name_or_path]
    for checkpoint in checkpoints:
        use_ths = args.model_type.startswith('ths')
        if args.mode == "sequence_classification":
            model = BertForSequenceClassification.from_pretrained(checkpoint, torchscript=use_ths)
        elif args.mode == "question_answering":
            model = BertForQuestionAnswering.from_pretrained(checkpoint, torchscript=use_ths)
        model.to(args.device)

        if args.data_type == 'fp16':
            logger.info("Use fp16")
            model.half()
        if args.model_type == 'thsext':
            logger.info("Use custom BERT encoder for TorchScript")
            from utils.encoder import EncoderWeights, CustomEncoder
            weights = EncoderWeights(
                model.config.num_hidden_layers, model.config.hidden_size,
                torch.load(os.path.join(checkpoint, 'pytorch_model.bin'), map_location='cpu'))
            weights.to_cuda()
            if args.data_type == 'fp16':
                weights.to_half()
            enc = CustomEncoder(model.config.num_hidden_layers,
                                model.config.num_attention_heads,
                                model.config.hidden_size//model.config.num_attention_heads,
                                weights,
                                remove_padding=args.remove_padding,
                                allow_gemm_test=(args.allow_gemm_test),
                                path=os.path.abspath(args.ths_path))
            enc_ = torch.jit.script(enc)
            model.replace_encoder(enc_)
        if use_ths:
            logger.info("Use TorchScript mode")
            fake_input_id = torch.LongTensor(args.batch_size, args.max_seq_length)
            fake_input_id.fill_(1)
            fake_input_id = fake_input_id.to(args.device)
            fake_mask = torch.ones(args.batch_size, args.max_seq_length).to(args.device)
            fake_type_id = fake_input_id.clone().detach()
            if args.data_type == 'fp16':
                fake_mask = fake_mask.half()
            model.eval()
            with torch.no_grad():
                print("********** input id and mask sizes ******",fake_input_id.size(),fake_mask.size() )
                model_ = torch.jit.trace(model, (fake_input_id, fake_mask))
                model = model_
                torch.jit.save(model,"traced_model.pt")

if __name__ == "__main__":
    main()