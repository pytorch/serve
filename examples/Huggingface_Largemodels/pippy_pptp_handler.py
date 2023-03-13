import json
import zipfile
from abc import ABC

import inspect
import logging
import os
import time

import pippy
import pippy.fx
from pippy import run_pippy
from pippy.IR import pipe_split
from pippy.IR import MultiUseParameterConfig, Pipe
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy.hf import PiPPyHFTracer
from pippy.microbatch import TensorChunkSpec
from pippy import split_on_size_threshold, split_into_equal_size
from transformers import  AutoModelForSeq2SeqLM
from transformers import OPTModel, BloomModel
from PIL import Image
import requests
from transformers import AutoFeatureExtractor, RegNetModel 
from transformers import OPTForCausalLM
import torch.distributed.rpc as rpc

import torch
import transformers
from transformers import BloomForCausalLM, BloomTokenizerFast


from pippy import run_pippy
from pippy.IR import pipe_split

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

PIPPY_VERBOSITY = os.environ.get("PIPPY_VERBOSITY", "DEBUG")
TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

schedules = {
    'FillDrain': PipelineDriverFillDrain,
    '1F1B': PipelineDriver1F1B,
    'Interleaved1F1B': PipelineDriverInterleaved1F1B,
}

class TransformersSeqClassifierHandler(BaseHandler, ABC):
    """
    Transformers handler class for sequence, token classification and question answering.
    """

    def __init__(self):
        super(TransformersSeqClassifierHandler, self).__init__()
        self.initialized = False
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=512,
            rpc_timeout=1800
        #    transports=None,
        )
         
       
        # if args.cuda:
        n_devs = torch.cuda.device_count()
        print(f"n_devs={n_devs}")
        dev_id = self.local_rank % n_devs 
        for i in range (self.world_size):
            print(f"worker{i}, {dev_id}: {i % n_devs}")
            options.set_device_map(f"worker{i}", {dev_id: i % n_devs})

        self.device = f"cuda:{dev_id}"
        print(
            f"rank = {self.local_rank} pid/device = "
            f"{os.getpid()}/{self.device}"
        )

        rpc.init_rpc(f"worker{self.local_rank}",
                     rank=self.local_rank,
                     world_size=self.world_size,
                     rpc_backend_options=options)

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        # parser = argparse.ArgumentParser()
        # args = parser.parse_args()
        # args.world_size = 4
        # args.gspmd = 1
        #if self.local_rank != 0:
        #    return

        self.manifest = ctx.manifest
        properties = ctx.system_properties
        model_dir = properties.get("model_dir")

        # self.device = torch.device(
        #     "cuda:" + str(properties.get("gpu_id"))
        #     if torch.cuda.is_available() and properties.get("gpu_id") is not None
        #     else "cpu"
        # )
        # Loading the model and tokenizer from checkpoint and config files based on the user's choice of mode
        # further setup config can be added.
        with zipfile.ZipFile(model_dir + "/model.zip", "r") as zip_ref:
            zip_ref.extractall(model_dir + "/model")

        # read configs for the mode, model_name, etc. from setup_config.json
        setup_config_path = os.path.join(model_dir, "setup_config.json")
        if os.path.isfile(setup_config_path):
            with open(setup_config_path) as setup_config_file:
                self.setup_config = json.load(setup_config_file)
        else:
            logger.warning("Missing the setup_config.json file.")

        torch.manual_seed(42)
        replicate = 0
        schedule = list(schedules.keys())[0]
        MULTI_USE_PARAM_CONFIG = MultiUseParameterConfig.REPLICATE if replicate else MultiUseParameterConfig.TRANSMIT
        print(f'REPLICATE config: {replicate} -> {MULTI_USE_PARAM_CONFIG}')
        print("Using schedule:", schedule)

        model = BloomModel.from_pretrained(
            model_dir + "/model", use_cache=False)

        self.tokenizer = BloomTokenizerFast.from_pretrained(
            model_dir + "/model", return_tensors="pt"
        )
        
        logger.info("********************* model loaded *************************", model_dir)

        # model = BloomModel.from_pretrained("bigscience/bloom-3b", use_cache=False)

        model_config = model.config

        model_config.use_cache = False  # don't output `past_key_values`
        model.eval()
    

        split_policy = split_into_equal_size(self.world_size)
        pp_ranks = [0,1,2,3]
        all_worker_ranks = list(range(self.world_size))
        chunks = 1
        bs = 1 * chunks
        seq_length = 16


        input_names = ['input_ids']
        sig = inspect.signature(model.forward)
        concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

        print('Instantiating model Pipeline')
        model_init_start = time.time()
        pipe_driver, stage_mode = pippy.all_compile(
            model,
            num_ranks=self.world_size,
            num_chunks=chunks,
            schedule="FillDrain",
            split_policy=split_policy,
            tracer=PiPPyHFTracer(),
            concrete_args=concrete_args,
        )
        # model_pipe = Pipe.from_tracing(self.model, MULTI_USE_PARAM_CONFIG, tracer=PiPPyHFTracer(), concrete_args=concrete_args,
        #                             output_loss_value_spec=None, split_policy=split_policy
        #                             )
    
        # model_pipe.defer_stage_init(self.device + self.local_rank)

        # pippy.utils.pp_group_barrier()
        
        # split_gm_children = list(model_pipe.split_gm.children())

        # pipe_driver: PipelineDriverBase = schedules["FillDrain"](model_pipe, chunks,
        #                                                         world_size=self.world_size,
        #                                                         all_ranks=all_worker_ranks,
        #                                                             )

        self.model = pipe_driver
        logger.info("Transformer model from path %s loaded successfully", model_dir)

        self.initialized = True


    def preprocess(self, requests):
        """Basic text preprocessing, based on the user's chocie of application mode.
        Args:
            requests (str): The Input data in the form of text is passed on to the preprocess
            function.
        Returns:
            list : The preprocess function returns a list of Tensor for the size of the word tokens.
        """
        input_ids_batch = None
        attention_mask_batch = None
        for idx, data in enumerate(requests):
            input_text = data.get("data")
            if input_text is None:
                input_text = data.get("body")
            if isinstance(input_text, (bytes, bytearray)):
                input_text = input_text.decode("utf-8")

            max_length = self.setup_config["max_length"]
            logger.info("Received text: '%s'", input_text)

            inputs = self.tokenizer.encode_plus(
                input_text,
                max_length=int(max_length),
                pad_to_max_length=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            # making a batch out of the recieved requests
            # attention masks are passed for cases where input tokens are padded.
            if input_ids.shape is not None:
                if input_ids_batch is None:
                    input_ids_batch = input_ids
                    attention_mask_batch = attention_mask
                else:
                    input_ids_batch = torch.cat((input_ids_batch, input_ids), 0)
                    attention_mask_batch = torch.cat(
                        (attention_mask_batch, attention_mask), 0
                    )
        return (input_ids_batch, attention_mask_batch)

    def inference(self, input_batch):
        """Predict the class (or classes) of the received text using the
        serialized transformers checkpoint.
        Args:
            input_batch (list): List of Text Tensors from the pre-process function is passed here
        Returns:
            list : It returns a list of the predicted value for the input text
        """
        (input_ids_batch, _) = input_batch
        inferences = []
        input_ids_batch = input_ids_batch.to(self.device)
        model_input_dict = {}
        model_input_dict["input_ids"]=input_ids_batch
        # outputs = self.model.generate(
        #     input_ids_batch, do_sample=True, max_length=50, top_p=0.95, top_k=60
        # )
        # for i, _ in enumerate(outputs):
        #     inferences.append(
        #         self.tokenizer.decode(outputs[i], skip_special_tokens=True)
        #     )
        if self.local_rank==0:
            output = self.model(**model_input_dict)
        # rpc.shutdown()
        print("************** here is the output",type(output))
        logger.info("Generated text: '%s'", inferences)
        inferences.append(output)
        print("Generated text", inferences)
        return inferences

    def postprocess(self, inference_output):
        """Post Process Function converts the predicted response into Torchserve readable format.
        Args:
            inference_output (list): It contains the predicted response of the input text.
        Returns:
            (list): Returns a list of the Predictions and Explanations.
        """
        return inference_output

    def handle(self, data, context):
        if self.local_rank != 0:
            pass
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics
        
        #run_pippy(self.initialize, context)

        is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
        if is_profiler_enabled:
            if PROFILER_AVAILABLE:
                output, _ = self._infer_with_profiler(data=data)
            else:
                raise RuntimeError(
                    "Profiler is enabled but current version of torch does not support."
                    "Install torch>=1.8.1 to use profiler."
                )
        else:
            if self._is_describe():
                output = [self.describe_handle()]
            else:
                data_preprocess = self.preprocess(data)

                if not self._is_explain():
                    output = self.inference(data_preprocess)
                    output = self.postprocess(output)
                else:
                    output = self.explain_handle(data_preprocess, data)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output
