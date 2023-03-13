import json
import zipfile
from abc import ABC

import inspect
import logging
import os
import time

import torch
import pippy
import pippy.fx
from torch.distributed._tensor import (
    DeviceMesh,
)
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
)


from pippy.IR import MultiUseParameterConfig, Pipe
from pippy.PipelineDriver import PipelineDriverFillDrain, PipelineDriver1F1B, PipelineDriverInterleaved1F1B, \
    PipelineDriverBase
from pippy import split_on_size_threshold, split_into_equal_size
from transformers import OPTModel, BloomModel
import torch.distributed.rpc as rpc

import transformers
from transformers import BloomForCausalLM, BloomTokenizerFast
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
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.pp_rank = 0
        self.pp_ranks = None

    def initialize(self, ctx):
        """In this initialize function, the BERT model is loaded and
        the Layer Integrated Gradients Algorithm for Captum Explanations
        is initialized here.
        Args:
            ctx (context): It is a JSON Object containing information
            pertaining to the model artefacts parameters.
        """
        pp_group_size = self.world_size
        num_worker_threads = 512
        rpc_timeout = 1800
        if ctx.model_yaml_config is not None:
            if ctx.system_properties.get("gpu_id") != -1 \
                    and ctx.model_yaml_config["deviceIds"] is not None:
                device_ids = ','.join(str(e) for e in ctx.model_yaml_config["deviceIds"][int(ctx.system_properties.get("gpu_id")):int(ctx.system_properties.get("gpu_id"))+self.world_size+1])
                os.environ["CUDA_VISIBLE_DEVICE"] = device_ids

            if ctx.model_yaml_config[pippy] is not None:
                if ctx.model_yaml_config["pippy"]["pp_group_size"] is not None \
                        and self.world_size % int(ctx.model_yaml_config["pippy"]["pp_group_size"]) == 0:
                    pp_group_size = int(ctx.model_yaml_config["pippy"]["pp_group_size"])

                if ctx.model_yaml_config["pippy"]["num_worker_threads"] is not None:
                    num_worker_threads = int(ctx.model_yaml_config["pippy"]["num_worker_threads"])

                if ctx.model_yaml_config["pippy"]["rpc_timeout"] is not None:
                    rpc_timeout = int(ctx.model_yaml_config["pippy"]["rpc_timeout"])

        if ctx.system_properties.get("gpu_id") != -1 and os.environ["CUDA_VISIBLE_DEVICE"] is None:
            os.environ["CUDA_VISIBLE_DEVICE"] = ','.join(str(e) for e in range(self.local_rank))

        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads,
            rpc_timeout
        )
        device_type = "cpu"
        if int(ctx.system_properties.get("gpu_id")) != -1:
            device_type = "cuda"
            n_devs = torch.cuda.device_count()
            dev_id = self.local_rank % n_devs
            for i in range(self.world_size):
                logging.info(f"worker{i}, {dev_id}: {i % n_devs}")
                options.set_device_map(f"worker{i}", {dev_id: i % n_devs})

            self.device = f"cuda:{dev_id}"
            logging.info(
                f"rank = {self.local_rank} pid/device = "
                f"{os.getpid()}/{self.device}"
            )
        else:
            self.device = "cpu"
        rpc.init_rpc(f"worker{self.rank}",
                     rank=self.rank,
                     world_size=self.world_size,
                     rpc_backend_options=options)

        tp_group_size = self.world_size // pp_group_size
        dp_group_size = self.world_size // pp_group_size

        logging.info(
            f"[PiPPy] World size: {self.world_size}, "
            f"DP group size: {dp_group_size}, "
            f"PP group size: {pp_group_size}"
        )
        pp_ranks_per_dp_group = [
            [i + rank for i in range(pp_group_size)]
            for rank in range(dp_group_size)
        ]
        self.pp_ranks = pp_ranks_per_dp_group[self.rank % dp_group_size]
        self.pp_rank = self.rank // tp_group_size
        logging.info(f"Global rank {self.rank}, pipeline: {self.pp_ranks}, my rank in pipe: {self.pp_rank}")

        d_hid = 256
        batch_size_per_chunk = 8
        chunks = pp_group_size
        #inp_size = [chunks * batch_size_per_chunk, d_hid]
        # Ensure all tp ranks have same input.
        #torch.manual_seed(0)
        #inp = torch.rand(*inp_size, device=device_type)

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


        input_names = ['input_ids']
        sig = inspect.signature(model.forward)
        concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

        torch.manual_seed(0)
        ec_tp = model(d_hid)
        ec_tp.to(self.device)
        start_idx = 0
        device_mesh = DeviceMesh(
            device_type,
            list(range(start_idx, start_idx + tp_group_size)),
        )
        logging.info(f"Rank {self.rank} calling parallelize_module with {device_mesh}")
        parallelize_module(ec_tp, device_mesh, PairwiseParallel())
        logging.info(f"Rank {self.rank} sharding complete")

        print('Instantiating model Pipeline')
        model_init_start = time.time()
        # Get:
        # - pipeline driver (for pipeline head rank)
        # - stage submodule (for all ranks)
        pipe_driver, submod = pippy.all_compile(
            model,
            pp_group_size,
            chunks,
            ranks=self.pp_ranks,
        )

        # Create TP device mesh
        my_device_mesh = None
        for stage in range(pp_group_size):
            start_rank = stage * tp_group_size
            tp_ranks = list(range(start_rank, start_rank + tp_group_size))
            tp_device_mesh = DeviceMesh(
                device_type,
                tp_ranks,
            )
            if stage == self.pp_rank:
                my_device_mesh = tp_device_mesh

        # Tensor parallelize submodules
        print(f"Rank {self.rank} calling parallelize_module with {my_device_mesh}")
        parallelize_module(submod, my_device_mesh, PairwiseParallel())


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
        #if self.pp_rank == 0:
        #    print(f"Rank {self.rank} Instantiated pipeline with ranks {self.pp_ranks}")
        output = self.model(**model_input_dict)


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
