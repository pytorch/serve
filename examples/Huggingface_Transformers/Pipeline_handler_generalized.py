'''
TODO:
(i) Add support for remaining Pipeline tasks
(ii) Add support for Captum explanations

'''

import os
import torch
import base64
import io
from PIL import Image
import logging

from ts.torch_handler.base_handler import BaseHandler
from transformers import pipeline

logger = logging.getLogger(__name__)
logger.info("Transformers version %s", transformers.__version__)

#Reference: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.task
pipeline_supported_tasks= [
    "image-classification",
    "sentiment-analysis",
    "text-classification"
]

task="<PLACEHOLDER>"
framework="<PLACEHOLDER>"

try:
    assert task in pipeline_supported_tasks
except:
    logger.info("Enter a task supported by 🤗 pipeline")
    exit(0)

class HFPipelineHandler(BaseHandler):
    class HFPipelinePreprocessDispatcher:
        def __init__(self):
            '''
            `map_to_preproc_fn` returns a function for the task that can preprocess a single raw inference request
            '''
            self.map_to_preproc_fn = {
                "image-classification":self.preprocess_image_classification_single,
                "sentiment-analysis":self.preprocess_sentiment_analysis_single,
                "text-classification":self.preprocess_sentiment_analysis_single
            }
            pass
        '''
        Takes a list of raw inference requests received via POST requests, preprocesses and returns the output as a list
        '''
        def __call__(self, data):
            if self.map_to_preproc_fn[task] is not None:
                return map(self.map_to_preproc_fn[task] ,data)
            else:
                return data
        def preprocess_image_classification_single(self, file):
            image = file.get("data") or file.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = base64.b64decode(image)
            # If the image is sent as bytesarray
            if isinstance(image, (bytearray, bytes)):
                image = Image.open(io.BytesIO(image))
            return image

        #Function to read .txt file and convert it into a string
        def preprocess_sentiment_analysis_single(self, file):
            text = file.get("data") or file.get("body")
            # Decode text if not a str but bytes or bytearray
            if isinstance(text, (bytes, bytearray)):
                text = text.decode("utf-8", errors='ignore')
            return text

    class HFPipelinePostprocessDispatcher:
        '''
            `map_to_postproc_fn` returns a function for the task that can postprocess a single inference output to convert it to serializeable form with proper JSON formatting,
            else if no postprocessing is required, `map_to_postproc_fn` returns None
        '''
        def __init__(self):
            self.map_to_postproc_fn = {
                "image-classification":None,
                "sentiment-analysis":None,
                "text-classification":None
            }
            pass 
        def __call__(self, data):
            if self.map_to_postproc_fn[task] is not None:
                return map(self.map_to_postproc_fn[task], data)
            else:
                return data
    
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.preprocess_dispatcher = self.HFPipelinePreprocessDispatcher()
        self.postprocess_dispatcher = self.HFPipelinePostprocessDispatcher()


    def load_model(self, device_id, model_name, hf_models_folder = "/home/model-server/HF-models"):
        logger.info('Entered `load_model` function')
        model_folder = os.path.join(hf_models_folder, model_name)

        logger.info("Creating pipeline")
        pipe = pipeline(task=task, framework=framework, model=model_folder, device = device_id)
        logger.info("Successfully loaded DistilBERT model from HF hub")
        return pipe

    def initialize(self, context):
        '''
        context.system_properties['gpu_id'] is decided by Torchserve server to utilize 
        all available GPUs for inference equally:
        https://github.com/pytorch/serve/blob/master/docs/custom_service.md#handling-model-execution-on-multiple-gpus
        '''
        properties = context.system_properties
        

        self.map_location = (
            "cuda"
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else "cpu"
        )
        self.device_id = ( -1 if self.map_location is "cpu" 
            else int(properties.get("gpu_id"))
        )

        self.manifest = context.manifest


        #Loading model on the 'device' decided by Torchserve
        #----------------------------------------
        self.initialized = False

        model_name = context.model_name
        self.model = self.load_model(self.device_id, model_name)

        self.initialized = True
        #----------------------------------------


    def preprocess(self, data):
        '''
        Need to write code to convert the input batch into List[str] that can be processed by the `pipeline` as in this example:
        https://huggingface.co/spaces/lewtun/twitter-sentiments/blob/main/app.py#L34
        '''

        #Assuming `data` to be List of txt files, where each txt file contains a single raw inference request input

        logger.info('Preprocessing raw inference requests')
        data = self.preprocess_dispatcher(data)
        logger.info('Successfully preprocessed raw inference requests')
        
        return list(data)

    def inference(self, data):
        logger.info(f"Data received by `inference` function is: {data}")
        preds = self.model(data)

        return preds

    def postprocess(self, data):
        data = self.postprocess_dispatcher(data)
        
        return data

    
    '''
    The `handle` function can also be overrided if we want to support multiple inputs for each Inference requests, for example,
    `left` and `right` inputs supported in this project: https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-siamese-network-with-torchserve#torchserve-in-action
    Multiple outputs can also be returned as in this example:
    https://github.com/pytorch/serve/issues/1647
    '''