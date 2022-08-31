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

from ts.torch_handler.base_handler import BaseHandler
from transformers import pipeline


#Reference: https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline.task
pipeline_supported_tasks= [
    "image-classification",
    "sentiment-analysis"
]

task="<PLACEHOLDER>"
framework="<PLACEHOLDER>"

try:
    assert task in pipeline_supported_tasks
except:
    print("Enter a task supported by 🤗 pipeline")
    exit(0)

class HFPipelinePreprocessFactory:
    def __init__(self):
        '''
        `map_to_preproc_fn` returns a function for the task that can preprocess a single raw inference request
        '''
        self.map_to_preproc_fn = {
            "image-classification":self.preprocess_image_classification_single,
            "sentiment-analysis":self.preprocess_sentiment_analysis_single
        }
        pass
    '''
    Takes a list of raw inference requests received via POST requests, preprocesses and returns the output as a list
    '''
    def __call__(self, data):
        if self.map_to_postproc_fn[task] is not None:
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
        with open(file, 'r') as file:
            readfile = file.read().replace('\n', '')
        return readfile
    


class HFPipelinePostprocessFactory:
    '''
        `map_to_postproc_fn` returns a function for the task that can postprocess a single inference output to convert it to serializeable form with proper JSON formatting,
        if no postprocessing is required, `map_to_postproc_fn` returns None
    '''
    def __init__(self):
        self.map_to_postproc_fn = {
            "image-classification":None,
            "sentiment-analysis":None
        }
        pass 
    def __call__(self, data):
        if self.map_to_postproc_fn[task] is not None:
            return map(self.map_to_postproc_fn[task], data)
        else:
            return data


class HFPipelineHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.preprocess_factory = HFPipelinePreprocessFactory()
        self.postprocess_factory = HFPipelinePostprocessFactory()


    def load_model(self, device_id, model_name, hf_models_folder = "/home/model-server/HF-models"):
        print('Entered `load_model` function')
        model_folder = os.path.join(hf_models_folder, model_name)

        print("Creating pipeline")
        pipe = pipeline(task=task, framework=framework, model=model_folder, device = device_id)
        print("Successfully loaded DistilBERT model from HF hub")
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

        print('Preprocessing raw inference requests')
        data = self.preprocess_factory(data)
        print('Successfully preprocessed raw inference requests')
        
        return list(data)

    def inference(self, data):
        print(f"Data received by `inference` function is: {data}")
        preds = self.model(data)

        
        '''
        `preds` is something like this for MobileViT XX Small pipeline predictions for 1 inference request:
        [[{'score': 0.5584330558776855, 'label': 'remote control, remote'}, 
        {'score': 0.11318826675415039, 'label': 'joystick'}, 
        {'score': 0.08971238136291504, 'label': 'mouse, computer mouse'}, 
        {'score': 0.027173032984137535, 'label': 'ocarina, sweet potato'}, 
        {'score': 0.019073771312832832, 'label': 'pick, plectrum, plectron'}]]
        '''

        return preds

    def postprocess(self, data):
        data = self.postprocess_factory(data)
        return data

    
    '''
    The `handle` function can also be overrided if we want to support multiple inputs for each Inference requests, for example,
    `left` and `right` inputs supported in this project: https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-siamese-network-with-torchserve#torchserve-in-action
    Multiple outputs can also be returned as in this example:
    https://github.com/pytorch/serve/issues/1647
    '''