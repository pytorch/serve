'''
References: 
(i) https://github.com/pytorch/serve/blob/master/docs/custom_service.md
(ii) https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py 
(iii) https://github.com/pytorch/serve/blob/master/ts/torch_handler/text_handler.py
(iv) https://github.com/cceyda/lit-NER/blob/master/lit_ner/serve_pretrained.py

Reference that implements handler class without inheriting the `BaseHandler`:
(i) https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-siamese-network-with-torchserve/blob/main/deployment/handler.py 


Batching 🤗 model & Torchserve inputs in the handler: https://youtu.be/M6adb1j2jPI
https://github.com/pytorch/serve/issues/1783

'''

import os
import torch
import base64
import io
from PIL import Image

from ts.torch_handler.base_handler import BaseHandler
from transformers import *


#Source: https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/pipelines#transformers.pipeline.task
pipeline_module_map = {
    "audio-classification" : AudioClassificationPipeline,
    "automatic-speech-recognition" : AutomaticSpeechRecognitionPipeline,
    "conversational" : ConversationalPipeline,
    "feature-extraction" : FeatureExtractionPipeline,
    "fill-mask" : FillMaskPipeline,
    "image-classification" : ImageClassificationPipeline,
    "question-answering": QuestionAnsweringPipeline,
    "table-question-answering": TableQuestionAnsweringPipeline,
    "text2text-generation": Text2TextGenerationPipeline,
    "text-classification" : TextClassificationPipeline,
    "sentiment-analysis": TextClassificationPipeline,
    "text-generation" : TextGenerationPipeline,
    "token-classification" : TokenClassificationPipeline,
    "ner": TokenClassificationPipeline,
    "translation": TranslationPipeline,
    "translation_xx_to_yy": TranslationPipeline,
    "summarization": SummarizationPipeline,
    "zero-shot-classification": ZeroShotClassificationPipeline
}

task="image-classification"
framework="pt"

try:
    assert task in pipeline_module_map.keys()
except:
    print("Enter a task supported by 🤗 pipeline")
    exit(0)

class ViTImageClassifier(BaseHandler):
    def __init__(self):
        super().__init__()
        self.tokenizer = None

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

    #Function to read .txt file and convert it into a string
    #Reference: https://stackoverflow.com/questions/8369219/how-to-read-a-text-file-into-a-string-variable-and-strip-newlines
    def read_text_file(self, filename):
        with open(filename, 'r') as file:
            readfile = file.read().replace('\n', '')
        return readfile

    def read_image_file(self, file):
        image = file.get("data") or file.get("body")
        if isinstance(image, str):
            # if the image is a string of bytesarray.
            image = base64.b64decode(image)

        # If the image is sent as bytesarray
        if isinstance(image, (bytearray, bytes)):
            image = Image.open(io.BytesIO(image))

        return image

    '''
    This has to be changed according to the task, for example, in case of text file, need to convert it into list, in case of images, it needs to be directly given the image
    '''
    def preprocess(self, data):
        '''
        Need to write code to convert the input batch into List[str] that can be processed by the `pipeline` as in this example:
        https://huggingface.co/spaces/lewtun/twitter-sentiments/blob/main/app.py#L34
        '''

        #Assuming `data` to be List of txt files, where each txt file contains a single input whose sentiments are to be predicted
        
        
        #Reference: https://www.geeksforgeeks.org/python-map-function/
        # print('Preprocessing request txt file')
        # data = map(self.read_text_file, data)
        # print('Successfully preprocessed request txt file')

        print('Preprocessing request image files')
        data = map(self.read_image_file, data)
        print('Successfully preprocessed request image files')
        
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
        return data

    
    '''
    The `handle` function can also be overrided if we want to support multiple inputs for each Inference requests, for example,
    `left` and `right` inputs supported in this project: https://github.com/aws-samples/amazon-sagemaker-endpoint-deployment-of-siamese-network-with-torchserve#torchserve-in-action

    Multiple outputs can also be returned as in this example:
    https://github.com/pytorch/serve/issues/1647
    '''
