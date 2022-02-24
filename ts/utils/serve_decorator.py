"""
@serve decorator
Original inspiration 
https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_pytorch_serving_container/torchserve.py
"""
from functools import wraps
from typing import List
import os
import logging 

logger = logging.getLogger()


def serve(func):
    @wraps(func)
    def serve_model(*args, **kwargs):
        config = create_torchserve_config()
        model = archive_model()
        start_torchserve(models=model, config=config)
        return NotImplemented

# TODO: Create decorator for model archiver as well?
@serve
def handle_test():
    return NotImplemented


def start_torchserve(handler="base_handler", model_store="model_store", ts_config="config.properties", model_mar="model.mar"):
    """
    Wrapper on top of torchserve --start args
    """
    ts_config = create_torchserve_config()
    
    ts_command = f'torchserve --start'
    f'--model_store {model_store}'
    f'--ts_config_file {ts_config}'
    f'--models {model_mar}'

    logger.info(ts_command)

    # TODO: Add error handling
    os.system(ts_command)



def create_torchserve_config(inference_http_port : int = 8080, management_http_port : int = 8081, batch_size : int = 1):
    """"
    Create a torchserve config.properties
    Currently this only supports inference and management port setting but 
    eventually should support everything in config manager
    """
    config = {
        f"inference_address": "http://0.0.0.0:{inference_http_port}",
        f"management_address": "http://0.0.0.0:{management_http_port}",
    }

    logger.info(config)

    for key, value in config.items():
        with open("config.properties", "w") as f:
            f.write(f"{key}={value}\n")
    

def archive_model(model_name : str = "model", handler : str = "base_handler", version : int = 1, extra_files : List[str] = []):
    """
    wrapper on top of torch-model-archiver
    """
    arch-command = f'torch-model-archiver'

    return NotImplemented

def torchserve_request(endpoint : str= "", extra_params : str = ""):
    """
    Make an inference or management request using async library
    request is synchronous so won't work well with dynamic batching
    Maybe don't need this function and can just make things clearer in documentation
    """
    return NotImplemented
