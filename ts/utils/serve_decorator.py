"""
@serve decorator
Original inspiration 
https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_pytorch_serving_container/torchserve.py
https://fastapi.tiangolo.com/
"""
from functools import wraps
from typing import List, Callable
import os
import logging 
import grequest # TODO: add this to requirements.txt
import inspect

logger = logging.getLogger()


def serve(func):
    """
    @serve is a decorator meant to be used on top of an inference function
    @serve
    def inference():
        perform_inference..

    The goal of @serve is to simplify the handler authoring experience in torchserve

    inference() can be as complex as you like and call other functions 

    If you already have a handler you can manually call create_torchserve_config(), archive_model() 
    and start_torchserve()

    What this will do is
    1. Override the base handler with a new handler with the inference function overloaded
    2. Create a new torchserve config 
    3. Archive a model.pt
    4. Start torchserve

    When done call torchserve_stop()

    """
    @wraps(func)
    def serve_model(*args, **kwargs):
        
        handler = create_handler(func)

        config = create_torchserve_config(inference_http_port=7070, management_http_port=7071)
        
        model = archive_model(serialized_file="model.pt")

        torchserve_start(handler= handler, models=model, config=config)
        
        # Error handle and
        # Return some error code
        return 0

def create_handler(handler : Callable, handler_file : str = "handler.py"):
    """
    Create a new handler that inherits from the base handler
    Update the inference() function with the function captured by the decorator
    Write the handler to disk

    TODO: Find a solution to make this work with workflows, pending more improvements in workflows
    """
    inference_handler = inspect.getsource(handler)
    with open(handler_file, "w") as f:
        f.write("class CustomHandler(BaseHandler):\n")
        for index, line in enumerate(inference_handler):
            if index == 0:
                f.write(f"\tdef inference(self, data, *args, **kwargs):")
            else:
                f.write(f"\t{line}")
    return handler_file

def writetofile(filename : str = "model.py"):
    """
    @writetofile decorator to add on top of a torch.nn.Module to write it to disk as model.py
    """
    @wraps(cls)
    def archive_model(*args, **kwargs):
        lines = inspect.getsource(cls)
        with open(filename, "w") as f:
            f.write(lines)


# TODO: Create decorator for model archiver as well?
@serve
def handle_test():
    # Need to figure out how to pick up the handler and replace it by the inference function that exists in the base handler
    return NotImplemented


def torchserve_start(handler : str, model_store : str ="model_store", ts_config : str ="config.properties", model_mar : str ="model.mar"):
    """
    Wrapper on top of torchserve --start args
    If you already have your own model packaged
    """

    ts_command = f'torchserve --start'
    f'--model_store {model_store}'
    f'--ts_config_file {ts_config}'
    f'--models {model_mar}'
    f'--handler {handler}'

    logger.info(ts_command)

    # TODO: Add error handling
    os.system(ts_command)



def create_torchserve_config(inference_http_port : int = 8080, management_http_port : int = 8081, batch_size : int = 1, config_properties : str ="config.properties"):
    """"
    Create a torchserve config.properties
    Currently this only supports inference and management port setting but 
    eventually should support everything in config manager
    """
    
    ## Torchserve specific configs
    ts_config = {
        f"inference_address": "http://0.0.0.0:{inference_http_port}",
        f"management_address": "http://0.0.0.0:{management_http_port}",
        f"number_of_netty_threads" :"32",
        f"job_queue_size" : "1000",
        f"vmargs" : "-Xmx4g -XX:+ExitOnOutOfMemoryError -XX:+HeapDumpOnOutOfMemoryError",
        f"prefer_direct_buffer" : "True",
        f"default_response_timeout" : "300",
        f"unregister_model_timeout" : "300",
        f"install_py_dep_per_model" : "true"
    }

    ## Model specific configs

#     model_config = {"
#         models={\
#   "model": {\
#     "1.0": {\
#         "defaultVersion": true,\
#         "marName": "model.mar",\
#         "minWorkers": 1,\
#         "maxWorkers": 1,\
#         "batchSize": 8,\
#         "maxBatchDelay": 50,\
#         "responseTimeout": 120\
#     }\
#   }\
# }

#     }"

    logger.info(ts_config)

    for key, value in ts_config.items():
        with open("config.properties", "w") as f:
            f.write(f"{key}={value}\n")
    
    return config_properties
    

def archive_model(serialized_file : str, model_file : str = None, model_name : str = "model", handler : str = "base_handler", version : int = 1, extra_files : List[str] = []):
    """
    wrapper on top of torch-model-archiver
    model_file is only needed for eager mode execution
    TODO: Plug this into writetofile() decorator so people can write their model.py to disk
    """

    # Eager mode model
    if model_file:
        arch_command = f"torch-model-archiver --model_file {model_file} --serialized_file {serialized_file} --model-name {model_name} --handler {handler} --version {version} --extra-files {extra_files}"
    
    # Torchscripted model
    else:
        arch_command = f"torch-model-archiver --serialized_file {serialized_file} --model-name {model_name} --handler {handler} --version {version} --extra-files {extra_files}"
    
    os.system(arch_command)
    
    return 0

def torchserve_stop():
    os.system("torchserve --stop")

def torchserve_request(endpoint : str= "", tasks : List[str] = [], extra_params : str = ""):
    """
    Make an inference or management request using async library
    request is synchronous so won't work well with dynamic batching and will be like setting batch size = 1 always
    Maybe don't need this function and can just make things clearer in documentation
    """
    rs = (grequest.get(task) for task in tasks)
    return [tasks, rs]
    
