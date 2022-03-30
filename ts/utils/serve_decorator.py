"""
@serve decorator
Original inspiration 
https://github.com/aws/sagemaker-pytorch-inference-toolkit/blob/master/src/sagemaker_pytorch_serving_container/torchserve.py
https://fastapi.tiangolo.com/
"""
from functools import wraps
from typing import List, Callable, Union
import os
import logging 
import inspect
from ts.torch_handler.base_handler import BaseHandler


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
    3. Archive a model assuming weights are in disk
    4. Start torchserve

    # TODO: Put everything in a temporary directory so user files don't get overwritten
    """
    @wraps(func)
    def serve_model(*args, **kwargs):
        """
        Can override inference and management port or model name here
        @serve(inference_http_port=8080, management_port=8081, serialized_file="my_model.pt", batch_size=32)
        Full available configurations can be found in ts_config`

        If you need to serve a full handler you can instead call create_handler, create_torchserve_config and torchserve_start manually
        """
        
        handler = create_handler(func)

        config = create_torchserve_config(inference_http_port=7070, management_http_port=7071, batch_size=32)
        
        model = archive_model(serialized_file="model.pt")

        torchserve_start(handler= handler, models=model, config=config)

        logger.warning("Don't forget to call torchserve --stop when you're done")
        
        return 0


def create_handler(inference_func : Union[Callable, BaseHandler], handler_file : str = "handler.py"):
    """
    Create a new handler that inherits from the base handler
    Update the inference() function with the function captured by the decorator
    Write the handler to disk

    TODO: Improve story for adding custom dependencies, all the hidden imports or library code
    TODO: Improve workflow story, coming in future PR

    """
    
    # If user provides a function then override inference() function in base handler
    if type(inference_func) is Callable:
        handler_class = BaseHandler()
        handler_class.inference = inference_func
    
    # If user provides a class then create a new custom handler
    else:
        handler_class = inference_func

    # Get source code for new class and write it to disk
    source = inspect.getsource(handler_class)
    with open(handler_file, "w") as f:
        f.write(source)
    
    return handler_file

def writetofile(filename : str = "model.py"):
    """
    @writetofile decorator to add on top of a torch.nn.Module to write it to disk as model.py
    Use this instead on top of another handler
    """
    @wraps(cls)
    def archive_model(*args, **kwargs):
        class_source = inspect.getsource(cls)
        with open(filename, "w") as f:
            f.write(class_source)


def torchserve_start(handler : str, model_store : str ="model_store", ts_config : str ="config.properties", model_mar : str ="model.mar"):
    """
    Wrapper on top of torchserve --start args
    If you already have your own model packaged
    """

    ts_command = f'torchserve --start --foreground'
    f'--model_store {model_store}'
    f'--ts_config_file {ts_config}'
    f'--models {model_mar}'
    f'--handler {handler}'

    logger.info(ts_command)

    # TODO: Add error handling
    os.system(ts_command)

    print("torchserve has started")



def create_torchserve_config(inference_http_port : int = 8080,
                             management_http_port : int = 8081,
                             model_mar : str = "model.mar",
                             netty_threads : int = 32,
                             batch_size : int = 1, 
                             batch_delay : int = 50,
                             num_workers : int = 1,
                             response_timeout : int = 150,
                             config_properties : str ="config.properties"):
    """"
    Create a torchserve config.properties
    Currently this only supports inference and management port setting but 
    eventually should support everything in config manager
    """
    
    ## Torchserve specific configs
    ts_config = {
        f"inference_address": "http://0.0.0.0:{inference_http_port}",
        f"management_address": "http://0.0.0.0:{management_http_port}",
        f"number_of_netty_threads" :"{netty_threads}",
        f"job_queue_size" : "1000",
        f"vmargs" : "-Xmx4g -XX:+ExitOnOutOfMemoryError -XX:+HeapDumpOnOutOfMemoryError",
        f"prefer_direct_buffer" : "True",
        f"default_response_timeout" : "300",
        f"unregister_model_timeout" : "300",
        f"install_py_dep_per_model" : "true"
    }

    ## Model specific configs

    model_config = {
            "defaultVersion": "true",
            f"marName": {model_mar},
            f"minWorkers": {num_workers},\
            f"maxWorkers": {num_workers},\
            f"batchSize": {batch_size},\
            f"maxBatchDelay": {batch_delay},\
            f"responseTimeout": {response_timeout},\
    }

    logger.info(ts_config)

    logger.info(model.config)

    # Torchserve configurations
    for key, value in ts_config.items():
        with open("config.properties", "w") as f:
            f.write(f"{key}={value}\n")
    


    
    return config_properties
    

def archive_model(serialized_file : str, model_file : str = None, model_name : str = "model", handler : str = "base_handler", version : int = 1, extra_files : List[str] = []):
    """
    wrapper on top of torch-model-archiver
    model_file is only needed for eager mode execution
    TODO: Plug this into writetofile() decorator so people can write their model.py to disk
    TODO: Feels strange to ask for serialized weights all the time because it means we can't archive regular python function
    """

    # Eager mode model
    if model_file:
        arch_command = f"torch-model-archiver --model_file {model_file} --serialized_file {serialized_file} --model-name {model_name} --handler {handler} --version {version} --extra-files {extra_files}"
    
    # Torchscripted model
    else:
        arch_command = f"torch-model-archiver --serialized_file {serialized_file} --model-name {model_name} --handler {handler} --version {version} --extra-files {extra_files}"
    
    logger.info(arch_command)

    os.system(arch_command)

    print("archived model {model_file}")
    
    return 0

def torchserve_stop():
    os.system("torchserve --stop")
    
