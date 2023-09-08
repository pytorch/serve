from custom_handler import ResNet50Classifier
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext
import io
from pathlib import Path
import torch
from unittest.mock import MagicMock
from ts.handler_utils.timer import timed

import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent.parent
EXAMPLE_ROOT_DIR = REPO_ROOT_DIR.joinpath("examples", "benchmarking", "resnet50")
TEST_DATA = "/home/ubuntu/serve/examples/benchmarking/resnet50/data.txt"
MODEL_PT_FILE = "trt_model_fp16.pt"
#MODEL_PT_FILE= "resnet50-11ad3fa6.pth"

def prepare_data(batch_size):
    f = io.open(TEST_DATA, "rb", buffering = 0)
    read_data = f.read()
    data = []
    for i in range(batch_size):
        tmp = {}
        tmp["data"] = read_data
        data.append(tmp)
    return data


def benchmark(batch_size):

    handler = ResNet50Classifier()
    ctx = MockContext(
        model_pt_file=MODEL_PT_FILE,
        model_dir=EXAMPLE_ROOT_DIR.as_posix(),
        model_file="model.py",
        #model_yaml_config_file="model-config.yaml"
    )

    torch.manual_seed(42 * 42)
    handler.initialize(ctx)
    handler.context = ctx

    x = prepare_data(batch_size)

    handle(handler, x, batch_size)

def handle(handler, x, batch_size):

    warm_up = 100
    print(f"Warm up for {warm_up} iterations")
    for i in range(warm_up):
        y = handler.preprocess(x)
        y = handler.inference(y)
        y = handler.postprocess(y)
    
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    x = handler.preprocess(x)
    x = handler.inference(x)
    x = handler.postprocess(x)

    end.record()
    torch.cuda.synchronize()
    duration = start.elapsed_time(end)
    print(f"Execuation time for batch size {batch_size} in ms {duration}") 
    print(f"Size of output is {len(x)}")

if __name__=="__main__":

    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        benchmark(batch_size)