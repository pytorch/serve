from custom_handler import ResNet50Classifier
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext
import io
from pathlib import Path
from PIL import Image
import torch
from unittest.mock import MagicMock
import numpy as np
import csv


import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

CURR_FILE_PATH = Path(__file__).parent
REPO_ROOT_DIR = CURR_FILE_PATH.parent.parent.parent
EXAMPLE_ROOT_DIR = REPO_ROOT_DIR.joinpath("examples", "benchmarking", "resnet50")
TEST_DATA = REPO_ROOT_DIR.joinpath("examples", "image_classifier", "kitten.jpg")
MODEL_PT_FILE = "trt_model_fp16.pt"
#MODEL_PT_FILE= "resnet50-11ad3fa6.pth"

def prepare_data(image_processing, batch_size):

    image = Image.open(TEST_DATA)
    image = image_processing(image)
    torch.save(image, "data.txt") 
    f = io.open("data.txt", "rb", buffering = 0)
    read_data = f.read()
    data = []
    for i in range(batch_size):
        tmp = {}
        tmp["data"] = read_data
        data.append(tmp)
    return data

def benchmark(artifacts, batch_size):

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

    x = prepare_data(handler.image_processing, batch_size)


    handle(handler, x, artifacts, batch_size)

def handle(handler, x, artifacts, batch_size):

    print("#################################################################################################")
    warm_up, n_runs = 100, 1000
    print(f"Batch size {batch_size}. Warm up for {warm_up} iterations")
    for i in range(warm_up):
        y = handler.preprocess(x)
        y = handler.inference(y)
        y = handler.postprocess(y)
    
    
    e_e_latency = []
    m_latency = []
    for i in range(n_runs):
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        m_start = torch.cuda.Event(enable_timing=True)
        m_end = torch.cuda.Event(enable_timing=True)
        e_start.record()

        # Pre-process
        y = handler.preprocess(x)


        #Inference
        m_start.record()
        y = handler.inference(y)
        m_end.record()
        torch.cuda.synchronize()
        m_latency.append(m_start.elapsed_time(m_end))

        #Post process
        y = handler.postprocess(y)
        e_end.record()
        torch.cuda.synchronize()
        e_e_latency.append(e_start.elapsed_time(e_end))
        p50_e_latency, p90_e_latency, p99_e_latency = np.percentile(e_e_latency, 50), np.percentile(e_e_latency, 90), np.percentile(e_e_latency, 99)
        p50_m_latency, p90_m_latency, p99_m_latency = np.percentile(m_latency, 50), np.percentile(m_latency, 90), np.percentile(m_latency, 99)
        e_mean_latency = np.mean(e_e_latency)
    throughput = 1000.0*batch_size/e_mean_latency
    print(f"Throughput is {throughput:.2f}")
    print(f"End to end latencies P50: {p50_e_latency:.2f} ms, P90: {p90_e_latency:.2f} ms, P99: {p99_e_latency:.2f} ms")
    print(f"Model latencies P50: {p50_m_latency:.2f} ms, P90: {p90_m_latency:.2f} ms, P99: {p99_m_latency:.2f} ms")

    
    artifacts.append([MODEL_PT_FILE, batch_size, warm_up, n_runs, round(throughput,2), round(p50_m_latency, 2), round(p90_m_latency, 2),
    round(p99_m_latency, 2), round(p50_e_latency, 2), round(p90_e_latency, 2), round(p99_e_latency, 2)]) 
    

if __name__=="__main__":

    artifacts = []
    artifacts.append(["Model", "Batch Size", "N_warmup", "N_runs", "Throughput", "Model latency P50", "Model latency P90",
    "Model latency P99", "End to End latency P50", "End to End latency P90", "End to End latency P99" ])
    #for batch_size in [1, 2]:
    for batch_size in [1, 2, 4, 8, 16, 32, 64]:
        benchmark(artifacts, batch_size)

    with open("report.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(artifacts)