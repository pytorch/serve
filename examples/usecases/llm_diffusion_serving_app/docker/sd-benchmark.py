import argparse
import importlib.metadata
import json
import numpy as np
import os
import subprocess
import sys
import time
from tabulate import tabulate
from datetime import datetime
from enum import Enum
from typing import Dict, Tuple, List

import torch
import openvino.torch
from PIL import Image
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler

class RunMode(Enum):
    EAGER = "eager"
    TC_INDUCTOR = "tc_inductor"
    TC_OPENVINO = "tc_openvino"

def setup_pipeline(run_mode: str, ckpt: str, dtype=torch.float16) -> DiffusionPipeline:
    """
    Setup the diffusion pipeline based on run mode configuration
    
    Args:
        run_mode: One of 'eager', 'tc_inductor', or 'tc_openvino'
        ckpt: Path to the model checkpoint
        dtype: Model dtype
    """
    print(f"\nInitializing pipeline with mode: {run_mode}")
    
    # Set compile options based on run mode
    if run_mode == RunMode.TC_OPENVINO.value:
        compile_options = {
            'backend': 'openvino',
            'options': {'device': 'CPU', 'config': {'PERFORMANCE_HINT': 'LATENCY'}}
        }
        print(f"Using OpenVINO backend with options: {compile_options}")
    elif run_mode == RunMode.TC_INDUCTOR.value:
        compile_options = {'backend': 'inductor', 'options': {}}
        print(f"Using Inductor backend with options: {compile_options}")
    else:  # eager mode
        compile_options = {}
        print("Using eager mode (no compilation)")
    
    # Initialize models
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt}/lcm/", torch_dtype=dtype)
    pipe = DiffusionPipeline.from_pretrained(ckpt, unet=unet, torch_dtype=dtype)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    
    # Apply compilation if using torch.compile
    if run_mode != RunMode.EAGER.value:
        print("Compiling models...")
        pipe.text_encoder = torch.compile(pipe.text_encoder, **compile_options)
        pipe.unet = torch.compile(pipe.unet, **compile_options)
        pipe.vae.decode = torch.compile(pipe.vae.decode, **compile_options)
    
    pipe.to("cpu")
    return pipe

def run_inference(pipe: DiffusionPipeline, params: Dict, iteration: int = 0) -> Tuple[Image.Image, float]:
    """Run inference and measure time"""
    start_time = time.time()
    image = pipe(
        params["prompt"],
        num_inference_steps=params["num_inference_steps"],
        guidance_scale=params["guidance_scale"],
        height=params["height"],
        width=params["width"],
    ).images[0]
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Iteration {iteration} execution time: {execution_time:.2f} seconds")
    
    return image, execution_time

def run_benchmark(run_mode: str, params: Dict, num_iter: int) -> Dict:
    """Run a single benchmark configuration with multiple iterations"""
    out_dir = "/home/model-server/model-store/"
    
    try:
        pipe = setup_pipeline(
            run_mode,
            params["ckpt"],
            params["dtype"]
        )
        
        # Warm-up run
        print("\nPerforming warm-up run...")
        warmup_image, warmup_time = run_inference(pipe, params, iteration=0)
        
        # Benchmark iterations
        print(f"\nRunning {num_iter} benchmark iterations...")
        iteration_times = []
        final_image = None
        
        for i in range(num_iter):
            image, exec_time = run_inference(pipe, params, iteration=i+1)
            iteration_times.append(exec_time)
            if i == num_iter - 1:
                final_image = image
        
        # Calculate statistics
        stats = {
            "mean": float(np.mean(iteration_times)),
            "std": float(np.std(iteration_times)),
            "all_iterations": iteration_times
        }
        
        # Save images
        final_image_filename = f"image-{run_mode}-final.png"
        final_image.save(os.path.join(out_dir, final_image_filename))
        
        return {
            "run_mode": run_mode,
            "warmup_time": warmup_time,
            "statistics": stats,
            "final_image": final_image_filename,
            "status": "success"
        }
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        return {
            "run_mode": run_mode,
            "status": "failed",
            "error": str(e)
        }

def get_hw_config():
    output = subprocess.check_output(["lscpu"]).decode("utf-8")
    lines = output.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("Model name:"):
            cpu_model = line.split("Model name:")[1].strip()
        elif line.startswith("CPU(s):"):
            cpu_count = line.split("CPU(s):")[1].strip()
        elif line.startswith("Thread(s) per core:"):
            threads_per_core = line.split("Thread(s) per core:")[1].strip()
        elif line.startswith("Core(s) per socket:"):
            cores_per_socket = line.split("Core(s) per socket:")[1].strip()
        elif line.startswith("Socket(s):"):
            socket_count = line.split("Socket(s):")[1].strip()

    output = subprocess.check_output(["head", "-n", "1", "/proc/meminfo"]).decode("utf-8")
    total_memory = int(output.split()[1]) / (1024.0 ** 2)
    total_memory_str = f"{total_memory:.2f} GB"
    
    return {
        "cpu_model": cpu_model,
        "cpu_count": cpu_count,
        "threads_per_core": threads_per_core,
        "cores_per_socket": cores_per_socket,
        "socket_count": socket_count,
        "total_memory": total_memory_str
    }
    
def get_sw_versions():
    sw_versions = {}
    packages = [
        ("TorchServe", "torchserve"),
        ("OpenVINO", "openvino"),
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
        ("Diffusers", "diffusers")
    ]

    sw_versions["Python"] = sys.version.split()[0]
    
    for name, package in packages:
        try:
            version = importlib.metadata.version(package)
            sw_versions[name] = version
        except Exception as e:
            sw_versions[name] = "Not installed"

    return sw_versions

def save_results(results: List[Dict], hw_config: List[Dict], sw_versions: List[Dict]):
    """Save benchmark results to a JSON file"""
    out_dir = "/home/model-server/model-store/"
    filename = f"sd_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    data = {"hw_config": hw_config, "sw_versions": sw_versions, "results": results}

    with open(os.path.join(out_dir, filename), 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"\nResults saved to {os.path.join(out_dir, filename)}")
    
def main():
    # Parse command-line args
    parser = argparse.ArgumentParser(description='Stable Diffusion Benchmark script')
    parser.add_argument('-ni', '--num_iter', type=int, default=3, help='Number of benchmark iterations')
    args = parser.parse_args()
    
    # Number of benchmark iterations
    num_iter = args.num_iter
    out_dir = "/home/model-server/model-store/"
    
    # Run modes to test
    run_modes = [
        RunMode.EAGER.value,
        RunMode.TC_INDUCTOR.value,
        RunMode.TC_OPENVINO.value
    ]
    
    # Parameters
    params = {
        "ckpt": "/home/model-server/model-store/stabilityai---stable-diffusion-xl-base-1.0/model",
        "guidance_scale": 5.0,
        "num_inference_steps": 4,
        "height": 768,
        "width": 768,
        "prompt": "a close-up picture of an old man standing in the rain",
        "dtype": torch.float16
    }
    
    # Run benchmarks
    results = []
    for mode in run_modes:
        print("\n" + "="*50)
        print(f"Running benchmark with run mode: {mode}")
        print(f"Number of iterations: {num_iter}")
        print("="*50)
        
        result = run_benchmark(mode, params, num_iter)
        results.append(result)
    
    print("-"*50)

    # Hardware and Software Info
    print("\nHardware Info:")
    print("-"*50)
    hw_config = get_hw_config()
    for key, value in hw_config.items():
        print(f"{key}: {value}")
    
    print("\nSoftware Versions:")
    sw_versions = get_sw_versions()
    print("-"*50)
    for name, version in sw_versions.items():
        print(f"{name}: {version}")
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-"*50)
    table_data = []
    for result in results:
        if result["status"] == "success":
            table_data.append([
                result['run_mode'],
                f"{result['warmup_time']:.2f} seconds",
                f"{result['statistics']['mean']:.2f} +/- {result['statistics']['std']:.2f} seconds",
                result['final_image']
            ])
        else:
            table_data.append([
                result['run_mode'],
                "Failed",
                result['error'],
                "-"
            ])
            
    headers = ["Run Mode", "Warm-up Time", f"Average Time for {num_iter} iter", "Image Saved as "]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Save results
    save_results(results, hw_config, sw_versions)
    print(f"\nResults and Images saved at {out_dir} which is a Docker container mount, corresponds to 'serve/model-store-local/' on the host machine.\n")
    
if __name__ == "__main__":
    main()