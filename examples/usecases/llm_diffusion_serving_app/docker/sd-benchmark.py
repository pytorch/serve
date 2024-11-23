"""
Stable Diffusion Benchmark Script.

Prerequisites:
- See https://github.com/pytorch/serve/tree/master/examples/usecases/llm_diffusion_serving_app/README.md
- This script assumes models are available in the mounted volume at
      /home/model-server/model-store/stabilityai---stable-diffusion-xl-base-1.0/model

This script benchmarks Stable Diffusion model inference across different execution modes:
- Eager mode (standard PyTorch)
- Torch.compile with Inductor backend
- Torch.compile with OpenVINO backend

Results are saved in a timestamped directory including:
- JSON file with complete benchmark data
- Generated images for each mode
- Profiling data when enabled

Usage:
    python sd-benchmark.py [--num_iter N] [--run_profiling]
"""

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
from torch.profiler import profile, record_function, ProfilerActivity
import openvino.torch  # noqa: F401  # Import to enable optimizations from OpenVINO
from PIL import Image
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler


class RunMode(Enum):
    EAGER = "eager"
    TC_INDUCTOR = "tc_inductor"
    TC_OPENVINO = "tc_openvino"


def setup_pipeline(run_mode: str, ckpt: str, dtype=torch.float16) -> DiffusionPipeline:
    """Setup function remains unchanged"""
    print(f"\nInitializing pipeline with mode: {run_mode}")

    if run_mode == RunMode.TC_OPENVINO.value:
        compile_options = {
            "backend": "openvino",
            "options": {"device": "CPU", "config": {"PERFORMANCE_HINT": "LATENCY"}},
        }
        print(f"Using OpenVINO backend with options: {compile_options}")
    elif run_mode == RunMode.TC_INDUCTOR.value:
        compile_options = {"backend": "inductor", "options": {}}
        print(f"Using Inductor backend with options: {compile_options}")
    else:  # eager mode
        compile_options = {}
        print("Using eager mode (no compilation)")

    unet = UNet2DConditionModel.from_pretrained(f"{ckpt}/lcm/", torch_dtype=dtype)
    pipe = DiffusionPipeline.from_pretrained(ckpt, unet=unet, torch_dtype=dtype)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    if run_mode != RunMode.EAGER.value:
        print("Torch.Compiling models...")
        pipe.text_encoder = torch.compile(pipe.text_encoder, **compile_options)
        pipe.unet = torch.compile(pipe.unet, **compile_options)
        pipe.vae.decode = torch.compile(pipe.vae.decode, **compile_options)

    pipe.to("cpu")
    return pipe


def run_inference(
    pipe: DiffusionPipeline,
    params: Dict,
    iteration: int = 0,
    enable_profiling: bool = False,
) -> Tuple[Image.Image, float, Dict]:
    """Run inference and measure time, with optional profiling"""
    profiler_output = None

    if enable_profiling:
        print(f"\nRunning inference with profiling for iteration: {iteration}")
        with profile(
            activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
        ) as prof:
            with record_function("model_inference"):
                start_time = time.time()
                image = pipe(
                    params["prompt"],
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    height=params["height"],
                    width=params["width"],
                ).images[0]
                end_time = time.time()

        # Process profiler data
        profiler_output = {
            "cpu_time": prof.key_averages().table(
                sort_by="cpu_time_total", row_limit=20
            ),
            "memory": prof.key_averages().table(
                sort_by="cpu_memory_usage", row_limit=20
            ),
        }
    else:
        print(f"\nRunning inference for iteration: {iteration}")
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

    return image, execution_time, profiler_output


def run_benchmark(
    run_mode: str, params: Dict, num_iter: int, enable_profiling: bool = False
) -> Dict:
    """Run a single benchmark configuration with multiple iterations"""

    try:
        pipe = setup_pipeline(run_mode, params["ckpt"])

        # Warm-up run
        print("\nPerforming warm-up run...")
        warmup_image, warmup_time, _ = run_inference(pipe, params, iteration=0)

        # Benchmark iterations
        print(f"\nRunning {num_iter} benchmark iterations...")
        iteration_times = []
        final_image = None
        profiler_data = None

        for i in range(1, num_iter + 1):
            image, exec_time, profiler_data = run_inference(
                pipe,
                params,
                iteration=i,
                enable_profiling=(
                    enable_profiling and i == 1
                ),  # if profile is enabled, run for 1 iteration only
            )

            iteration_times.append(exec_time)

            if i == num_iter:  # Save final image from the last iteration
                final_image = image

        # Calculate statistics
        stats = {
            "mean": float(np.mean(iteration_times)),
            "std": float(np.std(iteration_times)),
            "all_iterations": iteration_times,
        }

        benchmark_results = {
            "run_mode": run_mode,
            "warmup_time": warmup_time,
            "statistics": stats,
            "final_image": final_image,
            "profiler_data": profiler_data if profiler_data else None,
            "status": "success",
        }

        return benchmark_results
    except Exception as e:
        print(f"Error during benchmark: {str(e)}")
        return {"run_mode": run_mode, "status": "failed", "error": str(e)}


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

    output = subprocess.check_output(["head", "-n", "1", "/proc/meminfo"]).decode(
        "utf-8"
    )
    total_memory = int(output.split()[1]) / (1024.0**2)
    total_memory_str = f"{total_memory:.2f} GB"

    return {
        "cpu_model": cpu_model,
        "cpu_count": cpu_count,
        "threads_per_core": threads_per_core,
        "cores_per_socket": cores_per_socket,
        "socket_count": socket_count,
        "total_memory": total_memory_str,
    }


def get_sw_versions():
    sw_versions = {}
    packages = [
        ("TorchServe", "torchserve"),
        ("OpenVINO", "openvino"),
        ("PyTorch", "torch"),
        ("Transformers", "transformers"),
        ("Diffusers", "diffusers"),
    ]

    sw_versions["Python"] = sys.version.split()[0]

    for name, package in packages:
        try:
            version = importlib.metadata.version(package)
            sw_versions[name] = version
        except Exception as e:
            sw_versions[name] = "Not installed"
            print(f"Exception trying to get {package} version. Error: {e}")

    return sw_versions


def save_results_1(results: List[Dict], hw_config: Dict, sw_versions: Dict):
    """
    Save benchmark results to a timestamped directory

    Args:
        results: List of benchmark results for different run modes
        hw_config: Dictionary containing hardware configuration details
        sw_versions: Dictionary containing software version information
    """
    out_dir = "/home/model-server/model-store/"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results directory with timestamp
    results_dir = os.path.join(out_dir, f"benchmark_results_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    # Save main results JSON
    results_file = os.path.join(results_dir, "benchmark_results.json")
    benchmark_data = [
        {k: result.get(k) for k in ["run_mode", "warmup_time", "statistics"]}
        for result in results
    ]

    full_results = {
        "timestamp": datetime.now().isoformat(),
        "hardware_config": hw_config,
        "software_versions": sw_versions,
        "benchmark_results": benchmark_data,
    }

    with open(results_file, "w") as f:
        json.dump(full_results, f, indent=2)

    # Copy images and profiler data for each run mode
    for result in results:
        if result["status"] == "success":
            run_mode = result["run_mode"]

            # Save the final image
            if result.get("final_image"):
                image_filename = f"image-{run_mode}-final.png"
                result.get("final_image").save(
                    os.path.join(results_dir, image_filename)
                )

            # Save profiler data
            if result.get("profiler_data"):
                profiler_data = result.get("profiler_data")
                profiler_filename = f"profile-{run_mode}.txt"
                with open(os.path.join(results_dir, profiler_filename), "w") as f:
                    f.write(
                        "CPU Time Profile (sort_by='cpu_time_total', row_limit=20):\n"
                    )
                    f.write(profiler_data["cpu_time"])
                    f.write(
                        "\n\nMemory Usage Profile (sort_by='cpu_memory_usage', row_limit=20):\n"
                    )
                    f.write(profiler_data["memory"])

    print(f"\nResults saved in directory: {results_dir}")
    print(f"Files in the {results_dir} directory:")
    for file in sorted(os.listdir(results_dir)):
        print(file)


def main():
    # Parse command-line args
    parser = argparse.ArgumentParser(description="Stable Diffusion Benchmark script")
    parser.add_argument(
        "-ni", "--num_iter", type=int, default=3, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "-rp",
        "--run_profiling",
        action="store_true",
        help="Run benchmark with profiling",
    )
    args = parser.parse_args()

    # Number of benchmark iterations
    num_iter = 1 if args.run_profiling else args.num_iter

    out_dir = "/home/model-server/model-store/"

    run_modes = [
        RunMode.EAGER.value,
        RunMode.TC_INDUCTOR.value,
        RunMode.TC_OPENVINO.value,
    ]

    params = {
        "ckpt": "/home/model-server/model-store/stabilityai---stable-diffusion-xl-base-1.0/model",
        "guidance_scale": 5.0,
        "num_inference_steps": 4,
        "height": 768,
        "width": 768,
        "prompt": "A close-up HD shot of a vibrant macaw parrot perched on a branch in a lush jungle ",
        "dtype": torch.float16,
    }
    # params["prompt"] = "A close-up of a blooming cherry blossom tree in full bloom"

    results = []
    for mode in run_modes:
        print("\n" + "=" * 80)
        print(
            f"Running benchmark with run mode: {mode}, num_iter: {num_iter}, run_profiling: {args.run_profiling}"
        )
        print("=" * 80)
        result = run_benchmark(mode, params, num_iter, args.run_profiling)
        results.append(result)
        print("-" * 80)

    # Hardware and Software Info
    print("\nHardware Info:")
    print("-" * 80)
    hw_config = get_hw_config()
    for key, value in hw_config.items():
        print(f"{key}: {value}")

    print("\nSoftware Versions:")
    sw_versions = get_sw_versions()
    print("-" * 80)
    for name, version in sw_versions.items():
        print(f"{name}: {version}")

    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 80)
    table_data = []
    for result in results:
        if result["status"] == "success":
            row = [
                result["run_mode"],
                f"{result['warmup_time']:.2f} seconds",
                f"{result['statistics']['mean']:.2f} +/- {result['statistics']['std']:.2f} seconds",
            ]
        else:
            row = [
                result["run_mode"],
                "Failed",
                result["error"],
            ]
        table_data.append(row)

    headers = ["Run Mode", "Warm-up Time", f"Average Time for {num_iter} iter"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Save results
    save_results_1(results, hw_config, sw_versions)
    if args.run_profiling:
        print("\nnum_iter is set to 1 as run_profiling flag is enabled !")
    print(
        f"\nResults saved at {out_dir} which is a Docker container mount, corresponds to 'serve/model-store-local/' on the host machine.\n"
    )


if __name__ == "__main__":
    main()
