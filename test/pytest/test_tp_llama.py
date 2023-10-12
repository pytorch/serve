import os
import pytest
import sys
import yaml
from multiprocessing import Process, Queue
from pathlib import Path

import torch

from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext


CURR_FILE_PATH = Path(__file__).parent
LLAMA_PATH = (CURR_FILE_PATH.parents[1] / "examples" / "large_models" / "tp_llama")
sys.path.append(LLAMA_PATH.as_posix())

converted_checkpoints_path = "llama/converted_checkpoints"

YAML_CONFIG = f"""
#frontend settings
minWorkers: 1
maxWorkers: 1
maxBatchDelay: 200
responseTimeout: 300
parallelType: "tp"
deviceType: "gpu"

torchrun:
    nproc-per-node: 1

handler:
    converted_ckpt_dir: "{converted_checkpoints_path}"
    tokenizer_path: "llama/tokenizer.model"
    model_args_path: "llama/model_args.json"
    max_new_tokens: 50
    temperature: 0.0
    top_p: 0.9
    manual_seed: 40
    mode: "text_completion" #choices are text_completion, chat
"""

def call_handler(rank: int, world_size: int, queue: Queue, yaml_path: str):
    from llama_handler import LlamaHandler
    handler = LlamaHandler()
    ctx = MockContext(
        model_pt_file=None,
        model_dir=LLAMA_PATH.as_posix(),
    )
    
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    
    ctx.model_yaml_config = config
    
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "1234"
    

    torch.manual_seed(42 * 42)
    handler.initialize(ctx)

    handler.context = ctx
    
    queue.put(handler.inference(["The capital of France", "what is the recipes for Mayonnaise?"]))
    
    
@pytest.mark.skipif(not (LLAMA_PATH / converted_checkpoints_path).exists(), reason="Required files are not present")
def test_tp_llama(tmp_path):
    world_size = 2
    
    model_config_yaml = tmp_path / "model-config.yaml"
    model_config_yaml.write_text(YAML_CONFIG)
    
    q = Queue()
    
    procs = [Process(target=call_handler, args=(rank,world_size,q,model_config_yaml,)) for rank in range(world_size)]
    
    for p in procs:
        p.start()
        
    for p in procs:
        p.join()
    
    results = []
    while not q.empty():
        results.append(q.get())
    
    assert len(results) == 2
    
    assert results[0][0]["generation"] == ", Paris, is a city of romance, art, and culture. It is also a city of fashion, food, and fun. Paris is a city that has something for everyone.\nParis is a city that is full of history."
    assert results[0][1]["generation"] == "\nI have a recipe for mayonnaise that I use all the time. It is very easy and tastes great.\n1. In a bowl, whisk together the egg yolks, mustard, lemon ju"
    