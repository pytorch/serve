import os
import sys
import shutil

def test_worker_utilization():

  # To help discover local modules
  REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
  sys.path.append(REPO_ROOT)
  import ts_scripts.tsutils as ts

  model_store = "/tmp/model_store"
  ts_config = "/tmp/ts_config.cfg"
  model_name = "bert_seq_classification_eager"
  n_workers = os.getenv("N_WORKERS")
  
  if(n_workers):
    n_workers = int(n_workers)
  else:
    # Default for CICD Env
    n_workers = 15

  with open(ts_config, "w") as f:
    f.write("number_of_gpu=1")

  shutil.rmtree(model_store)
  os.makedirs(model_store)

  ts.start_torchserve(model_store=model_store, snapshot_file=ts_config, no_config_snapshots=True)
  ts.register_model(model_name=model_name)
  ts.get_gpu_usage(device_id=0)
  ts.scale_up_model(model_name, workers=5)
  ts.get_gpu_usage(device_id=0)
  ts.stop_torchserve()
  ts.get_gpu_usage(device_id=0)
