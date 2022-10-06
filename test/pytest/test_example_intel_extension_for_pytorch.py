import json
import os
import subprocess
import shutil
import pytest
import requests
import test_utils
from test_handler import run_inference_using_url_with_data
from ts.torch_handler.utils.conf import CONFIGURATIONS
from ts.torch_handler.utils.optimization import OPTIMIZATIONS
import yaml
import glob

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
config_properties_ipex = os.path.join(REPO_ROOT, "test/config_ipex.properties")
data_file_kitten = os.path.join(REPO_ROOT, "examples/image_classifier/kitten.jpg")
TS_LOG = "./logs/ts_log.log"

MANAGEMENT_API = "http://localhost:8081"
INFERENCE_API = "http://localhost:8080"

ipex_launcher_available = False
cmd = ["python", "-m", "intel_extension_for_pytorch.cpu.launch", "--no_python", "ls"]
r = subprocess.run(cmd)
if r.returncode == 0:
    ipex_launcher_available = True


def setup_module():
    test_utils.torchserve_cleanup()
    test_yaml1 = ''

    test_yaml2 = '''
        dtype: bfloat16
        '''
        
    test_yaml3 = '''
        channels_last: False
        '''
    
    invalid_yaml1 = '''
        dtype: int8
        '''
    
    invalid_yaml2 = '''
        channels_last: None
        '''
    
    build_test_yaml(test_yaml1, 'test_yaml1.yaml')
    build_test_yaml(test_yaml2, 'test_yaml2.yaml')
    build_test_yaml(test_yaml3, 'test_yaml3.yaml')
    build_test_yaml(invalid_yaml1, 'invalid_yaml1.yaml')
    build_test_yaml(invalid_yaml2, 'invalid_yaml2.yaml')
    
    response = requests.get("https://torchserve.pytorch.org/mar_files/resnet-18.mar", allow_redirects=True)
    open(os.path.join(test_utils.MODEL_STORE, "resnet-18.mar"), "wb").write(response.content)
    
    response = requests.get("https://download.pytorch.org/models/resnet18-f37072fd.pth", allow_redirects=True)
    open(os.path.join(test_utils.MODEL_STORE, "resnet18-f37072fd.pth"), "wb").write(response.content)
    
    build_test_yaml(test_yaml1, 'ipex_config.yaml')

def teardown_module(module):
    test_utils.torchserve_cleanup()
    delete_all_model_store()
    delete_logs_dir()
    
def delete_all_model_store():
    for f in glob.glob(test_utils.MODEL_STORE + "/*"):
        os.remove(f)

def delete_logs_dir():
    if os.path.exists("logs"):
        shutil.rmtree('logs', ignore_errors=True)

def build_test_yaml(cfg, cfg_file_name):
    y = yaml.load(cfg, Loader=yaml.SafeLoader)
    with open(os.path.join(test_utils.MODEL_STORE, cfg_file_name), "w", encoding="utf-8") as f:
        yaml.dump(y,f)
    f.close()
    
def setup_torchserve():
    delete_logs_dir()
    test_utils.start_torchserve(
        test_utils.MODEL_STORE, config_properties_ipex, gen_mar=False
    )

def run_inference(model_name):
    files = {
        "data": (data_file_kitten, open(data_file_kitten, "rb")),
    }
    response = run_inference_using_url_with_data(
        "http://localhost:8080/predictions/{}".format(model_name), files
    )
    return response

def test_ipex_config_default():
    cfg = CONFIGURATIONS["ipex"](os.path.join(test_utils.MODEL_STORE, 'test_yaml1.yaml'))
    optimization = OPTIMIZATIONS["ipex"](cfg)
    assert optimization.dtype == 'float32'
    assert optimization.channels_last == True

def test_ipex_config_dtype():
    cfg = CONFIGURATIONS["ipex"](os.path.join(test_utils.MODEL_STORE, 'test_yaml2.yaml'))
    optimization = OPTIMIZATIONS["ipex"](cfg)
    assert optimization.dtype == 'bfloat16'
    assert optimization.channels_last == True
    
    with pytest.raises(Exception) as e:
        cfg = CONFIGURATIONS["ipex"](os.path.join(test_utils.MODEL_STORE, 'invalid_yaml1.yaml'))
    assert str(e.value) == "dtype int8 is NOT supported"

def test_ipex_config_channels_last():
    cfg = CONFIGURATIONS["ipex"](os.path.join(test_utils.MODEL_STORE, 'test_yaml3.yaml'))
    optimization = OPTIMIZATIONS["ipex"](cfg)
    assert optimization.dtype == 'float32'
    assert optimization.channels_last == False
    
    with pytest.raises(Exception) as e:
        cfg = CONFIGURATIONS["ipex"](os.path.join(test_utils.MODEL_STORE, 'invalid_yaml2.yaml'))
    assert str(e.value) == "channels last must be type bool"

@pytest.mark.skipif(
    not ipex_launcher_available,
    reason="Make sure intel-extension-for-pytorch is installed",
)
def test_inference_with_ipex_config_yaml():
    cmd = test_utils.model_archiver_command_builder(
                                                    model_name='resnet-18-ipex-fp32',
                                                    version='1.0',
                                                    model_file= os.path.join(REPO_ROOT, "examples/image_classifier/resnet_18/model.py"),
                                                    serialized_file=os.path.join(test_utils.MODEL_STORE, 'resnet18-f37072fd.pth'),
                                                    handler='image_classifier',
                                                    extra_files=os.path.join(test_utils.MODEL_STORE, "ipex_config.yaml"),
                                                    force=True
                                                    )
    assert subprocess.run(cmd.split(" ")).returncode == 0, ".mar file could not be generated with --extra_files ipex_config.yaml"
    
    setup_torchserve()
    test_utils.register_model("resnet-18-ipex-fp32", "resnet-18-ipex-fp32.mar")
    response = run_inference("resnet-18-ipex-fp32")
    assert response.status_code == 200, "inference with --extra files ipex_config.yaml failed"
    assert ("converting to channels last memory format" in open(TS_LOG).read())
    assert ("optimizing model with data type torch.float32" in open(TS_LOG).read())
    test_utils.unregister_model("resnet-18-ipex-fp32")

def get_worker_affinity(num_workers, worker_idx):
    from intel_extension_for_pytorch.cpu.launch import CPUinfo

    cpuinfo = CPUinfo()
    num_cores = cpuinfo.physical_core_nums()

    num_cores_per_worker = num_cores // num_workers
    start = worker_idx * num_cores_per_worker
    end = (worker_idx + 1) * num_cores_per_worker - 1
    curr_worker_cores = [i for i in range(start, end + 1)]
    affinity = "numactl -C {}-{}".format(str(start), str(end))
    affinity += " -m {}".format(
        ",".join(
            [str(numa_id) for numa_id in cpuinfo.numa_aware_check(curr_worker_cores)]
        )
    )
    return affinity


def scale_workers_with_core_pinning(scaled_num_workers):
    params = (("min_worker", str(scaled_num_workers)),)
    requests.put("http://localhost:8081/models/resnet-18", params=params)
    response = requests.get("http://localhost:8081/models/resnet-18")
    return response


@pytest.mark.skipif(
    not ipex_launcher_available,
    reason="Make sure intel-extension-for-pytorch is installed",
)
def test_single_worker_affinity():
    num_workers = 1
    worker_idx = 0
    setup_torchserve()
    requests.post(
        "http://localhost:8081/models?initial_workers={}&synchronous=true&url=resnet-18.mar".format(
            num_workers
        )
    )

    response = run_inference("resnet-18")
    assert (
        response.status_code == 200
    ), "single-worker inference with core pinning failed"

    affinity = get_worker_affinity(num_workers, worker_idx)
    assert affinity in open(TS_LOG).read(), "workers are not correctly pinned to cores"
    test_utils.unregister_model("resnet-18")
    

@pytest.mark.skipif(
    not ipex_launcher_available,
    reason="Make sure intel-extension-for-pytorch is installed",
)
def test_multi_worker_affinity():
    num_workers = 4
    setup_torchserve()
    requests.post(
        "http://localhost:8081/models?initial_workers={}&synchronous=true&url=resnet-18.mar".format(
            num_workers
        )
    )

    response = run_inference("resnet-18")
    assert (
        response.status_code == 200
    ), "multi-worker inference with core pinning failed"

    for worker_idx in range(num_workers):
        curr_worker_affinity = get_worker_affinity(num_workers, worker_idx)
        assert (
            curr_worker_affinity in open(TS_LOG).read()
        ), "workers are not correctly pinned to cores"
    test_utils.unregister_model("resnet-18")
    

@pytest.mark.skipif(
    not ipex_launcher_available,
    reason="Make sure intel-extension-for-pytorch is installed",
)
def test_worker_scale_up_affinity():
    initial_num_workers = 2
    setup_torchserve()
    requests.post(
        "http://localhost:8081/models?initial_workers={}&synchronous=true&url=resnet-18.mar".format(
            initial_num_workers
        )
    )

    scaled_up_num_workers = 4
    response = scale_workers_with_core_pinning(scaled_up_num_workers)
    resnet18_list = json.loads(response.content)
    assert (
        len(resnet18_list[0]["workers"]) == scaled_up_num_workers
    ), "workers failed to scale up with core pinning"

    response = run_inference("resnet-18")
    assert (
        response.status_code == 200
    ), "scaled up workers inference with core pinning failed"

    for worker_idx in range(scaled_up_num_workers):
        curr_worker_affinity = get_worker_affinity(scaled_up_num_workers, worker_idx)
        assert (
            curr_worker_affinity in open(TS_LOG).read()
        ), "workers are not correctly pinned to cores"
    test_utils.unregister_model("resnet-18")


@pytest.mark.skipif(
    not ipex_launcher_available,
    reason="Make sure intel-extension-for-pytorch is installed",
)
def test_worker_scale_down_affinity():
    initial_num_workers = 4
    setup_torchserve()
    requests.post(
        "http://localhost:8081/models?initial_workers={}&synchronous=true&url=resnet-18.mar".format(
            initial_num_workers
        )
    )

    scaled_down_num_workers = 2
    response = scale_workers_with_core_pinning(scaled_down_num_workers)
    resnet18_list = json.loads(response.content)
    assert (
        len(resnet18_list[0]["workers"]) == scaled_down_num_workers
    ), "workers failed to scale down with core pinning"

    response = run_inference("resnet-18")
    assert (
        response.status_code == 200
    ), "scaled down workers inference with core pinning failed"

    for worker_idx in range(scaled_down_num_workers):
        curr_worker_affinity = get_worker_affinity(scaled_down_num_workers, worker_idx)
        assert (
            curr_worker_affinity in open(TS_LOG).read()
        ), "workers are not correctly pinned to cores"
    test_utils.unregister_model("resnet-18")