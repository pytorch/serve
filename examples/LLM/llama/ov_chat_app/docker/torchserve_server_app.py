import json
import os
import subprocess
import time
import sys
import importlib.metadata

import requests
import streamlit as st

MODEL_NAME_LLM = os.environ["MODEL_NAME_LLM"]
MODEL_NAME_LLM = MODEL_NAME_LLM.replace("/", "---")
MODEL_LLM = MODEL_NAME_LLM.split("---")[1]

MODEL_NAME_SD = os.environ["MODEL_NAME_SD"]
MODEL_NAME_SD = MODEL_NAME_SD.replace("/", "---")
MODEL_SD = MODEL_NAME_SD.split("---")[1]

# App title
st.set_page_config(page_title="TorchServe Server")


def start_server():
    subprocess.run(
        ["torchserve --start --ts-config /home/model-server/config.properties --disable-token-auth --enable-model-api"],
        shell=True,
        check=True,
    )
    while True:
        try:
            res = requests.get(url="http://localhost:8080/ping")
            if res.status_code == 200:
                break
            else:
                server_state_container.error("Not able to start TorchServe", icon="🚫")
        except:
            time.sleep(0.1)

    st.session_state.started = True
    st.session_state.stopped = False
    st.session_state.registered = {
        MODEL_NAME_LLM: False,
        MODEL_NAME_SD: False,
    }


def stop_server():
    os.system("torchserve --stop")
    st.session_state.stopped = True
    st.session_state.started = False
    st.session_state.registered = {
        MODEL_NAME_LLM: False,
        MODEL_NAME_SD: False,
    }


def _register_model(url, MODEL_NAME):
    res = requests.post(url)
    if res.status_code != 200:
        server_state_container.error(f"Error registering model: {MODEL_NAME}", icon="🚫")
        st.session_state.started = True
        return False

    print(f"registering {MODEL_NAME}")
    st.session_state.registered[MODEL_NAME] = True
    st.session_state.stopped = False
    server_state_container.caption(res.text)

    return True


def register_model(MODEL_NAME):
    if not st.session_state.started:
        server_state_container.caption("TorchServe is not running. Start it")
        return
    url = f"http://127.0.0.1:8081/models?model_name={MODEL_NAME}&url={MODEL_NAME}&batch_size=1&max_batch_delay=3000&initial_workers=1&synchronous=true&disable_token_authorization=true"
    return _register_model(url, MODEL_NAME)

def register_models(models: list):
    for model in models: 
        if not register_model(model):
            return

def get_model_status():
    for MODEL_NAME in [MODEL_NAME_LLM, MODEL_NAME_SD]:
        print(
            f"registered state for {MODEL_NAME} is {st.session_state.registered[MODEL_NAME]}"
        )
        if st.session_state.registered[MODEL_NAME]:
            url = f"http://localhost:8081/models/{MODEL_NAME}"
            res = requests.get(url)
            if res.status_code != 200:
                model_state_container.error(
                    f"Error getting model status for {MODEL_NAME}", icon="🚫"
                )
                return
            print(res.text)
            status = json.loads(res.text)[0]
            model_state_container.write(status)
        else:
            model_state_container.write(f"{MODEL_NAME} is not registered ! ")


def scale_sd_workers(workers):
    if st.session_state.registered[MODEL_NAME_SD]:
        num_workers = st.session_state[workers]
        url = (
            f"http://localhost:8081/models/{MODEL_NAME_SD}?min_worker="
            f"{str(num_workers)}&synchronous=true"
        )
        res = requests.put(url)
        server_state_container.caption(res.text)

def get_hw_config_output():
    output = subprocess.check_output(['lscpu']).decode('utf-8')
    lines = output.split('\n')
    cpu_model = None
    cpu_count = None
    threads_per_core = None
    cores_per_socket = None
    socket_count = None

    for line in lines:
        line = line.strip()
        if line.startswith('Model name:'):
            cpu_model = line.split('Model name:')[1].strip()
        elif line.startswith('CPU(s):'):
            cpu_count = line.split('CPU(s):')[1].strip()
        elif line.startswith('Thread(s) per core:'):
            threads_per_core = line.split('Thread(s) per core:')[1].strip()
        elif line.startswith('Core(s) per socket:'):
            cores_per_socket = line.split('Core(s) per socket:')[1].strip()
        elif line.startswith('Socket(s):'):
            socket_count = line.split('Socket(s):')[1].strip()

    return {
        'cpu_model': cpu_model,
        'cpu_count': cpu_count,
        'threads_per_core': threads_per_core,
        'cores_per_socket': cores_per_socket,
        'socket_count': socket_count
    }

def get_sw_versions():
    sw_versions = {}
    sw_versions['Python'] = sys.version
    sw_versions['TorchServe'] = importlib.metadata.version('torchserve')
    sw_versions['OpenVINO'] = importlib.metadata.version('openvino')
    sw_versions['PyTorch'] = importlib.metadata.version('torch')
    sw_versions['Transformers'] = importlib.metadata.version('transformers')
    sw_versions['Diffusers'] = importlib.metadata.version('diffusers')
    
    return sw_versions

if "started" not in st.session_state:
    st.session_state.started = False
if "stopped" not in st.session_state:
    st.session_state.stopped = False
if "registered" not in st.session_state:
    st.session_state.registered = {
        MODEL_NAME_LLM: False,
        MODEL_NAME_SD: False,
    }

# Server Page Sidebar UI
with st.sidebar:
    st.title("TorchServe Controls ")

    st.button("Start TorchServe", on_click=start_server)
    st.button("Stop Server", on_click=stop_server)
    st.button(f"Register Models", on_click=register_models, args=([MODEL_NAME_LLM, MODEL_NAME_SD],))

    st.subheader("SD Model parameters")

    workers_sd = st.sidebar.number_input(
        "Num Workers SD",
        key="Num Workers SD",
        min_value=1,
        max_value=4,
        value=4,
        on_change=scale_sd_workers,
        args=("Num Workers SD",),
    )
    
    if st.session_state.started:
        st.success("Started TorchServe", icon="✅")

    if st.session_state.stopped:
        st.success("Stopped TorchServe", icon="🛑")

    if st.session_state.registered[MODEL_NAME_LLM]:
        st.success(f"Registered model {MODEL_NAME_LLM}", icon="✅")
        
    if st.session_state.registered[MODEL_NAME_SD]:
        st.success(f"Registered model {MODEL_NAME_SD}", icon="✅")



# Server Page UI

st.title("Multi-Image Generation App Control Center")
image_container = st.container()
with image_container:
    st.markdown("""
    This Streamlit app is designed to generate multiple images based on a provided text prompt.
    It leverages **TorchServe** for efficient model serving and management, and utilizes **LLaMA3** 
    with **GPT-FAST** and **4-bit weight compression** for prompt generation, and **Stable Diffusion** 
    with **latent-consistency/lcm-sdxl** and **Torch.compile** using **OpenVINO backend** for image generation.
    """)
    st.image("workflow-1.png")
    
server_state_container = st.container()
server_state_container.subheader("TorchServe Status:")

if st.session_state.started:
    server_state_container.success("Started TorchServe", icon="✅")
elif st.session_state.stopped:
    server_state_container.success("Stopped TorchServe", icon="🛑")
else:
    server_state_container.write("TorchServe not started !")

if st.session_state.registered[MODEL_NAME_LLM]:
    server_state_container.success(f"Registered model {MODEL_NAME_LLM}", icon="✅")

if st.session_state.registered[MODEL_NAME_SD]:
    server_state_container.success(f"Registered model {MODEL_NAME_SD}", icon="✅")


model_state_container = st.container()
with model_state_container:
    st.subheader("Model Status:")

with model_state_container:
    st.button("Click here for Model Status", on_click=get_model_status)

# Hardware and Software Info
hw_info_container = st.container()
with hw_info_container:
    hw_config_output = get_hw_config_output()
    st.subheader("Hardware Config:")
    for key, value in hw_config_output.items():
        st.write(f"{key}: {value}")
 
sw_info_container = st.container()
with sw_info_container:
    sw_versions = get_sw_versions()
    st.subheader("Software Versions:")   
    for name, version in sw_versions.items():
        st.write(f"{name}: {version}")