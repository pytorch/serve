import json
import os
import subprocess
import time

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
        server_state_container.error("Error registering model", icon="🚫")
        st.session_state.started = True
        return
    print(f"registering {MODEL_NAME}")
    st.session_state.registered[MODEL_NAME] = True
    st.session_state.stopped = False
    server_state_container.caption(res.text)


def register_model(MODEL_NAME):
    if not st.session_state.started:
        server_state_container.caption("TorchServe is not running. Start it")
        return
    url = f"http://127.0.0.1:8081/models?model_name={MODEL_NAME}&url={MODEL_NAME}&batch_size=1&max_batch_delay=3000&initial_workers=1&synchronous=true&disable_token_authorization=true"
    _register_model(url, MODEL_NAME)

def register_models(models: list):
    for model in models: register_model(model)

def get_status():
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


def scale_sd_workers(workers):
    if st.session_state.registered[MODEL_NAME_SD]:
        num_workers = st.session_state[workers]
        url = (
            f"http://localhost:8081/models/{MODEL_NAME_SD}?min_worker="
            f"{str(num_workers)}&synchronous=true"
        )
        res = requests.put(url)
        server_state_container.caption(res.text)


# def update_is_xl(is_xl):
#     if st.session_state.registered[MODEL_NAME_SD]:
#         is_xl = st.session_state[is_xl]
        # url = (
        #     f"http://localhost:/models/{MODEL_NAME_SD}?="
        #     f"{str(is_xl)}&synchronous=true"
        # )
        # res = requests.put(url)
        # server_state_container.caption(res.text)


if "started" not in st.session_state:
    st.session_state.started = False
if "stopped" not in st.session_state:
    st.session_state.stopped = False
if "registered" not in st.session_state:
    st.session_state.registered = {
        MODEL_NAME_LLM: False,
        MODEL_NAME_SD: False,
    }

with st.sidebar:
    st.title("TorchServe Server ")

    st.button("Start Server", on_click=start_server)
    st.button("Stop Server", on_click=stop_server)
    st.button(f"Register models", on_click=register_models, args=([MODEL_NAME_LLM, MODEL_NAME_SD],))

    st.subheader("SD Model parameters")
    # is_xl = st.checkbox(
    #     "SDXL model", 
    #     value=False, 
    #     key="SDXL model",
    #     on_change=update_is_xl,
    #     args=("SDXL model",),
    # )

    workers_sd = st.sidebar.slider(
        "Num Workers SD",
        key="Num Workers SD",
        min_value=1,
        max_value=4,
        value=2,
        step=1,
        on_change=scale_sd_workers,
        args=("Num Workers SD",),
    )

    if st.session_state.started:
        st.success("Started TorchServe", icon="✅")

    if st.session_state.stopped:
        st.success("Stopped TorchServe", icon="✅")

    if st.session_state.registered[MODEL_NAME_LLM]:
        st.success(f"Registered model {MODEL_NAME_LLM}", icon="✅")
        
    if st.session_state.registered[MODEL_NAME_SD]:
        st.success(f"Registered model {MODEL_NAME_SD}", icon="✅")


st.title("TorchServe Status")
server_state_container = st.container()
server_state_container.subheader("Server status:")

if st.session_state.started:
    server_state_container.success("Started TorchServe", icon="✅")

if st.session_state.stopped:
    server_state_container.success("Stopped TorchServe", icon="✅")

if st.session_state.registered[MODEL_NAME_LLM]:
    server_state_container.success(f"Registered model {MODEL_NAME_LLM}", icon="✅")

if st.session_state.registered[MODEL_NAME_SD]:
    server_state_container.success(f"Registered model {MODEL_NAME_SD}", icon="✅")


model_state_container = st.container()
with model_state_container:
    st.subheader("Model Status")

with model_state_container:
    st.button("Model Status", on_click=get_status)
