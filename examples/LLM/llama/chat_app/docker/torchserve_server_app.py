import json
import os
import subprocess
import time

import requests
import streamlit as st

MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_NAME = MODEL_NAME.replace("/", "---")
MODEL = MODEL_NAME.split("---")[1]

# App title
st.set_page_config(page_title="TorchServe Server")


def start_server():
    subprocess.run(
        ["torchserve --start --ts-config /home/model-server/config.properties"],
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
        MODEL_NAME: False,
    }


def stop_server():
    os.system("torchserve --stop")
    st.session_state.stopped = True
    st.session_state.started = False
    st.session_state.registered = {
        MODEL_NAME: False,
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
    url = f"http://localhost:8081/models?model_name={MODEL_NAME}&url={MODEL_NAME}&batch_size=1&max_batch_delay=3000&initial_workers=1&synchronous=true"
    _register_model(url, MODEL_NAME)


def get_status():
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


def scale_workers(workers):
    if st.session_state.registered[MODEL_NAME]:
        num_workers = st.session_state[workers]
        # num_workers = workers
        url = (
            f"http://localhost:8081/models/{MODEL_NAME}?min_worker="
            f"{str(num_workers)}&synchronous=true"
        )
        res = requests.put(url)
        server_state_container.caption(res.text)


def set_batch_size(batch_size):
    if st.session_state.registered[MODEL_NAME]:
        url = f"http://localhost:8081/models/{MODEL_NAME}/1.0"
        res = requests.delete(url)
        server_state_container.caption(res.text)
        print(f"Unregistering {MODEL_NAME}")
        st.session_state.registered[MODEL_NAME] = False
        print(f"batch size is {batch_size}")

        batch_size = st.session_state[batch_size]
        url = (
            f"http://localhost:8081/models?model_name={MODEL_NAME}&url={MODEL_NAME}"
            f"&batch_size={str(batch_size)}&initial_workers={str(workers)}"
            f"&synchronous=true&max_batch_delay={str(max_batch_delay)}"
        )
        _register_model(url, MODEL_NAME)


def set_max_batch_delay(max_batch_delay):
    if st.session_state.registered[MODEL_NAME]:
        url = f"http://localhost:8081/models/{MODEL_NAME}/1.0"
        res = requests.delete(url)
        server_state_container.caption(res.text)
        print(f"Unregistering {MODEL_NAME}")
        st.session_state.registered[MODEL_NAME] = False

        max_batch_delay = st.session_state[max_batch_delay]
        url = (
            f"http://localhost:8081/models?model_name={MODEL_NAME}&url="
            f"{MODEL_NAME}&batch_size={str(batch_size)}&initial_workers="
            f"{str(workers)}&synchronous=true&max_batch_delay={str(max_batch_delay)}"
        )
        _register_model(url, MODEL_NAME)


if "started" not in st.session_state:
    st.session_state.started = False
if "stopped" not in st.session_state:
    st.session_state.stopped = False
if "registered" not in st.session_state:
    st.session_state.registered = {
        MODEL_NAME: False,
    }

with st.sidebar:
    st.title("TorchServe Server ")

    st.button("Start Server", on_click=start_server)
    st.button("Stop Server", on_click=stop_server)
    st.button(f"Register {MODEL}", on_click=register_model, args=(MODEL_NAME,))

    workers = st.sidebar.slider(
        "Num Workers",
        key="Num Workers",
        min_value=1,
        max_value=4,
        value=1,
        step=1,
        on_change=scale_workers,
        args=("Num Workers",),
    )
    batch_size = st.sidebar.select_slider(
        "Batch Size",
        key="Batch Size",
        options=[2**j for j in range(0, 8)],
        on_change=set_batch_size,
        args=("Batch Size",),
    )
    max_batch_delay = st.sidebar.slider(
        "Max Batch Delay",
        key="Max Batch Delay",
        min_value=3000,
        max_value=10000,
        value=3000,
        step=100,
        on_change=set_max_batch_delay,
        args=("Max Batch Delay",),
    )

    if st.session_state.started:
        st.success("Started TorchServe", icon="✅")

    if st.session_state.stopped:
        st.success("Stopped TorchServe", icon="✅")

    if st.session_state.registered[MODEL_NAME]:
        st.success(f"Registered model {MODEL_NAME}", icon="✅")


st.title("TorchServe Status")
server_state_container = st.container()
server_state_container.subheader("Server status:")

if st.session_state.started:
    server_state_container.success("Started TorchServe", icon="✅")

if st.session_state.stopped:
    server_state_container.success("Stopped TorchServe", icon="✅")

if st.session_state.registered[MODEL_NAME]:
    server_state_container.success(f"Registered model {MODEL_NAME}", icon="✅")


model_state_container = st.container()
with model_state_container:
    st.subheader("Model Status")

with model_state_container:
    st.button("Model Status", on_click=get_status)
