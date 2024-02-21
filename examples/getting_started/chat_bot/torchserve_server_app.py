import json
import os

import requests
import streamlit as st

MODEL_NAME_1 = os.environ["MODEL_NAME_1"]
MODEL_NAME_2 = os.environ["MODEL_NAME_2"]
MODEL1 = MODEL_NAME_1.split("---")[1]
MODEL2 = MODEL_NAME_2.split("---")[1]
# App title
st.set_page_config(page_title="TorchServe Server")


def start_server():
    os.system("torchserve --start --ts-config /home/model-server/config.properties")
    st.session_state.started = True
    st.session_state.stopped = False
    st.session_state.registered = {MODEL_NAME_1: False, MODEL_NAME_2: False}


def stop_server():
    os.system("torchserve --stop")
    st.session_state.stopped = True
    st.session_state.started = False
    st.session_state.registered = {MODEL_NAME_1: False, MODEL_NAME_2: False}


def _register_model(url, MODEL_NAME):
    res = requests.post(url)
    if res.status_code != 200:
        server_state_container.error("Error registering model", icon="🚫")
        st.session_state.started = True
        return
    st.session_state.registered[MODEL_NAME] = True
    # st.session_state.started = False
    st.session_state.stopped = False
    server_state_container.caption(res.text)


def register_model(MODEL_NAME):
    if not st.session_state.started:
        server_state_container.caption("TorchServe is not running. Start it")
        return
    url = f"http://localhost:8081/models?model_name={MODEL_NAME}&url={MODEL_NAME}&batch_size=1&max_batch_delay=3000&initial_workers=1&synchronous=true"
    _register_model(url, MODEL_NAME)


def get_status():
    for MODEL_NAME in [MODEL_NAME_1, MODEL_NAME_2]:
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
    for MODEL_NAME in [MODEL_NAME_1, MODEL_NAME_2]:
        if st.session_state.registered[MODEL_NAME]:
            num_workers = st.session_state[workers]
            url = (
                f"http://localhost:8081/models/{MODEL_NAME}?min_worker="
                f"{str(num_workers)}&synchronous=true"
            )
            res = requests.put(url)
            server_state_container.caption(res.text)


def set_batch_size(batch_size):
    for MODEL_NAME in [MODEL_NAME_1, MODEL_NAME_2]:
        if st.session_state.registered[MODEL_NAME]:
            url = f"http://localhost:8081/models/{MODEL_NAME}/1.0"
            res = requests.delete(url)
            server_state_container.caption(res.text)
            st.session_state.registered[MODEL_NAME] = False

            batch_size = st.session_state[batch_size]
            url = (
                f"http://localhost:8081/models?model_name={MODEL_NAME}&url={MODEL_NAME}"
                f"&batch_size={str(batch_size)}&initial_workers={str(workers)}"
                f"&synchronous=true&max_batch_delay={str(max_batch_delay)}"
            )
            _register_model(url, MODEL_NAME)


def set_max_batch_delay(max_batch_delay):
    for MODEL_NAME in [MODEL_NAME_1, MODEL_NAME_2]:
        if st.session_state.registered[MODEL_NAME]:
            url = f"http://localhost:8081/models/{MODEL_NAME}/1.0"
            res = requests.delete(url)
            server_state_container.caption(res.text)
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
    st.session_state.registered = {MODEL_NAME_1: False, MODEL_NAME_2: False}

with st.sidebar:
    st.title("TorchServe Server ")

    st.button("Start Server", on_click=start_server)
    st.button("Stop Server", on_click=stop_server)
    st.button(f"Register {MODEL1}", on_click=register_model, args=(MODEL_NAME_1,))
    st.button(f"Register {MODEL2}", on_click=register_model, args=(MODEL_NAME_2,))
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

    if st.session_state.registered[MODEL_NAME_1]:
        st.success(f"Registered model {MODEL_NAME_1}", icon="✅")

    if st.session_state.registered[MODEL_NAME_2]:
        st.success(f"Registered model {MODEL_NAME_2}", icon="✅")

st.title("TorchServe Status")
server_state_container = st.container()
server_state_container.subheader("Server status:")

if st.session_state.started:
    server_state_container.success("Started TorchServe", icon="✅")

if st.session_state.stopped:
    server_state_container.success("Stopped TorchServe", icon="✅")

if st.session_state.registered[MODEL_NAME_1]:
    server_state_container.success(f"Registered model {MODEL_NAME_1}", icon="✅")

if st.session_state.registered[MODEL_NAME_2]:
    server_state_container.success(f"Registered model {MODEL_NAME_2}", icon="✅")

model_state_container = st.container()
with model_state_container:
    st.subheader("Model Status")

with model_state_container:
    st.button("Model Status", on_click=get_status)

import json
from datetime import datetime, timedelta

import altair
import pandas as pd
import requests
from numpy import float64
from pandas import DataFrame, Series, Timestamp

MINUTES_BACK = 60
DEFAULT_TIME_BACK = timedelta(minutes=-MINUTES_BACK)
DEFAULT_QUERY = "GPUMemoryUsed"
STEP_DURATION = "30s"


@st.cache_data
def full_url(url: str, has_time_range: bool = True) -> str:
    if has_time_range:
        return f"{url}/api/v1/query_range"  # Range query
    return f"{url}/api/v1/query"  # Instant query


def get_metrics(
    the_payload: dict[str, any],
    url: str,
    start_range: datetime = None,
    end_range: datetime = None,
) -> (dict[any, any], int):
    new_query = {}
    new_query.update(the_payload)
    if start_range and end_range:
        new_query["start"] = start_range.timestamp()
        new_query["end"] = end_range.timestamp()
        new_query["step"] = STEP_DURATION
    print("url=%s, params=%s", url, new_query)
    response = requests.get(url=url, params=new_query)
    return response.json(), response.status_code


def transform(m_data: dict[any, any]) -> DataFrame:
    """
    Convert a Prometheus data structure into a Panda DataFrame
    :param m_data:
    :return: DataFrame
    """
    df = DataFrame(
        {
            mtr["metric"]["__name__"]: Series(
                data=[float64(vl[1]) for vl in mtr["values"]],
                index=[Timestamp(vl[0], unit="s") for vl in mtr["values"]],
                name="GPUMemoryUsed",
            )
            for mtr in m_data["data"]["result"]
        }
    )
    print(f"Columns: {df.columns}")
    print(f"Index: {df.index}")
    print(f"Index: {df}")
    print(df.head())
    return df


def display_metrics():
    code = 0
    metrics = {}
    PROM_URL = full_url("http://172.18.0.2:9090", has_time_range=True)
    query = DEFAULT_QUERY
    payload = {"query": query}
    # First query we boostrap with a reasonable time range
    END: datetime = datetime.now()
    START = END + DEFAULT_TIME_BACK
    if payload:
        metrics, code = get_metrics(
            url=PROM_URL,
            the_payload=payload,
            start_range=START,
            end_range=END,
        )
        data: DataFrame = DataFrame()
        if code == 200:
            now = datetime.now()
            print("metrics is ", metrics)
            data = transform(m_data=metrics)
            data = data.reset_index()
            # data = DataFrame(metrics)
            data["Time"] = pd.to_datetime(data["index"])
            print("data is ", data)
            chart = (
                altair.Chart(data)
                .mark_line()
                .encode(
                    x="Time",
                    y="GPUMemoryUsed",
                )
            )
            metrics_container.altair_chart(chart, use_container_width=True)


metrics_container = st.container()

with metrics_container:
    st.subheader("Metrics")

with metrics_container:
    st.button("Get Prometheus Metrics", on_click=display_metrics)
