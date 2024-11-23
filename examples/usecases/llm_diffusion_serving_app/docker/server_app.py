import json
import os
import subprocess
import time
import sys
import importlib.metadata

import requests
import streamlit as st
import logging

# App title
st.set_page_config(page_title="Multi-Image Gen App Control Center")

logger = logging.getLogger(__name__)

MODEL_NAME_LLM = os.environ["MODEL_NAME_LLM"]
MODEL_NAME_LLM = MODEL_NAME_LLM.replace("/", "---")
MODEL_LLM = MODEL_NAME_LLM.split("---")[1]

MODEL_NAME_SD = os.environ["MODEL_NAME_SD"]
MODEL_NAME_SD = MODEL_NAME_SD.replace("/", "---")
MODEL_SD = MODEL_NAME_SD.split("---")[1]

# Init Session State variables
st.session_state.started = st.session_state.get("started", False)
st.session_state.stopped = st.session_state.get("stopped", True)
st.session_state.registered = st.session_state.get(
    "registered",
    {
        MODEL_NAME_LLM: False,
        MODEL_NAME_SD: False,
    },
)


def is_server_running():
    """Check if the TorchServe server is running."""
    try:
        res = requests.get("http://localhost:8080/ping")
        return res.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def init_model_registrations():
    for model_name in [MODEL_NAME_LLM, MODEL_NAME_SD]:
        url = f"http://localhost:8081/models/{model_name}"
        try:
            res = requests.get(url)
            if res.status_code == 200:
                st.session_state.registered[model_name] = True
            else:
                st.session_state.registered[model_name] = False
        except requests.exceptions.RequestException as e:
            logger.info(f"Error checking model registration: {e}")
            st.session_state.registered[model_name] = False


# Update Session State variables
if is_server_running():
    st.session_state.started = True
    st.session_state.stopped = False
    init_model_registrations()


def start_torchserve_server():
    """Starts the TorchServe server if it's not already running."""

    def launch_server():
        """Launch the TorchServe server with the specified configurations."""
        subprocess.run(
            [
                "torchserve --start --ts-config /home/model-server/config.properties "
                "--disable-token-auth --enable-model-api"
            ],
            shell=True,
            check=True,
        )

    # Check if the server is already running
    if is_server_running():
        sidebar_status_container.info("TorchServe is already running.")
    else:
        # Start the server
        launch_server()

        # Attempt to connect up to 5 times with a short delay in between
        num_attempts = 5
        for attempt in range(num_attempts):
            if is_server_running():
                break
            else:
                time.sleep(0.5)
                logger.info(
                    f"Unable to start TorchServe: Attempt {attempt + 1} of {num_attempts}"
                )
        else:
            sidebar_status_container.error(
                f"Failed to start TorchServe after {num_attempts} attempts", icon="🚫"
            )
            return False

    # Update session state variables if server started successfully
    st.session_state.started = True
    st.session_state.stopped = False
    init_model_registrations()


def stop_server():
    """Stops the TorchServe server if it is running, and updates the session state."""
    if st.session_state.stopped:
        sidebar_status_container.info("TorchServe is already stopped.")
    else:
        try:
            # Stop the server if it is running
            subprocess.run(["torchserve", "--stop"], check=True)

            # Update session state upon successful server stop
            st.session_state.stopped = True
            st.session_state.started = False
            st.session_state.registered = {
                MODEL_NAME_LLM: False,
                MODEL_NAME_SD: False,
            }

        except subprocess.CalledProcessError as e:
            sidebar_status_container.error(f"Failed to stop TorchServe: {e}", icon="🚫")


def unregister_model(MODEL_NAME):
    url = f"http://127.0.0.1:8081/models/{MODEL_NAME}"
    response = requests.delete(url)
    if response.status_code == 200:
        logger.info(f"Model {MODEL_NAME} successfully unregistered")
        st.session_state.registered[MODEL_NAME] = False
        return True
    else:
        server_state_container.error(
            f"Failed to unregister model {MODEL_NAME}."
            f"Status code: {response.status_code}"
            f"Response: {response.text}",
            icon="🚫",
        )

        return False


def _register_model(url, MODEL_NAME):
    server_state_container.write(f"Registering {MODEL_NAME}")
    res = requests.post(url)

    if res.status_code != 200:
        server_state_container.error(
            f"Error {res.status_code}: Failed to register model: {MODEL_NAME}",
            icon="🚫",
        )
        return False

    st.session_state.registered[MODEL_NAME] = True
    server_state_container.write(res.text)

    return True


def register_model(MODEL_NAME):
    if not st.session_state.started:
        sidebar_status_container.warning(
            "TorchServe Server is not running. Start it !", icon="⚠️"
        )
        return False

    url = (
        f"http://127.0.0.1:8081/models"
        f"?model_name={MODEL_NAME}"
        f"&url={MODEL_NAME}"
        f"&batch_size=1"
        f"&max_batch_delay=3000"
        f"&initial_workers=1"
        f"&synchronous=true"
        f"&disable_token_authorization=true"
    )

    return _register_model(url, MODEL_NAME)


def register_models(models: list):
    for model in models:
        success = register_model(model)
        # If registration fails, exit the function early
        if not success:
            logger.error(f"Failed to register model: {model}")
            return False
    # Call scale_sd_workers after model registration, which overrides min_workers in model-config.yaml
    scale_sd_workers()
    logger.info("Both models registered successfully.")


def get_model_status():
    for MODEL_NAME in [MODEL_NAME_LLM, MODEL_NAME_SD]:
        print(
            f"Registered state for {MODEL_NAME} is {st.session_state.registered[MODEL_NAME]}"
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


def scale_sd_workers(workers_key="sd_workers"):
    if st.session_state.registered[MODEL_NAME_SD]:
        num_workers = st.session_state.get(workers_key, 2)
        url = (
            f"http://localhost:8081/models/{MODEL_NAME_SD}?"
            f"min_worker={str(num_workers)}&synchronous=true"
        )
        res = requests.put(url)
        server_state_container.write(res.text)


def get_hw_config_output():
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


# Server Page Sidebar UI
with st.sidebar:
    st.title("TorchServe Controls ")

    st.button("Start TorchServe Server", on_click=start_torchserve_server)
    st.button("Stop TorchServe Server", on_click=stop_server)
    st.button(
        "Register Models",
        on_click=register_models,
        args=([MODEL_NAME_LLM, MODEL_NAME_SD],),
    )

    sidebar_status_container = st.container()

    if st.session_state.started:
        sidebar_status_container.success("Started TorchServe", icon="✅")

    if st.session_state.stopped:
        sidebar_status_container.success("TorchServe is Stopped", icon="🛑")

    st.subheader("Stable Diffusion Model Config Parameters")

    sd_workers_inp = st.sidebar.number_input(
        "Num Workers for Stable Diffusion",
        key="sd_workers",
        min_value=1,
        max_value=8,
        value=2,
        on_change=scale_sd_workers,
        args=("sd_workers",),
    )

    if st.session_state.registered[MODEL_NAME_LLM]:
        sidebar_status_container.success(
            f"Registered model {MODEL_NAME_LLM}", icon="✅"
        )

    if st.session_state.registered[MODEL_NAME_SD]:
        sidebar_status_container.success(f"Registered model {MODEL_NAME_SD}", icon="✅")


# Server Page UI

intro_container = st.container()
with intro_container:
    st.markdown(
        """
    ### Multi-Image Generation App Control Center
    Manage the Multi-Image Generation App workflow with this administrative interface.
    Use this app to Start/stop TorchServe, load/register models, scale up/down workers,
    and review TorchServe Server and Model info.
    See [GitHub](https://github.com/pytorch/serve/tree/master/examples/usecases/llm_diffusion_serving_app) for details.
    """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div style='background-color: #232628; font-size: 14px; padding: 10px;
                border: 1px solid #ddd; border-radius: 5px;'>
        <b>NOTE</b>: After Starting TorchServe and Registering models, proceed to Client App running at port 8085.
        </div>""",
        unsafe_allow_html=True,
    )

    st.image("./img/workflow-1.png")

server_state_container = st.container()
server_state_container.subheader("TorchServe Status:")

if st.session_state.started:
    server_state_container.success("Started TorchServe", icon="✅")
elif st.session_state.stopped:
    server_state_container.success("TorchServe is Stopped", icon="🛑")
else:
    server_state_container.warning("TorchServe not started !", icon="⚠️")

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
