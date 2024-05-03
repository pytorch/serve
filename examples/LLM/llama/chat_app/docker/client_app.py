import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st

MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_NAME = MODEL_NAME.replace("/", "---")

# App title
st.set_page_config(page_title="TorchServe Chatbot")

with st.sidebar:
    st.title("TorchServe Chatbot")

    st.session_state.model_loaded = False
    try:
        res = requests.get(url="http://localhost:8080/ping")
        res = requests.get(url=f"http://localhost:8081/models/{MODEL_NAME}")
        status = "NOT READY"
        if res.status_code == 200:
            status = json.loads(res.text)[0]["workers"][0]["status"]

        if status == "READY":
            st.session_state.model_loaded = True
            st.success("Proceed to entering your prompt message!", icon="👉")
        else:
            st.warning("Model not loaded in TorchServe", icon="⚠️")

    except requests.ConnectionError:
        st.warning("TorchServe is not up. Try again", icon="⚠️")

    if st.session_state.model_loaded:
        st.success(f"Model loaded: {MODEL_NAME}!", icon="👉")

    st.subheader("Model parameters")
    temperature = st.sidebar.slider(
        "temperature", min_value=0.1, max_value=1.0, value=0.5, step=0.1
    )
    top_p = st.sidebar.slider(
        "top_p", min_value=0.1, max_value=1.0, value=0.5, step=0.1
    )
    max_new_tokens = st.sidebar.slider(
        "max_new_tokens", min_value=48, max_value=512, value=50, step=4
    )
    concurrent_requests = st.sidebar.select_slider(
        "concurrent_requests", options=[2**j for j in range(0, 8)]
    )

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)


def generate_model_response(prompt_input, executor):
    string_dialogue = (
        "Question: What are the names of the planets in the solar system? Answer: "
    )
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME}"
    data = json.dumps(
        {
            "prompt": prompt_input,
            "params": {
                "max_new_tokens": max_new_tokens,
                "top_p": top_p,
                "temperature": temperature,
            },
        }
    )
    res = [
        executor.submit(requests.post, url=url, data=data, headers=headers, stream=True)
        for i in range(concurrent_requests)
    ]

    return res, max_new_tokens


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            with ThreadPoolExecutor() as executor:
                futures, max_tokens = generate_model_response(prompt, executor)
                placeholder = st.empty()
                full_response = ""
                count = 0
                for future in futures:
                    response = future.result()
                    for chunk in response.iter_content(chunk_size=None):
                        if chunk:
                            data = chunk.decode("utf-8")
                            full_response += data
                            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
