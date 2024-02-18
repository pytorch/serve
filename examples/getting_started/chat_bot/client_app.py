import json
import os
import time

import requests
import streamlit as st

MODEL_NAME = os.environ["MODEL_NAME_1"]
# MODEL_NAME = "mistral-7b"

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

    st.subheader("Model Selection")
    MODEL_NAME = st.select_slider(
        "Select a model",
        options=[os.environ["MODEL_NAME_1"], os.environ["MODEL_NAME_2"]],
    )
    if st.session_state.model_loaded:
        st.success(f"Model selected: {MODEL_NAME}!", icon="👉")

    st.subheader("Model parameters")
    temperature = st.sidebar.slider(
        "temperature", min_value=0.01, max_value=5.0, value=0.8, step=0.01
    )
    top_p = st.sidebar.slider(
        "top_p", min_value=0.01, max_value=1.0, value=0.95, step=0.01
    )
    max_new_tokens = st.sidebar.slider(
        "max_new_tokens", min_value=48, max_value=512, value=50, step=4
    )

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]
if "first_token" not in st.session_state.keys():
    st.session_state.first_token = [0]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How may I assist you today?"}
    ]


st.sidebar.button("Clear Chat History", on_click=clear_chat_history)
st.sidebar.subheader("Metrics")


# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
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
            "req_time": time.time(),
        }
    )
    req_time = time.time()
    res = requests.post(url=url, data=data, headers=headers, stream=True)

    # res = requests.post(url=url, data=data, headers=headers)
    # return res.text
    return res, req_time, max_new_tokens


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, req_time, max_tokens = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ""
            first_token = False
            count = 0
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    if not first_token:
                        first_token_time = time.time()
                        first_token = True
                        # print("First token")
                        st.sidebar.write(
                            f"Time to first token : {first_token_time - req_time:.2f} seconds"
                        )
                    data = chunk.decode("utf-8")
                    # print("data is ", data)
                    full_response += data
                    placeholder.markdown(full_response)
            last_token_time = time.time()
            st.session_state.first_token.append(first_token_time - req_time)
            # print(st.session_state.first_token)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
    st.sidebar.write(f"Tokens/sec : {1.0*max_tokens/(last_token_time - req_time):.2f}")
