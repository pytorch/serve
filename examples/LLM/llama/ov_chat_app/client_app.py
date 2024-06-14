import json

import requests
import streamlit as st

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

with st.sidebar:
    st.title("🦙💬 Llama 2 Chatbot")

    try:
        res = requests.get(url="http://localhost:8080/ping")
        res = requests.get(url="http://localhost:8081/models/llamacpp")
        status = json.loads(res.text)[0]["workers"][0]["status"]

        if status == "READY":
            st.success("Proceed to entering your prompt message!", icon="👉")
        else:
            st.warning("Model not loaded in TorchServe", icon="⚠️")

    except requests.ConnectionError:
        st.warning("TorchServe is not up. Try again", icon="⚠️")

    st.subheader("Model parameters")
    temperature = st.sidebar.slider(
        "temperature", min_value=0.01, max_value=5.0, value=0.8, step=0.01
    )
    top_p = st.sidebar.slider(
        "top_p", min_value=0.01, max_value=1.0, value=0.95, step=0.01
    )
    max_tokens = st.sidebar.slider(
        "max_tokens", min_value=128, max_value=512, value=100, step=8
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


# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = (
        "Question: What are the names of the planets in the solar system? Answer: "
    )
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    url = "http://127.0.0.1:8080/predictions/llamacpp"
    data = json.dumps(
        {
            "prompt": prompt_input,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
        }
    )

    res = requests.post(url=url, data=data, headers=headers)

    return res.text


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ""
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
