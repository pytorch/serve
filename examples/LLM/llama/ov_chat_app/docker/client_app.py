import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st
import asyncio
import aiohttp
import time

import numpy as np
from PIL import Image

MODEL_NAME_LLM = os.environ["MODEL_NAME_LLM"]
MODEL_NAME_LLM = MODEL_NAME_LLM.replace("/", "---")

MODEL_NAME_SD = os.environ["MODEL_NAME_SD"]
MODEL_NAME_SD = MODEL_NAME_SD.replace("/", "---")

# App title
st.set_page_config(page_title="Image Generation with SDXL and OpenVino")

with st.sidebar:
    st.title("Image Generation with SDXL and OpenVino")

    st.session_state.model_sd_loaded = False

    try:
        res = requests.get(url="http://localhost:8080/ping")

        sd_res = requests.get(
            url=f"http://localhost:8081/models/{MODEL_NAME_SD}")
        sd_status = "NOT READY"

        llm_res = requests.get(
            url=f"http://localhost:8081/models/{MODEL_NAME_LLM}")
        llm_status = "NOT READY"

        if sd_res.status_code == 200:
            sd_status = json.loads(sd_res.text)[0]["workers"][0]["status"]

        if llm_res.status_code == 200:
            llm_status = json.loads(llm_res.text)[0]["workers"][0]["status"]

        st.session_state.model_sd_loaded = True if sd_status == "READY" else False
        if st.session_state.model_sd_loaded:
            st.success(f"Model loaded: {MODEL_NAME_SD}", icon="👉")
        else:
            st.warning(
                f"Model {MODEL_NAME_SD} not loaded in TorchServe", icon="⚠️")

        st.session_state.model_llm_loaded = True if llm_status == "READY" else False
        if st.session_state.model_llm_loaded:
            st.success(f"Model loaded: {MODEL_NAME_LLM}", icon="👉")
        else:
            st.warning(
                f"Model {MODEL_NAME_LLM} not loaded in TorchServe", icon="⚠️")

        if sd_status == "READY" and llm_status == "READY":
            st.success("Proceed to entering your prompt input!", icon="👉")

    except requests.ConnectionError:
        st.warning("TorchServe is not up. Try again", icon="⚠️")

    st.subheader("SD Model parameters")
    num_inference_steps = st.sidebar.slider(
        "steps", min_value=1, max_value=150, value=10, step=1
    )
    guidance_scale = st.sidebar.slider(
        "guidance_scale", min_value=1.0, max_value=30.0, value=5.0, step=0.5
    )
    height = st.sidebar.slider(
        "height", min_value=256, max_value=2048, value=512, step=8
    )
    width = st.sidebar.slider(
        "max_tokens", min_value=256, max_value=2048, value=512, step=8
    )

    # st.subheader("LLM Model parameters")
    # temperature = st.sidebar.slider(
    #     "temperature", min_value=0.1, max_value=1.0, value=0.5, step=0.1
    # )
    # top_p = st.sidebar.slider(
    #     "top_p", min_value=0.1, max_value=1.0, value=0.5, step=0.1
    # )
    # max_new_tokens = st.sidebar.slider(
    #     "max_new_tokens", min_value=48, max_value=512, value=50, step=4
    # )
    # concurrent_requests = st.sidebar.select_slider(
    #     "concurrent_requests", options=[2**j for j in range(0, 8)]
    # )


prompt = st.text_input("Text Prompt", "An astronaut riding a horse")

# # TODO: For Tests, delete when LLM added
# prompt = [prompt,
#           "A robot playing a violin",
#           "A dragon flying over the mountains",
#           "A mermaid painting sunset on the beach"
#           ]


def generate_sd_response_v1(prompt_input):
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_SD}"
    response = []
    for pr in prompt_input:
        data_input = json.dumps(
            {
                "prompt": pr,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "height": height,
                "width": width,
            }
        )
        response.append(requests.post(url=url, data=data_input).text)
    return response


async def send_inference_request(session, prompt_input):
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_SD}"

    data_input = json.dumps(
        {
            "prompt": prompt_input,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "height": height,
            "width": width,
        }
    )

    async with session.post(url, data=data_input) as response:
        assert response.status == 200
        resp_text = await response.text()
        return resp_text


async def generate_sd_response_v2(prompts):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        tasks = []
        for prompt in prompts:
            tasks.append(send_inference_request(session, prompt))

        return await asyncio.gather(*tasks)


def sd_response_postprocess(response):
    return [Image.fromarray(np.array(json.loads(text), dtype="uint8")) for text in response]


def preprocess_llm_input(input_prompt):
    return f"Generate 3 prompts similar to the \"{input_prompt}\", return as plain text without numeration, prompts should be divided by \";\" symbol, first prompt should be original one."

def generate_llm_model_response(prompt_input):
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_LLM}"
    data = json.dumps(
        {
            "prompt": prompt_input,
            # "params": {
            #     "max_new_tokens": max_new_tokens,
            #     "top_p": top_p,
            #     "temperature": temperature,
            # },
        }
    )
    res = requests.post(url=url, data=data,).text
    print('!!!!!!!!!!!!!!!!!!!!', res)
    return res


def llm_response_postprocess(generated_prompts):
    return [item.strip() for item in generated_prompts.split(";")]


# if st.button("Generate Images"):
#     with st.spinner('Generating images...'):
#         start_time = time.time()
#         res = generate_sd_response_v1(prompt)
#         inference_time = time.time() - start_time

#         images = sd_response_postprocess(res)

#         st.write(f"Inference time: {inference_time:.2f} seconds")
#         st.image(images, caption=["Generated Image"] * len(images), use_column_width=True


if st.button("Generate Images"):
    with st.spinner('Generating images...'):
        start_time = time.time()
        llm_res = generate_llm_model_response(preprocess_llm_input(prompt))
        llm_res = llm_response_postprocess(llm_res)
        sd_res = asyncio.run(generate_sd_response_v2(llm_res))
        inference_time = time.time() - start_time

        images = sd_response_postprocess(sd_res)

        st.write(f"Inference time: {inference_time:.2f} seconds")
        st.image(images, caption=prompt, use_column_width=True)
