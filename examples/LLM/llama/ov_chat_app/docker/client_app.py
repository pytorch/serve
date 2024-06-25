import json
import os
from concurrent.futures import ThreadPoolExecutor

import requests
import streamlit as st
import asyncio
import aiohttp
import time
import re

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
        "steps", min_value=1, max_value=150, value=30, step=1
    )
    guidance_scale = st.sidebar.slider(
        "guidance_scale", min_value=1.0, max_value=30.0, value=5.0, step=0.5
    )
    height = st.sidebar.slider(
        "height", min_value=256, max_value=2048, value=768, step=8
    )
    width = st.sidebar.slider(
        "width", min_value=256, max_value=2048, value=768, step=8
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


prompt = st.text_input("Text Prompt")

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
    return f"Generate 4 prompts in English similar to \"{input_prompt}\". Prompts should be divided by \";\" symbol and included  in \"[\"\"]\" brackets. \
             Return only prompts as plain text without numeration, any extra symbols or notes."


def get_prompts_string(input_string):
    match = re.search(r'\[(.*?)\]', input_string)
    if match:
        return match.group(1)
    else:
        return None

def llm_response_postprocess(generated_prompts):
    prompts = get_prompts_string(generated_prompts)
    prompts = [item.strip() for item in prompts.split(";")]
    assert len(prompts) == 4

    return prompts

def generate_llm_model_response(input_prompt):
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_LLM}"

    prompt_input = preprocess_llm_input(input_prompt)
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
    res = requests.post(url=url, data=data, headers=headers, stream=True)
    assert res.status_code == 200

    return llm_response_postprocess(res.text)


# def llm_response_postprocess(generated_prompts):
#     prompts = [item.strip() for item in generated_prompts.split(";")]
#     assert len(prompt) == 4

#     return prompts


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
        llm_res = generate_llm_model_response(prompt)
        sd_res = asyncio.run(generate_sd_response_v2(llm_res))
        inference_time = time.time() - start_time

        images = sd_response_postprocess(sd_res)
        inference_time = time.time() - start_time
        st.write(f"Inference time: {inference_time:.2f} seconds")
        st.image(images, caption=llm_res, use_column_width=True)
