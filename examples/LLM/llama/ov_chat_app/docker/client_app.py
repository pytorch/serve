import json
import os
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
st.set_page_config(page_title="Image Generation with SDXL and OpenVINO")

with st.sidebar:
    st.title("Image Generation with SDXL and OpenVINO")

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

    st.subheader("Number of images to generate")
    images_num = st.sidebar.number_input(
        "num_of_img", min_value=1, max_value=6, value=2, step=1
    )

    st.subheader("Llama Model parameters")
    max_new_tokens = st.sidebar.number_input(
        "max_new_tokens", min_value=30, max_value=250, value=50, step=10
    )

    temperature = st.sidebar.number_input(
        "temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.1
    )

    top_k = st.sidebar.number_input(
        "top_k", min_value=1, max_value=200, value=200, step=1
    )


    st.subheader("SD Model parameters")
    num_inference_steps = st.sidebar.number_input(
        "steps", min_value=1, max_value=100, value=5, step=1
    )

    guidance_scale = st.sidebar.number_input(
        "guidance_scale", min_value=1.0, max_value=30.0, value=5.0, step=0.5
    )

    height = st.sidebar.number_input(
        "height", min_value=256, max_value=2048, value=768, step=8
    )

    width = st.sidebar.number_input(
        "width", min_value=256, max_value=2048, value=768, step=8
    )


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


async def send_inference_request(session, input_prompt):
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_SD}"
    data_input = json.dumps(
        {
            "prompt": input_prompt,
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
        tasks = [send_inference_request(session, prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)

        return results


def sd_response_postprocess(response):
    return [Image.fromarray(np.array(json.loads(text), dtype="uint8")) for text in response]


def preprocess_llm_input(input_prompt, images_num = 2):
    return f"Generate {images_num} different unique prompts similar to: {input_prompt}. \
             Add semicolon between the prompts. \
             Generated string of prompts should be included in square brackets. E.g.:"

def trim_prompt(s):
    return re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', s)


def postprocess_llm_response(prompts, original_prompt=None, images_num=2):
    
    prompts = [trim_prompt(item) for item in prompts.split(";")]
    prompts = list(filter(None, prompts))
    
    if original_prompt:
        if len(prompts) == images_num - 1:
            prompts.insert(0, original_prompt)
        else:
            prompts[0] = original_prompt

    prompts = prompts[:images_num]
    assert len(prompts) == images_num, "Llama Model generated too few prompts!"

    return prompts


def generate_llm_model_response(input_prompt):
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_LLM}"
    data = json.dumps(
        {
            "prompt": input_prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        }
    )

    try:
        res = requests.post(url=url, data=data, headers=headers, stream=True)
        
        if res.status_code == 200:
            response_text = res.text
        else:
            response_text = "Sorry, there was an issue generating prompts. Please try again !"

    except requests.RequestException:
        response_text = "Sorry, there was an error connecting to the server. Please try again !"

    return response_text
    

# Client Page UI
st.title("Multi-Image Generation App with TorchServe and OpenVINO")
intro_container = st.container()
with intro_container:
    st.markdown("""
        The multi-image generation use case enhances user prompts using **LLaMA3**, which is optimized with 
        GPT-FAST and 4-bit weight compression, to generate four similar prompts with additional context.
        These refined prompts are then processed in parallel by **Stable Diffusion**, which is optimized 
        using the latent-consistency/lcm-sdxl model and accelerated with **Torch.compile** using the 
        **OpenVINO** backend. This approach enables efficient and high-quality image generation, 
        offering users a selection of interpretations to choose from.
    """)
    st.image("workflow-2.png")
    
prompt = st.text_input("Enter Text Prompt:")

prompt_container = st.container()
image_container = st.container()

if 'gen_images' not in st.session_state:
    st.session_state.gen_images = []
if 'gen_captions' not in st.session_state:
    st.session_state.gen_captions = []

def display_images_in_grid(images, captions):
    cols = image_container.columns(2)
    for i, (img, caption) in enumerate(zip(images, captions)):
        col = cols[i % 2]
        col.image(img, caption=caption, use_column_width=True)

def display_prompts():
    prompt_container.write(f"Generated prompts:")
    for pr in st.session_state.llm_prompts:
        prompt_container.write(pr)
        
if 'llm_prompts' not in st.session_state:
    st.session_state.llm_prompts = None

if 'llm_time' not in st.session_state:
    st.session_state.llm_time = 0

if prompt_container.button("Generate Prompts"):
    with st.spinner('Generating prompts...'):
        llm_start_time = time.time()

        st.session_state.llm_prompts = [prompt]
        if images_num > 1:
            prompt_input = preprocess_llm_input(prompt, images_num)
            llm_prompts = generate_llm_model_response(prompt_input)
            st.session_state.llm_prompts = postprocess_llm_response(llm_prompts, prompt, images_num)

        st.session_state.llm_time = time.time() - llm_start_time
        display_prompts()


if not st.session_state.llm_prompts:
    prompt_container.write(f"You need to generate prompts at first!")
    pass
elif len(st.session_state.llm_prompts) != images_num:
    prompt_container.write(f"Generate the prompts again!")
    pass
else:
    with st.spinner('Generating images...'):
        if image_container.button("Generate Images"):
            st.session_state.gen_captions[:0] = st.session_state.llm_prompts
            sd_start_time = time.time()
            
            sd_res = asyncio.run(generate_sd_response_v2(st.session_state.llm_prompts))

            images = sd_response_postprocess(sd_res)
            st.session_state.gen_images[:0] = images

            display_images_in_grid(st.session_state.gen_images, st.session_state.gen_captions)

            sd_time = time.time() - sd_start_time
            
            e2e_time = sd_time + st.session_state.llm_time
            
            timing_info = f"""Time Taken: {e2e_time:.2f} sec.
            LLM time: {st.session_state.llm_time:.2f} sec.
            SD time: {sd_time:.2f} sec."""
            
            print(timing_info)
            display_prompts()
            prompt_container.write(timing_info)
