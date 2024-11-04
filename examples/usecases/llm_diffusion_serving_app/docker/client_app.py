import json
import os
import requests
import streamlit as st
import asyncio
import aiohttp
import time
import re
import logging

import numpy as np
from PIL import Image

MODEL_NAME_LLM = os.environ["MODEL_NAME_LLM"]
MODEL_NAME_LLM = MODEL_NAME_LLM.replace("/", "---")

MODEL_NAME_SD = os.environ["MODEL_NAME_SD"]
MODEL_NAME_SD = MODEL_NAME_SD.replace("/", "---")

logger = logging.getLogger(__name__)

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
    num_images = st.sidebar.number_input(
        "num_of_img", min_value=1, max_value=8, value=2, step=1
    )

    st.subheader("LLM Model parameters")
    max_new_tokens = st.sidebar.number_input(
        "max_new_tokens", min_value=30, max_value=250, value=40, step=5
    )

    temperature = st.sidebar.number_input(
        "temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.5
    )

    top_k = st.sidebar.number_input(
        "top_k", min_value=1, max_value=200, value=50, step=10
    )
    
    top_p = st.sidebar.number_input(
        "top_p",  min_value=0.0, max_value=1.0, value=0.95, step=0.5
    )


    st.subheader("Stable Diffusion Parameters")
    num_inference_steps = st.sidebar.number_input(
        "steps", min_value=1, max_value=100, value=5, step=1
    )

    guidance_scale = st.sidebar.number_input(
        "guidance_scale", min_value=1.0, max_value=30.0, value=5.0, step=0.5
    )

    height = st.sidebar.number_input(
        "height", min_value=256, max_value=2048, value=768, step=128
    )

    width = st.sidebar.number_input(
        "width", min_value=256, max_value=2048, value=768, step=128
    )


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
    try:
        async with session.post(url, data=data_input) as response:
            if response.status == 200:
                resp_text = await response.text()
                return resp_text
            else:
                return f"Error: Server returned status code {response.status}. Modify Stable Diffusion Parameters or Reload the model. \n"
    except Exception as e:
        return f"Error: Failed to process request - {str(e)}"

async def generate_sd_response_v2(prompts):
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        tasks = [send_inference_request(session, prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks)
        
        # Check if any result contains an error
        errors = [r for r in results if isinstance(r, str) and r.startswith('Error:')]
        if errors:
            st.error('\n'.join(errors))
            return None
        return results


def sd_response_postprocess(response):
    return [Image.fromarray(np.array(json.loads(text), dtype="uint8")) for text in response]


def preprocess_llm_input(user_prompt, num_images = 2):
    template =  """ Below is an instruction that describes a task. Write a response that appropriately completes the request.
    Generate {} unique prompts similar to '{}' by changing the context, keeping the core theme intact. 
    Give the output in square brackets seperated by semicolon.
    Do not generate text beyond the specified output format. Do not explain your response.
    ### Response: 
    """
    
    # Get 'num_images-1' prompts as the user_prompt is included.
    prompt_template_with_user_input = template.format(num_images-1, user_prompt)

    return prompt_template_with_user_input


def postprocess_llm_response(prompts, original_prompt=None, num_images=2):
    # Parse the JSON string into a Python list
    prompts = prompts.strip()
    prompts = json.loads(prompts)
    logging.info(f"LLM Model Responded with prompts: {prompts}")
    
    if len(prompts) < num_images:
         logging.info(f"Llama Model generated too few prompts! Required: {num_images}, Generated: {len(prompts)}")
    else:
        prompts = prompts[:num_images]

    return prompts


def generate_llm_model_response(prompt_template_with_user_input, user_prompt):
    headers = {"Content-type": "application/json", "Accept": "text/plain"}
    url = f"http://127.0.0.1:8080/predictions/{MODEL_NAME_LLM}"
    data = json.dumps(
        {
            "prompt_template": prompt_template_with_user_input,
            "user_prompt": user_prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
        }
    )

    try:
        res = requests.post(url=url, data=data, headers=headers, stream=True)
        logging.info(f"LLM Model Request Response Code: {res.status_code}")
        
        if res.status_code == 200:
            response_text = res.text
        else:
            response_text = "Sorry, there was an issue generating prompts. Please try again !"
            logging.error(f"LLM Model Request failed with status code: {res.status_code}")

    except requests.RequestException as req_exception:
        response_text = "Sorry, there was an error connecting to the server. Please try again !"
        logging.error(f"LLM Model Request failed with error: {req_exception}")

    return response_text
    

# Client Page UI
st.title("Multi-Image Generation App with TorchServe and OpenVINO")
intro_container = st.container()
with intro_container:
    st.markdown("""
        The multi-image generation app generates similar image generation prompts using **LLaMA-3.2** and 
        these prompts are then processed in parallel by **Stable Diffusion**, which is optimized 
        using the **latent-consistency/lcm-sdxl** model and accelerated with **Torch.compile** using the 
        **OpenVINO** backend. This approach enables efficient and high-quality image generation, 
        offering users a selection of interpretations to choose from.
    """)
    st.image("workflow-2.png")
    st.markdown("""
        **NOTE:** The initial image generations might take longer due to model initialization and warm-up. 
        Subsequent generations should be faster !
    """)
    
user_prompt = st.text_input("Enter an Image Generation Prompt :")

prompt_container = st.container()
status_container = st.container()

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
    prompt_container.write(f"Generated Prompts:")
    prompt_list = ""
    for i, pr in enumerate(st.session_state.llm_prompts, 1):
        prompt_list += f"{i}. {pr}\n"
    prompt_container.markdown(prompt_list)
        
if 'llm_prompts' not in st.session_state:
    st.session_state.llm_prompts = None

if 'llm_time' not in st.session_state:
    st.session_state.llm_time = 0

if st.session_state.model_sd_loaded:
    if prompt_container.button("Generate Prompts"):
        with st.spinner('Generating prompts...'):
            llm_start_time = time.time()

            st.session_state.llm_prompts = [user_prompt]
            if num_images > 1:
                prompt_template_with_user_input = preprocess_llm_input(user_prompt, num_images)
                llm_prompts = generate_llm_model_response(prompt_template_with_user_input, user_prompt)
                st.session_state.llm_prompts = postprocess_llm_response(llm_prompts, user_prompt, num_images)

            st.session_state.llm_time = time.time() - llm_start_time
            display_prompts()
            prompt_container.write(f"LLM time: {st.session_state.llm_time:.2f} sec.")
else:
    st.warning("Start TorchServe and Register models in the Server App running at port 8084.", icon="⚠️")


if not st.session_state.llm_prompts:
    prompt_container.write(f"Enter Image Generation Prompt and Click Generate Prompts !")
    pass
elif len(st.session_state.llm_prompts) != num_images:
    st.warning("Generate the prompts again !", icon="⚠️")
    pass
else:
    if st.button("Generate Images"):
        with st.spinner('Generating images...'):
            st.session_state.gen_captions[:0] = st.session_state.llm_prompts
            sd_start_time = time.time()
            
            sd_res = asyncio.run(generate_sd_response_v2(st.session_state.llm_prompts))

            if sd_res is not None: 
                images = sd_response_postprocess(sd_res)
                st.session_state.gen_images[:0] = images

                image_container = st.container()
                display_images_in_grid(st.session_state.gen_images, st.session_state.gen_captions)

                sd_time = time.time() - sd_start_time
                e2e_time = sd_time + st.session_state.llm_time
                
                timing_info = f"""
                Total Time Taken: {e2e_time:.2f} sec.
                LLM time: {st.session_state.llm_time:.2f} sec.
                SD time: {sd_time:.2f} sec."""
                
                print(timing_info)
                display_prompts()
                prompt_container.write(timing_info)