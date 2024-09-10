import json
import random
import shutil
import sys
from pathlib import Path

import pytest
import requests
import test_utils
import torch
from model_archiver.model_archiver_config import ModelArchiverConfig

CURR_FILE_PATH = Path(__file__).parent
VLLM_PATH = CURR_FILE_PATH.parents[1] / "examples" / "large_models" / "vllm"
LORA_SRC_PATH = VLLM_PATH / "lora"
CONFIG_PROPERTIES_PATH = CURR_FILE_PATH.parents[1] / "test" / "config_ts.properties"

LLAMA_MODEL_PATH = "model/models--meta-llama--Meta-Llama-3.1-8B/snapshots/48d6d0fc4e02fb1269b36940650a1b7233035cbb"

ADAPTER_PATH = "adapters/model/models--llama-duo--llama3.1-8b-summarize-gpt4o-128k/snapshots/4ba83353f24fa38946625c8cc49bf21c80a22825"

TOKENIZER_CONFIG = LORA_SRC_PATH / LLAMA_MODEL_PATH / "tokenizer_config.json"

CHAT_TEMPLATE = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = "26 Jul 2024" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- "Tools: " + builtin_tools | reject(\'equalto\', \'code_interpreter\') | join(", ") + "\\n\\n"}}\n{%- endif %}\n{{- "Cutting Knowledge Date: December 2023\\n" }}\n{{- "Today Date: " + date_string + "\\n\\n" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- "<|python_tag|>" + tool_call.name + ".call(" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + \'="\' + arg_val + \'"\' }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- endif %}\n                {%- endfor %}\n            {{- ")" }}\n        {%- else  %}\n            {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n            {{- \'{"name": "\' + tool_call.name + \'", \' }}\n            {{- \'"parameters": \' }}\n            {{- tool_call.arguments | tojson }}\n            {{- "}" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we\'re in ipython mode #}\n            {{- "<|eom_id|>" }}\n        {%- else %}\n            {{- "<|eot_id|>" }}\n        {%- endif %}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n)'

YAML_CONFIG = f"""
# TorchServe frontend parameters
minWorkers: 1
maxWorkers: 1
batchSize: 1
maxBatchDelay: 100
responseTimeout: 1200
deviceType: "gpu"
asyncCommunication: true
parallelType: "custom"
parallelLevel: {torch.cuda.device_count()}

handler:
    model_path: "{(LORA_SRC_PATH / LLAMA_MODEL_PATH).as_posix()}"
    vllm_engine_config:
        enable_lora: true
        max_loras: 4
        max_lora_rank: 32
        max_cpu_loras: 4
        max_num_seqs: 16
        max_model_len: 250
        tensor_parallel_size: {torch.cuda.device_count()}
        served_model_name:
            - "Meta-Llama-31-8B"

    chat_template: "chat_template.txt"

    adapters:
        adapter_1: "{(LORA_SRC_PATH / ADAPTER_PATH).as_posix()}"
"""

PROMPTS = [
    {
        "prompt": "A robot may not injure a human being",
        "temperature": 0.0,
        "logprobs": 1,
        "max_tokens": 20,
        "model": "Meta-Llama-31-8B",
        "stream": True,
    },
    {
        "prompt": "Paris is,",
        "logprobs": 1,
        "max_tokens": 20,
        "temperature": 0.0,
        "top_p": 0.1,
        "model": "Meta-Llama-31-8B",
        "seed": 42,
        "stream": True,
    },
    {
        "prompt": "Paris is,",
        "logprobs": 1,
        "max_tokens": 20,
        "temperature": 0.0,
        "top_p": 0.1,
        "model": "adapter_1",
        "seed": 42,
        "stream": True,
    },
]
EXPECTED = [
    " or, through inaction, allow a human being to come to harm.\nA robot must obey the",
    " without a doubt, one of the most beautiful cities in the world. It is a city that is",
    " without a doubt, one of the most beautiful cities in the world. Its rich history, stunning architecture",
]

try:
    import vllm  # noqa

    VLLM_MISSING = False
except ImportError:
    VLLM_MISSING = True

try:
    from openai import OpenAI  # noqa

    OPENAI_MISSING = False
except ImportError:
    OPENAI_MISSING = True


def necessary_files_unavailable():
    LLAMA = LORA_SRC_PATH / LLAMA_MODEL_PATH
    ADAPTER = LORA_SRC_PATH / ADAPTER_PATH
    if not (LLAMA.exists() and ADAPTER.exists()):
        return {
            "condition": True,
            "reason": f"Required files are not present (see README): {LLAMA.as_posix()} + {ADAPTER.as_posix()}",
        }
    elif VLLM_MISSING:
        return {
            "condition": True,
            "reason": f"VLLM is not installed",
        }
    else:
        return {
            "condition": False,
            "reason": "None",
        }


@pytest.fixture
def add_paths():
    sys.path.append(VLLM_PATH.as_posix())
    sys.path.append(LORA_SRC_PATH.as_posix())
    yield
    sys.path.pop()
    sys.path.pop()


@pytest.fixture(scope="module")
def model_name():
    yield "Meta-Llama-31-8B"


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory, model_name):
    return tmp_path_factory.mktemp(model_name)


@pytest.fixture(scope="module", name="mar_file_path")
def create_mar_file(work_dir, model_archiver, model_name, request):
    mar_file_path = Path(work_dir).joinpath(model_name)

    model_config_yaml = Path(work_dir) / "model-config.yaml"
    model_config_yaml.write_text(YAML_CONFIG)

    chat_template_txt = Path(work_dir) / "chat_template.txt"
    chat_template_txt.write_text(CHAT_TEMPLATE)

    config = ModelArchiverConfig(
        model_name=model_name,
        version="1.0",
        handler="vllm_handler",
        serialized_file=None,
        export_path=work_dir,
        requirements_file=None,
        runtime="python",
        force=False,
        config_file=model_config_yaml.as_posix(),
        extra_files=chat_template_txt.as_posix(),
        archive_format="no-archive",
    )

    model_archiver.generate_model_archive(config)

    assert mar_file_path.exists()

    yield mar_file_path.as_posix()

    # Clean up files
    shutil.rmtree(mar_file_path)


@pytest.fixture(scope="module", name="model_name")
def register_model(mar_file_path, model_store, torchserve):
    """
    Register the model in torchserve
    """
    file_name = Path(mar_file_path).name
    model_name = Path(file_name).stem

    shutil.copytree(mar_file_path, model_store + f"/{model_name}")

    params = (
        ("model_name", model_name),
        ("url", file_name),
        ("initial_workers", "1"),
        ("synchronous", "true"),
        ("batch_size", "1"),
    )

    test_utils.reg_resp = test_utils.register_model_with_params(params)

    yield model_name

    test_utils.unregister_model(model_name)

    # Clean up files
    shutil.rmtree(Path(model_store) / model_name)


def extract_text(chunk):
    if not isinstance(chunk, str):
        chunk = chunk.decode("utf-8")
    if chunk.startswith("data:"):
        chunk = chunk[len("data:") :].split("\n")[0].strip()
        if chunk.startswith("[DONE]"):
            return ""
    return json.loads(chunk)["choices"][0]["text"]


def extract_chat(chunk):
    if not isinstance(chunk, str):
        chunk = chunk.decode("utf-8")
    if chunk.startswith("data:"):
        chunk = chunk[len("data:") :].split("\n")[0].strip()
        if chunk.startswith("[DONE]"):
            return ""
    try:
        return json.loads(chunk)["choices"][0].get("message", {})["content"]
    except KeyError:
        return json.loads(chunk)["choices"][0].get("delta", {}).get("content", "")


@pytest.mark.skipif(**necessary_files_unavailable())
def test_vllm_lora(model_name):
    """
    Register the model in torchserve
    """

    base_url = f"http://localhost:8080/predictions/{model_name}/1.0/v1/completions"

    responses = []

    for _ in range(10):
        idx = random.randint(0, len(PROMPTS) - 1)

        response = requests.post(base_url, json=PROMPTS[idx], stream=True)

        assert response.status_code == 200

        assert response.headers["Transfer-Encoding"] == "chunked"
        responses += [(response, EXPECTED[idx])]

    predictions = []
    expected_result = []
    for response, expected in responses:
        prediction = ""
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                prediction += extract_text(chunk)
        predictions += [prediction]
        expected_result += [expected]

    assert all(len(p) > 1 for p in predictions)
    assert all("".join(p) == e for p, e in zip(predictions, expected_result))


@pytest.mark.skipif(**necessary_files_unavailable())
@pytest.mark.parametrize("stream", [True, False])
def test_openai_api_completions(model_name, stream):
    base_url = f"http://localhost:8080/predictions/{model_name}/1.0/v1/completions"

    data = {
        "model": model_name,
        "prompt": "Hello world",
        "temperature": 0.0,
        "stream": stream,
    }

    response = requests.post(base_url, json=data, stream=stream)

    if stream:
        assert response.headers["Transfer-Encoding"] == "chunked"

        EXPECTED = (
            "! I’m a new blogger and I’m excited to share my thoughts and experiences"
        )
        i = 0

        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                text = extract_text(chunk)
                assert text == EXPECTED[i : i + len(text)]
                i += len(text)
        assert i > 0

    else:
        assert (
            extract_text(response.text)
            == "! I’m a new blogger and I’m excited to share my thoughts and experiences"
        )


@pytest.mark.skipif(**necessary_files_unavailable())
@pytest.mark.parametrize("stream", [True, False])
def test_openai_api_chat_complations(model_name, stream):
    base_url = f"http://localhost:8080/predictions/{model_name}/1.0/v1/chat/completions"

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {
                "role": "assistant",
                "content": "The Los Angeles Dodgers won the World Series in 2020.",
            },
            {"role": "user", "content": "Where was it played?"},
        ],
        "temperature": 0.0,
        "max_tokens": 50,
        "stream": stream,
    }

    response = requests.post(base_url, json=data, stream=stream)

    EXPECTED = " The World Series was played in Arlington, Texas. The Dodgers defeated the Tampa Bay Rays in six games."
    if stream:
        assert response.headers["Transfer-Encoding"] == "chunked"

        text = ""
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                text += extract_chat(chunk)
        assert text.startswith(EXPECTED)

    else:
        assert extract_chat(response.text).startswith(EXPECTED)


@pytest.mark.skipif(**necessary_files_unavailable())
def test_openai_api_models(model_name):
    base_url = f"http://localhost:8080/predictions/{model_name}/1.0/v1/models"

    response = requests.post(base_url)

    data = json.loads(response.text)

    models = [m["id"] for m in data["data"]]

    assert model_name in models

    assert "adapter_1" in models
