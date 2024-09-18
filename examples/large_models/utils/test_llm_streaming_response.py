import argparse
import json
import random
import threading
from queue import Queue

import requests

max_prompt_random_tokens = 20


class Predictor(threading.Thread):
    def __init__(self, args, queue):
        super().__init__()
        self.args = args
        self.queue = queue

    def run(self):
        for _ in range(self.args.num_requests_per_thread):
            self._predict()

    def _predict(self):
        payload = self._format_payload()
        if self.args.demo_streaming:
            print(f"payload={payload}\n, output=")
        with requests.post(self._get_url(), json=payload, stream=True) as response:
            combined_text = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    text = self._extract_text(chunk)
                    if self.args.demo_streaming:
                        print(text, end="", flush=True)
                    else:
                        combined_text += text
        if not self.args.demo_streaming:
            self.queue.put_nowait(f"payload={payload}\n, output={combined_text}\n")

    def _extract_completion(self, chunk):
        chunk = chunk.decode("utf-8")
        if chunk.startswith("data:"):
            chunk = chunk[len("data:") :].split("\n")[0].strip()
            if chunk.startswith("[DONE]"):
                return ""
        return json.loads(chunk)["choices"][0]["text"]

    def _extract_chat(self, chunk):
        chunk = chunk.decode("utf-8")
        if chunk.startswith("data:"):
            chunk = chunk[len("data:") :].split("\n")[0].strip()
            if chunk.startswith("[DONE]"):
                return ""
        try:
            return json.loads(chunk)["choices"][0].get("message", {})["content"]
        except KeyError:
            return json.loads(chunk)["choices"][0].get("delta", {}).get("content", "")

    def _extract_text(self, chunk):
        if self.args.openai_api:
            if "chat" in self.args.api_endpoint:
                return self._extract_chat(chunk)
            else:
                return self._extract_completion(chunk)
        else:
            return json.loads(chunk).get("text", "")

    def _get_url(self):
        if self.args.openai_api:
            return f"http://localhost:8080/predictions/{self.args.model}/{self.args.model_version}/{self.args.api_endpoint}"
        else:
            return f"http://localhost:8080/predictions/{self.args.model}"

    def _format_payload(self):
        prompt_input = _load_curl_like_data(self.args.prompt_text)
        if "chat" in self.args.api_endpoint:
            assert self.args.prompt_json, "Use prompt json file for chat interface"
            assert self.args.openai_api, "Chat only work with openai api"
            prompt_input = json.loads(prompt_input)
            messages = prompt_input.get("messages", None)
            assert messages is not None
            rt = int(prompt_input.get("max_tokens", self.args.max_tokens))
            prompt_input["max_tokens"] = rt
            if self.args.demo_streaming:
                prompt_input["stream"] = True
            return prompt_input
        if self.args.prompt_json:
            prompt_input = json.loads(prompt_input)
            prompt = prompt_input.get("prompt", None)
            assert prompt is not None
            prompt_list = prompt.split(" ")
            rt = int(prompt_input.get("max_tokens", self.args.max_tokens))
        else:
            prompt_list = prompt_input.split(" ")
            rt = self.args.max_tokens
        rp = len(prompt_list)
        if self.args.prompt_randomize:
            rp = random.randint(0, max_prompt_random_tokens)
            rt = rp + self.args.max_tokens
            for _ in range(rp):
                prompt_list.insert(0, chr(ord("a") + random.randint(0, 25)))
        cur_prompt = " ".join(prompt_list)
        if self.args.prompt_json:
            prompt_input["prompt"] = cur_prompt
            prompt_input["max_tokens"] = rt
        else:
            prompt_input = {
                "prompt": cur_prompt,
                "max_tokens": rt,
            }
        if self.args.demo_streaming and self.args.openai_api:
            prompt_input["stream"] = True
        return prompt_input


def _load_curl_like_data(text):
    """
    Either use the passed string or load from a file if the string is `@filename`
    """
    if text.startswith("@"):
        try:
            with open(text[1:], "r") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {text[1:]}") from e
    else:
        return text


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        required=True,
        type=str,
        help="The model to use for generating text.",
    )
    parser.add_argument(
        "--prompt-text",
        required=True,
        type=str,
        help="Prompt text to use instead of generating one. It can be a file reference starting with an ampersand, e.g. `@prompt.txt`",
    )
    parser.add_argument(
        "--prompt-randomize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include a few random numbers in the generated prompt to avoid caching",
    )
    parser.add_argument(
        "-o",
        "--max-tokens",
        type=int,
        default=64,
        help="Max number of tokens to generate.",
    )
    parser.add_argument(
        "-t",
        "--num-threads",
        type=int,
        default=1,
        help="Enable the number of threads to execute prediction",
    )
    parser.add_argument(
        "-n",
        "--num-requests-per-thread",
        type=int,
        default=1,
        help="Execute the number of prediction in each thread",
    )
    parser.add_argument(
        "--prompt-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Flag the imput prompt is a json format with prompt parameters",
    )
    parser.add_argument(
        "--demo-streaming",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Demo streaming response, force num-requests-per-thread=1 and num-threads=1",
    )
    parser.add_argument(
        "--openai-api",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use OpenAI compatible API",
    )
    parser.add_argument(
        "--api-endpoint",
        type=str,
        default="v1/completions",
        help="OpenAI endpoint suffix",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="1.0",
        help="Model version. Default: 1.0",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.model) == 0:
        print("model argument can not be empty.")
        exit(1)

    if len(args.prompt_text) == 0:
        print("prompt argument can not be empty.")
        exit(1)

    if args.demo_streaming:
        args.num_threads = 1
        args.num_requests_per_thread = 1

    queue = Queue()
    predictors = []
    for i in range(args.num_threads):
        predictor = Predictor(args, queue)
        predictors.append(predictor)
        predictor.start()

    for predictor in predictors:
        predictor.join()

    print("Tasks are completed")

    while not queue.empty():
        print(queue.get())


if __name__ == "__main__":
    main()
