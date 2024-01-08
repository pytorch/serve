import argparse
import json
import random
import threading
from queue import Queue

import orjson
import requests


class Predictor(threading.Thread):
    def __init__(self, args, queue):
        threading.Thread.__init__(self)
        self.args = args
        self.queue = queue

    def run(self):
        for _ in range(self.args.num_requests_per_thread):
            self._predict()

    def _predict(self):
        payload = self._format_payload()
        with requests.post(
            self._get_url(), json=json.dumps(payload), stream=True
        ) as response:
            combined_text = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    data = orjson.loads(chunk)
                    combined_text += data["text"]
        self.queue.put_nowait(f"payload={payload}\n, output={combined_text}\n")

    def _get_url(self):
        return f"http://localhost:8080/predictions/{self.args.model}"

    def _format_payload(self):
        prompt = _load_curl_like_data(self.args.prompt_text)
        prompt_list = prompt.split(" ")
        rp = len(prompt_list)
        rt = self.args.max_tokens
        if self.args.prompt_randomize:
            rp = random.randint(1, len(prompt_list))
            rt = random.randint(10, self.args.max_tokens)
        cur_prompt_list = prompt_list[:rp]
        cur_prompt = " ".join(cur_prompt_list)
        return {
            "prompt": cur_prompt,
            "max_new_tokens": rt,
        }


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
        type=str,
        help="The model to use for generating text. If not specified we will pick the first model from the service as returned by /v1/models",
    )
    parser.add_argument(
        "--prompt-text",
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

    return parser.parse_args()


def main():
    args = parse_args()
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
