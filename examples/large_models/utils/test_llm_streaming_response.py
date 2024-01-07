import argparse
import random
import threading
from queue import Queue

import requests


class Predictor(threading.Thread):
    def __init__(self, args, queue):
        threading.Thread.__init__(self)
        self.args = args
        self.queue = queue

    def run(self):
        for _ in range(self.args.num_requests_per_thread):
            self._predict(self.args, self.queue)

    def _predict(self):
        payload = self._format_payload(self.args)
        with requests.post(self._get_url(self.args), json=payload) as response:
            combined_text = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    data = chunk.decode("utf-8")
                    combined_text += data["text"]

        with self.queue.mutex:
            self.queue.put_nowait(f'prompt={payload["data"]}\n, output={combined_text}')

    def _get_url(self):
        return f"http://localhost:8080/predictions/{self.args.model}"

    def _format_payload(self):
        prompt_list = self.args.prompt.split(" ")
        r = random.randint(1, len(prompt_list))
        cur_prompt_list = prompt_list[:r]
        cur_prompt = " ".join(cur_prompt_list)
        return {
            "data": cur_prompt,
            "max_new_token": random.randint(10, self.args.max_tokens),
        }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="The model to use for generating text. If not specified we will pick the first model from the service as returned by /v1/models",
    )
    parser.add_argument(
        "-p",
        "--prompt-tokens",
        env_var="PROMPT_TOKENS",
        type=int,
        default=512,
        help="Length of the prompt in tokens. Default 512",
    )
    parser.add_argument(
        "--prompt-chars",
        env_var="PROMPT_CHARS",
        type=int,
        help="Length of the prompt in characters.",
    )
    parser.add_argument(
        "--prompt-text",
        env_var="PROMPT_TEXT",
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
        env_var="MAX_TOKENS",
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

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    queue = Queue()
    predictors = []
    for i in range(args.num_threads):
        predictor = Predictor(args, queue)
        predictor.start()
        predictors.append(predictor)

    for predictor in predictors:
        predictor.join()

    while not queue.empty():
        print(queue.get())


if __name__ == "__main__":
    main()
