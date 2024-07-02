import subprocess
import threading
from io import TextIOWrapper
from queue import Empty, Full, Queue
from subprocess import PIPE, STDOUT, Popen

import requests


def stop(wait=True):
    cmd = ["torchserve", "--stop"]
    if wait:
        cmd += ["--foreground"]
    subprocess.run(cmd)


class Tee(threading.Thread):
    def __init__(self, reader):
        super().__init__()
        self.reader = reader
        self.queue1 = Queue(maxsize=1000)
        self.queue2 = Queue(maxsize=1000)

    def run(self):
        for line in self.reader:
            try:
                self.queue1.put_nowait(line)
            except Full:
                pass
            try:
                self.queue2.put_nowait(line)
            except Full:
                pass

        # If queues are full, clear them out and send None
        # This is probably not necessary as the runner consuming the queue will have presumably died
        # But we want to avoid a confusing additional exception if there is any error
        try:
            if self.queue1.full():
                while not self.queue1.empty():
                    self.queue1.get(False)
        except Empty:
            pass

        try:
            if self.queue2.full():
                while not self.queue2.empty():
                    self.queue2.get(False)
        except Empty:
            pass

        self.queue1.put_nowait(None)
        self.queue2.put_nowait(None)


class PrintTillTheEnd(threading.Thread):
    def __init__(self, queue):
        super().__init__()
        self._queue = queue

    def run(self):
        while True:
            line = self._queue.get()
            if not line:
                break
            print(line.strip())


def start(
    model_store=None,
    snapshot_file=None,
    no_config_snapshots=False,
    plugin_folder=None,
    disable_token=False,
    models=None,
    enable_model_api=False,
):
    stop()
    cmd = ["torchserve", "--start"]
    cmd.extend(["--model-store", model_store])
    if plugin_folder:
        cmd.extend(["--plugins-path", plugin_folder])
    if snapshot_file:
        cmd.extend(["--ts-config", snapshot_file])
    if no_config_snapshots:
        cmd.extend(["--no-config-snapshots"])
    if disable_token:
        cmd.append("--disable-token-auth")
    if models:
        cmd.extend(["--models", models])
    if enable_model_api:
        cmd.extend(["--enable-model-api"])
    p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT)
    for line in p.stdout:
        print(line.decode("utf8").strip())
        if "Model server started" in str(line).strip():
            break

    splitter = Tee(TextIOWrapper(p.stdout))
    splitter.start()
    print_thread = PrintTillTheEnd(splitter.queue1)
    print_thread.start()

    return splitter.queue2


def register_model_with_params(params):
    response = requests.post("http://localhost:8081/models", params=params)
    return response


def register_model(model_name, url):
    params = (
        ("model_name", model_name),
        ("url", url),
        ("initial_workers", "1"),
        ("synchronous", "true"),
    )
    return register_model_with_params(params)
