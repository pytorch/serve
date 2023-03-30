import os
import queue
import threading
import time
from typing import Dict

try:

    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


def execute_call(in_queue, out_queue, handle, event):
    while not event.is_set():
        try:
            idx, in_data = in_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        out_data = handle(in_data)
        out_queue.put((idx, out_data))


HANDLER_METHODS = ["preprocess", "inference", "postprocess"]


class MicroBatching(object):
    def __init__(
        self, parent_handler, micro_batch_size: int = 1, parallelism: Dict = None
    ):
        self.handler = parent_handler
        self.micro_batch_size = micro_batch_size
        self.parallelism = parallelism if parallelism is not None else {}
        self.threads = []
        self.queues = []
        self.terminate = threading.Event()
        self.initialize_threads()

    def shutdown(self):
        self.terminate.set()
        for t in self.threads:
            t.join()

    def initialize_threads(self):
        calls = (
            (getattr(self.handler, c), self.parallelism.get(c, 2))
            for c in HANDLER_METHODS
        )

        self.queues.append(queue.Queue())
        tasks = []

        for c, p in calls:
            self.queues.append(queue.Queue())
            for _ in range(p):
                t = threading.Thread(
                    target=execute_call,
                    args=(self.queues[-2], self.queues[-1], c, self.terminate),
                )
                t.start()
                tasks.append(t)

    def handle(self, data):

        num_batches = 0
        for idx, i in enumerate(range(0, len(data), self.micro_batch_size)):
            self.queues[0].put_nowait((idx, data[i : i + self.micro_batch_size]))
            num_batches += 1

        output = []
        while len(output) != num_batches:
            output.append(self.queues[-1].get())

        return [item for batch in sorted(output) for item in batch[1]]

    def __call__(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """

        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.handler.context = context
        metrics = self.handler.context.metrics

        is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
        if is_profiler_enabled:
            if PROFILER_AVAILABLE:
                output, _ = self.handler._infer_with_profiler(data=data)
            else:
                raise RuntimeError(
                    "Profiler is enabled but current version of torch does not support."
                    "Install torch>=1.8.1 to use profiler."
                )
        else:
            if self.handler._is_describe():
                output = [self.handler.describe_handle()]
            elif self.handler._is_explain():
                data_preprocess = self.handler.preprocess(data)
                output = self.handler.explain_handle(data_preprocess, data)
            else:
                output = self.handle(data)

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        return output
