import os
import queue
import threading
import time
from copy import copy
from dataclasses import dataclass
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


@dataclass
class WorkerThread:
    event: threading.Event
    thread: threading.Thread


class MicroBatching(object):
    def __init__(
        self, parent_handler, micro_batch_size: int = 1, parallelism: Dict = None
    ):
        self.handler = parent_handler
        self.micro_batch_size = micro_batch_size
        self._parallelism = parallelism if parallelism is not None else {}
        self.thread_groups = {c: [] for c in HANDLER_METHODS}
        self.queues = {}
        self.terminate = threading.Event()
        self._create_queues()
        self._update_threads()

    @property
    def parallelism(self):
        return copy(self._parallelism)

    @parallelism.setter
    def parallelism(self, new_parallelism):
        self._parallelism.update(new_parallelism)
        self._update_threads()

    def _create_queues(self):
        self.queues[HANDLER_METHODS[0] + "_in"] = queue.Queue()
        for i in range(len(HANDLER_METHODS) - 1):
            self.queues[HANDLER_METHODS[i] + "_out"] = queue.Queue()
            self.queues[HANDLER_METHODS[i + 1] + "_in"] = self.queues[
                HANDLER_METHODS[i] + "_out"
            ]
        self.queues[HANDLER_METHODS[-1] + "_out"] = queue.Queue()

    def shutdown(self):
        for _, tg in self.thread_groups.items():
            for t in tg:
                t.event.set()
                t.thread.join()

    def _update_threads(self):
        for c in HANDLER_METHODS:
            tgt_parallelism = self._parallelism.get(c, 2)
            assert tgt_parallelism >= 0
            cur_parallelism = lambda: len(self.thread_groups[c])

            while tgt_parallelism > cur_parallelism():
                in_queue = self.queues[c + "_in"]
                out_queue = self.queues[c + "_out"]
                call = getattr(self.handler, c)
                event = threading.Event()
                t = threading.Thread(
                    target=execute_call,
                    args=(in_queue, out_queue, call, event),
                )
                t.start()
                self.thread_groups[c].append(WorkerThread(event, t))

            while tgt_parallelism < cur_parallelism():
                self.thread_groups[c][-1].event.set()
                self.thread_groups[c][-1].thread.join()
                self.thread_groups[c].pop()

    def handle(self, data):

        num_batches = 0
        for idx, i in enumerate(range(0, len(data), self.micro_batch_size)):
            self.queues[HANDLER_METHODS[0] + "_in"].put_nowait(
                (idx, data[i : i + self.micro_batch_size])
            )
            num_batches += 1

        output = []
        while len(output) != num_batches:
            output.append(self.queues[HANDLER_METHODS[-1] + "_out"].get())

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
