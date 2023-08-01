import queue
from typing import Dict

from ts.handler_utils.micro_batching import HANDLER_METHODS, MicroBatching


class ContinuousBatching(MicroBatching):
    def __init__(
        self, parent_handler, micro_batch_size: int = 1, parallelism: Dict = None
    ):
        super(ContinuousBatching, self).__init__()

    def handle(self, data):
        batch_size = len(data)
        for idx, req in enumerate(data):
            self.queues[HANDLER_METHODS[0] + "_in"].put_nowait((idx, req))

        output = []
        while len(output) != batch_size:
            output.append(self.queues[HANDLER_METHODS[-1] + "_out"].get())

        return [item for batch in sorted(output) for item in batch[1]]

    def execute_call(self, in_queue, out_queue, handle, event):
        repeats = 1
        if handle.__name__ == "inference":
            repeats = self.micro_batch_size
        while not event.is_set():
            in_data = []
            for i in range(0, repeats):
                try:
                    idx, req = in_queue.get(timeout=0.5)
                    in_data.append(req)
                except queue.Empty:
                    continue
            out_data = handle(in_data)
            out_queue.put(out_data)

    def refill(self, idx, ids, requests):
        try:
            idx, req = self.queues["inference_in"].get_nowait()
            ids[idx] = idx
            requests[idx] = req
        except queue.Empty:
            pass
