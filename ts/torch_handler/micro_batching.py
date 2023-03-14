import asyncio
import os
import time

try:

    PROFILER_AVAILABLE = True
except ImportError:
    PROFILER_AVAILABLE = False


async def execute_call(in_queue, out_queue, handle):
    while True:
        in_data = await in_queue.get()
        out_data = handle(in_data)
        await out_queue.put(out_data)


async def execute_serial_calls(calls, batches):

    queues = [asyncio.Queue()]
    tasks = []

    for c in calls:
        queues.append(asyncio.Queue())
        for _ in range(2):
            t = asyncio.create_task(execute_call(queues[-2], queues[-1], c))
            tasks.append(t)

    for b in batches:
        queues[0].put_nowait(b)

    output = []
    while len(output) != len(batches):
        output.append(await queues[-1].get())

    for t in tasks:
        t.cancel()

    return output


class MicroBatchingHandler(object):
    def __init__(self, parent_handler, micro_batch_size=1):
        self.handler = parent_handler
        self.micro_batch_size = micro_batch_size

    def handle(self, data):

        serial_calls = (
            self.handler.preprocess,
            self.handler.inference,
            self.handler.postprocess,
        )

        micro_batches = []
        for i in range(0, len(data), self.micro_batch_size):
            micro_batches.append(data[i : i + self.micro_batch_size])

        execute = execute_serial_calls(serial_calls, micro_batches)

        output = asyncio.run(execute)

        return [item for batch in output for item in batch]

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
