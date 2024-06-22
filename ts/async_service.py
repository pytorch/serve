import asyncio
import copy
import logging
import sys
import time
import traceback
import types
from asyncio.queues import Queue as AsyncQueue
from concurrent.futures.thread import ThreadPoolExecutor
from functools import partial
from queue import Empty, Queue
from threading import Thread

from ts.handler_utils.utils import create_predict_response
from ts.protocol.otf_message_handler import retrieve_msg
from ts.service import PREDICTION_METRIC, PredictionException, Service

logger = logging.getLogger(__name__)


async def predict(self, batch):
    """
    PREDICT COMMAND = {
        "command": "predict",
        "batch": [ REQUEST_INPUT ]
    }
    :param batch: list of request
    :return:

    """
    headers, input_batch, req_id_map = Service.retrieve_data_for_inference(batch)

    context = copy.deepcopy(self.context)

    context.request_ids = req_id_map
    context.request_processor = headers
    context.cl_socket = self.cl_socket
    metrics = context.metrics
    metrics.request_ids = req_id_map

    start_time = time.time()

    # noinspection PyBroadException
    try:
        print(f"{self._entry_point=}")
        ret = await self._entry_point(input_batch, context)
    except MemoryError:
        logger.error("System out of memory", exc_info=True)
        return create_predict_response(None, req_id_map, "Out of resources", 507)
    except PredictionException as e:
        logger.error("Prediction error", exc_info=True)
        return create_predict_response(None, req_id_map, e.message, e.error_code)
    except Exception as ex:  # pylint: disable=broad-except
        if "CUDA" in str(ex):
            # Handles Case A: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED (Close to OOM) &
            # Case B: CUDA out of memory (OOM)
            logger.error("CUDA out of memory", exc_info=True)
            return create_predict_response(None, req_id_map, "Out of resources", 507)
        else:
            logger.warning("Invoking custom service failed.", exc_info=True)
            return create_predict_response(None, req_id_map, "Prediction failed", 503)

    if not isinstance(ret, list):
        logger.warning(
            "model: %s, Invalid return type: %s.",
            context.model_name,
            type(ret),
        )
        return create_predict_response(
            None, req_id_map, "Invalid model predict output", 503
        )

    if len(ret) != len(input_batch):
        logger.warning(
            "model: %s, number of batch response mismatched, expect: %d, got: %d.",
            context.model_name,
            len(input_batch),
            len(ret),
        )
        return create_predict_response(
            None, req_id_map, "number of batch response mismatched", 503
        )

    duration = round((time.time() - start_time) * 1000, 2)
    metrics.add_time(PREDICTION_METRIC, duration)

    return create_predict_response(
        ret, req_id_map, "Prediction success", 200, context=context
    )


class AsyncService(object):
    def __init__(self, service):
        self.service = service
        self.service.predict = types.MethodType(predict, self.service)
        self.in_queue = Queue()
        self.out_queue = None
        self.exception_queue = Queue()
        self.loop = None

    def receive_requests(self):
        while True:
            logging.debug("Waiting for new message")
            cmd, msg = retrieve_msg(self.service.cl_socket)

            if cmd == b"I":
                logging.debug(f"Putting msg in queue: {msg}")
                self.in_queue.put(msg)
            else:
                logging.debug(f"Unexpected request: {cmd}")

    async def call_predict(self, batch):
        response = await self.service.predict(batch)
        await self.out_queue.put(response)

    def fetch_batches(self):
        MAX_WAIT = 0.1
        BATCH_SIZE = 1
        while True:
            st = time.time()
            batch = []
            try:
                request = self.in_queue.get()
                batch += request
                while len(batch) < BATCH_SIZE and (time.time() - st) < MAX_WAIT:
                    timeout = max(0, MAX_WAIT - (time.time() - st))
                    request = self.in_queue.get(timeout=timeout)
                    batch += request
            except Empty:
                pass

            asyncio.run_coroutine_threadsafe(self.call_predict(batch), self.loop)

    def send_responses(self):
        while True:
            future = asyncio.run_coroutine_threadsafe(self.out_queue.get(), self.loop)
            self.service.cl_socket.sendall(future.result())

    def run(self):
        async def main():
            self.loop = asyncio.get_running_loop()
            self.out_queue = (
                AsyncQueue(loop=self.loop)
                if sys.version_info <= (3, 9)
                else AsyncQueue()
            )

            def catch_all(func):
                try:
                    func()
                except Exception as e:
                    self.exception_queue.put(str(traceback.format_exc()))

            threads = []
            threads.append(Thread(target=partial(catch_all, self.fetch_batches)))
            threads[-1].start()
            threads.append(Thread(target=partial(catch_all, self.receive_requests)))
            threads[-1].start()
            threads.append(Thread(target=partial(catch_all, self.send_responses)))
            threads[-1].start()

            logging.debug("Running async run")

            def check_threads():
                while True:
                    if not all(t.is_alive() for t in threads):
                        return
                    time.sleep(1)

            with ThreadPoolExecutor(1) as executor:
                await asyncio.get_event_loop().run_in_executor(executor, check_threads)

        asyncio.get_event_loop().run_until_complete(main())
        if not self.exception_queue.empty():
            return self.exception_queue.get()
        else:
            return None
