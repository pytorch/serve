import asyncio
import copy
import logging
import time
import types
from asyncio.queues import Queue as AsyncQueue
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
        ret = self._entry_point(input_batch, context)
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
        self.out_queue = AsyncQueue()
        self.executor = None
        self.loop = None

    def receive_requests(self):
        while True:
            logging.info("Waiting for new message")
            cmd, msg = retrieve_msg(self.service.cl_socket)

            if cmd == b"I":
                logging.info(f"Putting msg in queue: {msg}")
                self.in_queue.put(msg)
            else:
                logging.info(f"Unexpected request: {cmd}")

    async def call_predict(self, batch):
        response = await self.service.predict(batch)
        await self.out_queue.put(response)

    def fetch_batches(self):
        MAX_WAIT = 0.1
        BATCH_SIZE = 8
        while True:
            st = time.time()
            batch = []
            try:
                logging.info(f"Waiting for INF")
                request = self.in_queue.get()
                logging.info(f"Got an INF")
                batch += request
                while len(batch) < BATCH_SIZE and (time.time() - st) < MAX_WAIT:
                    timeout = max(0, MAX_WAIT - (time.time() - st))
                    request = self.in_queue.get(timeout=timeout)
                    batch += request
            except Empty:
                pass

            logging.info(f"Call predict with batch_size: {len(batch)}")
            future = asyncio.run_coroutine_threadsafe(
                self.call_predict(batch), self.loop
            )
            future.result()

    def send_responses(self):
        while True:
            future = asyncio.run_coroutine_threadsafe(self.out_queue.get(), self.loop)
            self.service.cl_socket.sendall(future.result())

    def run(self):
        with asyncio.Runner() as runner:
            self.loop = runner.get_loop()

            fetch = Thread(target=self.fetch_batches)
            fetch.start()
            receive = Thread(target=self.receive_requests)
            receive.start()
            send = Thread(target=self.send_responses)
            send.start()

            logging.info("Running async run")
            # runner.run(self.async_main())
            self.loop.run_forever()
