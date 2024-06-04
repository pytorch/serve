import logging
import time
from abc import ABC

from lru import LRU

from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class StatefulHandler(BaseHandler, ABC):
    DEFAULT_CAPACITY = 10

    def __init__(self):
        super().__init__()
        self.cache: LRU = None

    def initialize(self, ctx: Context):
        """
        Loads the model and Initializes the necessary artifacts
        """

        # context cache includes 2 types of keys
        # key1: sequence_id
        # value is a dict which records the sequence's status: start, end, cancel, number of the requests in this batch.
        #
        # key2: request_id
        # value is a dict which records a request's streaming status:
        # None(ie. non response streaming request), True or False (ie. streaming complete or not)
        ctx.cache = {}
        if ctx.model_yaml_config["handler"] is not None:
            self.cache = LRU(
                int(
                    ctx.model_yaml_config["handler"]
                    .get("cache", {})
                    .get("capacity", StatefulHandler.DEFAULT_CAPACITY)
                )
            )

        self.initialized = True

    def preprocess(self, data):
        """
        Preprocess function to convert the request input to a tensor(Torchserve supported format).
        The user needs to override to customize the pre-processing

        Args :
            data (list): List of the data from the request input.

        Returns:
            tensor: Returns the tensor data of the input
        """

        results = []
        for idx, row in enumerate(data):
            sequence_id = self.context.get_sequence_id(idx)
            # SageMaker sticky router relies on response header to identify the sessions
            # The sequence_id from request headers must be set in response headers
            self.context.set_response_header(
                idx, self.context.header_key_sequence_id, sequence_id
            )

            # check if sequence_id exists
            if self.context.get_request_header(
                idx, self.context.header_key_sequence_start
            ):
                prev = int(0)
                self.context.cache[sequence_id] = {
                    "start": True,
                    "cancel": False,
                    "end": False,
                    "num_requests": 0,
                }
            elif self.cache.has_key(sequence_id):
                prev = int(self.cache[sequence_id])
            else:
                prev = None
                logger.error(
                    f"Not received sequence_start request for sequence_id:{sequence_id} before"
                )

            req_id = self.context.get_request_id(idx)
            # process a new request
            if req_id not in self.context.cache:
                logger.info(
                    f"received a new request sequence_id={sequence_id}, request_id={req_id}"
                )
                request = row.get("data") or row.get("body")
                if isinstance(request, (bytes, bytearray)):
                    request = request.decode("utf-8")

                self.context.cache[req_id] = {
                    "stopping_criteria": self._create_stopping_criteria(
                        req_id=req_id, seq_id=sequence_id
                    ),
                    "stream": True,
                }
                self.context.cache[sequence_id]["num_requests"] += 1

                if type(request) is dict and "input" in request:
                    request = request.get("input")

                # -1: cancel
                if int(request) == -1:
                    self.context.cache[sequence_id]["cancel"] = True
                    self.context.cache[req_id]["stream"] = False
                    results.append(int(request))
                elif prev is None:
                    logger.info(
                        f"Close the sequence:{sequence_id} without open session request"
                    )
                    self.context.cache[sequence_id]["end"] = True
                    self.context.cache[req_id]["stream"] = False
                    self.context.set_response_header(
                        idx, self.context.header_key_sequence_end, sequence_id
                    )
                    results.append(int(request))
                else:
                    val = prev + int(request)
                    self.cache[sequence_id] = val
                    # 0: end
                    if int(request) == 0:
                        self.context.cache[sequence_id]["end"] = True
                        self.context.cache[req_id]["stream"] = False
                        self.context.set_response_header(
                            idx, self.context.header_key_sequence_end, sequence_id
                        )
                    # non stream input:
                    elif int(request) % 2 == 0:
                        self.context.cache[req_id]["stream"] = False

                    results.append(val)
            else:
                # continue processing stream
                logger.info(
                    f"received continuous request sequence_id={sequence_id}, request_id={req_id}"
                )
                time.sleep(1)
                results.append(prev)

        return results

    def inference(self, data, *args, **kwargs):
        return data

    def postprocess(self, data):
        """
        The post process function makes use of the output from the inference and converts into a
        Torchserve supported response output.

        Returns:
            List: The post process function returns a list of the predicted output.
        """
        self.context.stopping_criteria = [
            self.context.cache[req_id]["stopping_criteria"]
            for req_id in self.context.request_ids.values()
        ]

        return data

    def clean_up(self, seq_id, req_id, del_seq):
        # clean up
        self.context.cache[seq_id]["num_requests"] -= 1
        if self.context.cache[seq_id]["num_requests"] == 0 and del_seq:
            del self.context.cache[seq_id]
        del self.context.cache[req_id]

    def _create_stopping_criteria(self, req_id, seq_id):
        class StoppingCriteria(object):
            def __init__(self, outer, req_id, seq_id):
                self.req_id = req_id
                self.seq_id = seq_id
                self.outer = outer
                self.counter = 10

            def __call__(self, res):
                # sequence end
                if self.outer.context.cache[seq_id]["end"]:
                    ret = True if self.outer.context.cache[req_id]["stream"] else None
                    self.outer.clean_up(self.seq_id, self.req_id, True)
                    logger.info(f"end sequence_id={self.seq_id}, ret={ret}")
                    return ret
                # cancel
                elif self.outer.context.cache[seq_id]["cancel"]:
                    ret = True if self.outer.context.cache[req_id]["stream"] else None
                    self.outer.clean_up(self.seq_id, self.req_id, False)
                    logger.info(
                        f"cancel sequence_id={self.seq_id}, request_id={self.req_id}, ret={ret}"
                    )
                    if self.outer.context.cache[seq_id]["num_requests"] == 0:
                        self.outer.context.cache[seq_id]["cancel"] = False
                    return ret
                # start
                elif self.outer.context.cache[seq_id]["start"]:
                    self.outer.clean_up(self.seq_id, self.req_id, False)
                    logger.info(
                        f"start sequence_id={self.seq_id}, request_id={self.req_id}, ret=None"
                    )
                    self.outer.context.cache[seq_id]["start"] = False
                    return None
                # non stream
                elif not self.outer.context.cache[req_id]["stream"]:
                    self.outer.clean_up(self.seq_id, self.req_id, False)
                    logger.info(
                        f"test non stream sequence_id={self.seq_id}, request_id={self.req_id}, ret=None"
                    )
                    return None
                # stream complete
                elif self.counter == 0:
                    self.outer.clean_up(self.seq_id, self.req_id, False)
                    logger.info(
                        f"finish sequence_id={self.seq_id}, request_id={self.req_id}, ret=True"
                    )
                    return True
                # stream running
                else:
                    self.counter -= 1
                    logger.info(
                        f"continue sequence_id={self.seq_id}, request_id={self.req_id}, ret=False"
                    )

                return False

        return StoppingCriteria(outer=self, req_id=req_id, seq_id=seq_id)
