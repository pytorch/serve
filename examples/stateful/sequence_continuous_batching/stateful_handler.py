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

            request = row.get("data") or row.get("body")
            if isinstance(request, (bytes, bytearray)):
                request = request.decode("utf-8")

            # -1: cancel
            if int(request) == -1:
                self.context.cache[sequence_id]["cancel"] = True
                results.append(int(request))
            elif prev is None:
                logger.info(
                    f"Close the sequence:{sequence_id} without open session request"
                )
                self.context.cache[sequence_id]["end"] = True
                self.context.cache[req_id]["end"] = True
                self.context.set_response_header(
                    idx, self.context.header_key_sequence_end, sequence_id
                )
                results.append(int(request))
            else:
                val = prev + int(request)
                self.cache[sequence_id] = val
                # 0: end
                if int(request) == 0:
                    self.context.set_response_header(
                        idx, self.context.header_key_sequence_end, sequence_id
                    )
                # -3: test streaming
                elif int(request) == -3:
                    time.sleep(1)

                results.append(val)

            req_id = self.context.get_request_id(idx)
            if req_id not in self.context.cache:
                self.context.cache[req_id] = {
                    "stopping_criteria": self._create_stopping_criteria(
                        req_id=req_id, seq_id=sequence_id
                    ),
                }

                self.context.cache[sequence_id]["num_requests"] += 1

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
                self.counter = 5

            def __call__(self, res):
                # sequence end
                if self.outer.context.cache[seq_id]["end"]:
                    self.outer.clean_up(self.seq_id, self.req_id, True)
                    return True
                # cancel
                elif self.outer.context.cache[seq_id]["cancel"] or self.counter == 0:
                    self.outer.clean_up(self.seq_id, self.req_id, False)
                    return True
                else:
                    self.counter -= 1

                return False

        return StoppingCriteria(outer=self, req_id=req_id, seq_id=seq_id)
