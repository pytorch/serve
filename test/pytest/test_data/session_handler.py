import json
import logging
import struct
import time
import uuid
from pathlib import Path

from torch.distributed import FileStore

from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

LAST_ACTIVE = "_last_activity"
OPEN_SESSION = "open_sessions"
TIMEOUT = 2


class CancelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.initialized = False

    def initialize(self, ctx):
        self.ctx = ctx
        self.manifest = ctx.manifest

        identifier = f"{self.manifest['model']['modelName']}_store"

        self.store_path = Path("/tmp") / identifier
        self.store = FileStore(self.store_path.as_posix(), -1)

        self.initialized = True

    def preprocess(self, requests):
        assert len(requests) == 1, "Batch size is expected to be 1!"

        input_data = requests[0].get("data")
        if input_data is None:
            input_data = requests[0].get("body")
        if isinstance(input_data, (bytes, bytearray)):
            input_data = input_data.decode("utf-8")

        input_data = json.loads(input_data)

        logger.info("Received data: '%s'", input_data)
        logger.info("Received data type: '%s'", type(input_data))

        assert (
            "request_type" in input_data
        ), "Application layer request type not provided"

        return [input_data]

    def inference(self, input_batch):
        if input_batch[0]["request_type"] == "open_session":
            return self.open_session()
        elif input_batch[0]["request_type"] == "close_session":
            success = self.close_session(input_batch[0]["session_id"])
            msg = (
                "Session successfully closed"
                if success
                else "Session was already closed"
            )
            return [
                json.dumps({"msg": msg, "session_id": input_batch[0]["session_id"]})
            ]
        else:
            self.update_session_activity(input_batch[0]["session_id"])
            # perform other tasks
            pass

    def postprocess(self, inference_output):
        timed_out_session = self.check_session_timeouts()
        # set timed out sessions in header
        return inference_output

    def update_session_activity(self, session_id):
        self.store.set(f"{session_id}{LAST_ACTIVE}", struct.pack("d", time.time()))

    def open_session(self):
        session_id = str(uuid.uuid4())

        logger.info(f"Opening Session {session_id}")
        # Try if this is the first session
        new_open_sessions = session_id
        ret = self.store.compare_set(OPEN_SESSION, "", new_open_sessions).decode(
            "utf-8"
        )
        while ret != new_open_sessions:
            # There are other open sessions
            new_open_sessions = ";".join(ret.split(";") + [session_id])
            ret = self.store.compare_set(OPEN_SESSION, ret, new_open_sessions).decode(
                "utf-8"
            )
        self.update_session_activity(session_id)
        return [
            json.dumps({"msg": "Session successfully opened", "session_id": session_id})
        ]

    def close_session(self, session_id):
        ret = self.store.compare_set(OPEN_SESSION, session_id, "").decode("utf-8")
        print(f"Closing {session_id=} Sessions open: {ret=}")
        if ret != "":
            # This session was not the only session
            if ret == session_id:
                # For some reason the session was closed before the key was set (should never happen -> error)
                # After the initial session the key should always be present (can be "" if no session is open)
                raise RuntimeError("open_sessions key was not set")

            success = False
            while not success:
                if session_id not in ret:
                    # The session was already closed through a different worker, maybe through timeout
                    return False
                else:
                    # Remove session_id and set in store
                    remaining_open_session = ";".join(
                        filter(lambda x: x != session_id, ret.split(";"))
                    )
                    ret = self.store.compare_set(
                        OPEN_SESSION, ret, remaining_open_session
                    ).decode("utf-8")
                    success = ret == remaining_open_session
        self.store.delete_key(f"{session_id}{LAST_ACTIVE}")
        return True

    def check_session_timeouts(self):
        now = time.time()
        ret = self.store.compare_set(OPEN_SESSION, "", "").decode("utf-8")
        if ret == "":
            # OPEN_SESSION was either empty or key did not exist
            return []
        timed_out_sessions = []
        for session_id in ret.split(";"):
            last_active = self.store.compare_set(
                f"{session_id}{LAST_ACTIVE}", "DOES NOT EXIST", ""
            )
            if last_active == "DOES NOT EXIST":
                # Was already cleaned up
                continue
            last_active = float(struct.unpack("d", last_active)[0])
            if now - last_active > TIMEOUT:
                print(f"Timeout: {session_id=} {now=} {last_active=}")
                if self.close_session(session_id):
                    timed_out_sessions += [session_id]
        return timed_out_sessions
