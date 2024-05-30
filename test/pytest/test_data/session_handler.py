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
CREATED = "_created"
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

        self.current_session = None

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
            self.close_session(input_batch[0]["session_id"])
            return [
                json.dumps(
                    {
                        "msg": "Session successfully closed",
                        "session_id": input_batch[0]["session_id"],
                    }
                )
            ]
        else:
            self.update_session_activity(input_batch[0]["session_id"])
            # perform other tasks
            pass

    def postprocess(self, inference_output):
        # set timed out sessions in header
        return inference_output

    def update_session_activity(self, session_id):
        self.store.set(f"{session_id}{LAST_ACTIVE}", struct.pack("d", time.time()))

    def open_session(self):
        if self.current_session is not None:
            # Worker was assigned a new session which means the previous session has times out
            # Lets clean up the model state here. and close the session (If not already happened).
            self.close_session(self.current_session)

        # This ID is actually generated in the frontend and will be read from header in ctx
        session_id = str(uuid.uuid4())
        self.current_session = session_id

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
        self.store.set(f"{session_id}{CREATED}", struct.pack("d", time.time()))
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
                    # The session was already closed, maybe through timeout
                    return
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
        self.current_session = None
