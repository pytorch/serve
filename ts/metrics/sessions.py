import argparse
import logging
import struct
import sys
import time
from pathlib import Path

from torch.distributed import FileStore

from ts.metrics.dimension import Dimension
from ts.metrics.metric import Metric

LAST_ACTIVE = "_last_activity"
CREATED = "_created"
OPEN_SESSION = "open_sessions"


def close_session(store, session_id):
    ret = store.compare_set(OPEN_SESSION, session_id, "").decode("utf-8")
    if ret != "":
        # This session was not the only session
        if ret == session_id:
            # For some reason the session was closed before the key was set (should never happen)
            # After the initial session the key should always be present (can be "" if no session is open)
            return

        success = False
        while not success:
            if session_id not in ret:
                # The session was already closed through a different worker, maybe through timeout
                return
            else:
                # Remove session_id and set in store
                remaining_open_session = ";".join(
                    filter(lambda x: x != session_id, ret.split(";"))
                )
                ret = store.compare_set(
                    OPEN_SESSION, ret, remaining_open_session
                ).decode("utf-8")
                success = ret == remaining_open_session
    store.delete_key(f"{session_id}{LAST_ACTIVE}")
    store.delete_key(f"{session_id}{CREATED}")
    return True


def check_session_timeouts(store, timeout, activity_timeout):
    now = time.time()
    ret = store.compare_set(OPEN_SESSION, "", "").decode("utf-8")
    if ret == "":
        # OPEN_SESSION was either empty or key did not exist
        return

    for session_id in ret.split(";"):
        created = store.compare_set(f"{session_id}{CREATED}", "DOES NOT EXIST", "")
        last_active = store.compare_set(
            f"{session_id}{LAST_ACTIVE}", "DOES NOT EXIST", ""
        )
        if "DOES NOT EXIST" in (created, last_active):
            # Was already cleaned up
            continue

        created = float(struct.unpack("d", created)[0])
        last_active = float(struct.unpack("d", last_active)[0])

        if now - created > timeout or now - last_active > activity_timeout:
            logging.info(f"Session timout: {session_id}")
            close_session(store, session_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--store_folder",
        dest="store_folder",
        help="Folder for store",
        type=str,
        default="/tmp",
    )

    parser.add_argument(
        "--model_name",
        dest="model_name",
        help="Model name",
        type=str,
        default="model_name",
    )

    parser.add_argument(
        "--timeout",
        dest="timeout",
        help="Total time until sessions times out in seconds",
        type=int,
        default=20 * 60,
    )

    parser.add_argument(
        "--activity_timeout",
        dest="activity_timeout",
        help="Time without activity until sessions times out in seconds",
        type=int,
        default=5 * 60,
    )

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    store_path = Path(args.store_folder) / f"{args.model_name}_store"
    store = FileStore(store_path.as_posix(), -1)

    check_session_timeouts(store, args.timeout, args.activity_timeout)

    open_sessions = store.compare_set(
        OPEN_SESSION, "SOMETHING UNEXPECTED WHICH RETURNS CURRENT VALUE", ""
    ).decode("utf-8")
    open_sessions_num = 0 if open_sessions == "" else len(open_sessions.split(";"))
    dimension = [Dimension("Level", "Host")]
    logging.info(str(Metric("OpenSessions", open_sessions_num, "count", dimension)))
    logging.info("")
