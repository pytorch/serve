import argparse
import logging
import sys
from pathlib import Path
from torch.distributed import FileStore
from ts.metrics.dimension import Dimension
from ts.metrics.metric import Metric


if __name__ == '__main__':

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

    args = parser.parse_args()

    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    store_path = Path(args.store_folder) / f"{args.model_name}_store"
    store = FileStore(store_path.as_posix(), -1)
    open_sessions_num = len(store.get("open_sessions").decode("utf-8").split(";"))
    dimension = [Dimension("Level", "Host")]
    logging.info(str(Metric("OpenSessions", open_sessions_num, "count", dimension)))
    logging.info("")