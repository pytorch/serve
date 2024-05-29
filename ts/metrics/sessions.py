import logging
import sys
from pathlib import Path
from torch.distributed import FileStore
from ts.metrics.dimension import Dimension
from ts.metrics.metric import Metric
if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)
    store_path = Path("/tmp") / "model_name_store"
    store = FileStore(store_path.as_posix(), -1)
    open_sessions_num = len(store.get("open_sessions").decode("utf-8").split(";"))
    dimension = [Dimension("Level", "Host")]
    logging.info(str(Metric("OpenSessions", open_sessions_num, "count", dimension)))
    logging.info("")