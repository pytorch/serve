import logging
import sys

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

DEFAULT_REGION = "us-west-2"

GPU_INSTANCES = ["p2", "p3", "p4", "g2", "g3", "g4"]
