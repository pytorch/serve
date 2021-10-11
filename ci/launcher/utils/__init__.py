import logging
import sys

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stderr))

UBUNTU_18_BASE_DLAMI_US_WEST_2 = "ami-032e40ca6b0973cf2"
DEFAULT_REGION = "us-west-2"

GPU_INSTANCES = ["p2", "p3", "p4", "g2", "g3", "g4"]

UL_AMI_LIST = ["ami-032e40ca6b0973cf2"]