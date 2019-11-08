

"""
Single start point for system metrics and process metrics script

"""
import logging
import sys

from mms.metrics import system_metrics
from mms.metrics.process_memory_metric import check_process_mem_usage

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, format="%(message)s", level=logging.INFO)

    system_metrics.collect_all(sys.modules['mms.metrics.system_metrics'])

    check_process_mem_usage(sys.stdin)
