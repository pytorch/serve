import csv
import logging
import subprocess
from io import StringIO

# Device ID,Device Name,Vendor Name
cmd_discovery = "xpu-smi discovery --dump 1,2,3"
# Timestamp, DeviceId, GPU Utilization (%), GPU Memory Utilization (%), GPU Memory Used (MiB); N/A if read failed
cmd_dump = "xpu-smi dump -d X -m 0,5,18 -n 1"


def check_cmd(cmd):
    out = None
    try:
        out = subprocess.check_output(cmd, shell=True, timeout=5, text=True)
    except subprocess.TimeoutExpired:
        logging.error("Timeout running %s", cmd)
    except FileNotFoundError:
        logging.error("xpu-smi command not found. Cannot collect Intel GPU metrics.")
    except subprocess.CalledProcessError as e:
        logging.error("Error running %s: %s", cmd, e)

    buff = StringIO(out)
    reader = csv.reader(buff)
    reader = list(reader).copy()
    if len(reader[-1]) <= 1:
        reader = reader[:-1]
    for line in reader:
        for i in range(len(line)):
            line[i] = line[i].strip()

    return reader


def count_gpus():
    cmd_out = check_cmd(cmd_discovery)
    cnt = 0
    for line in cmd_out:
        if len(line) > 1:
            cnt += 1
    return cnt - 1


def list_gpu_info(num_gpus):
    if num_gpus == 0:
        return []
    gpus = ",".join([str(i) for i in range(num_gpus)])
    cmd_out = check_cmd(cmd_dump.replace("X", gpus))
    if len(cmd_out) == 0:
        raise Exception(
            "Error reading from {}. Please also check input.".format(cmd_dump)
        )
    else:
        return cmd_out
