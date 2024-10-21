"""
OTF Codec
"""

import json
import logging
import os
import struct
import sys
import time
from builtins import bytearray

bool_size = 1
int_size = 4
END_OF_LIST = -1
LOAD_MSG = b"L"
PREDICT_MSG = b"I"
RESPONSE = 3


def retrieve_msg(conn):
    """
    Retrieve a message from the socket channel.

    :param conn:
    :return:
    """
    cmd = _retrieve_buffer(conn, 1)
    if cmd == LOAD_MSG:
        msg = _retrieve_load_msg(conn)
    elif cmd == PREDICT_MSG:
        msg = _retrieve_inference_msg(conn)
        logging.info("Backend received inference at: %d", time.time())
    else:
        raise ValueError("Invalid command: {}".format(cmd))

    return cmd, msg


def encode_response_headers(resp_hdr_map):
    msg = bytearray()
    msg += struct.pack("!i", len(resp_hdr_map))
    for k, v in resp_hdr_map.items():
        msg += struct.pack("!i", len(k.encode("utf-8")))
        msg += k.encode("utf-8")
        msg += struct.pack("!i", len(v.encode("utf-8")))
        msg += v.encode("utf-8")
    return msg


def create_load_model_response(code, message):
    """
    Create load model response.

    :param code:
    :param message:
    :return:
    """
    msg = bytearray()
    msg += struct.pack("!i", code)

    buf = message.encode("utf-8")
    msg += struct.pack("!i", len(buf))
    msg += buf
    msg += struct.pack("!i", -1)  # no predictions

    return msg


def _retrieve_buffer(conn, length):
    data = bytearray()

    while length > 0:
        pkt = conn.recv(length)
        if len(pkt) == 0:
            logging.info("Frontend disconnected.")
            sys.exit(0)

        data += pkt
        length -= len(pkt)

    return data


def _retrieve_int(conn):
    data = _retrieve_buffer(conn, int_size)
    return struct.unpack("!i", data)[0]


def _retrieve_bool(conn):
    data = _retrieve_buffer(conn, bool_size)
    return struct.unpack("!?", data)[0]


def _retrieve_load_msg(conn):
    """
    MSG Frame Format:

    | cmd value |
    | int model-name length | model-name value |
    | int model-path length | model-path value |
    | int batch-size length |
    | int handler length | handler value |
    | int gpu id |
    | bool limitMaxImagePixels |

    :param conn:
    :return:
    """
    msg = {}
    length = _retrieve_int(conn)
    msg["modelName"] = _retrieve_buffer(conn, length)
    length = _retrieve_int(conn)
    msg["modelPath"] = _retrieve_buffer(conn, length)
    msg["batchSize"] = _retrieve_int(conn)
    length = _retrieve_int(conn)
    msg["handler"] = _retrieve_buffer(conn, length)
    gpu_id = _retrieve_int(conn)
    if gpu_id >= 0:
        msg["gpu"] = gpu_id

    length = _retrieve_int(conn)
    msg["envelope"] = _retrieve_buffer(conn, length)
    msg["limitMaxImagePixels"] = _retrieve_bool(conn)

    return msg


def _retrieve_inference_msg(conn):
    """
    MSG Frame Format:

    | cmd value |
    | batch: list of requests |
    """
    msg = []
    while True:
        request = _retrieve_request(conn)
        if request is None:
            break

        msg.append(request)

    return msg


def _retrieve_request(conn):
    """
    MSG Frame Format:

    | request_id |
    | request_headers: list of request headers|
    | parameters: list of request parameters |
    """
    length = _retrieve_int(conn)
    if length == -1:
        return None

    request = {}
    request["requestId"] = _retrieve_buffer(conn, length)

    headers = []
    while True:
        header = _retrieve_reqest_header(conn)
        if header is None:
            break
        headers.append(header)

    request["headers"] = headers

    model_inputs = []
    while True:
        input_data = _retrieve_input_data(conn)
        if input_data is None:
            break
        model_inputs.append(input_data)

    request["parameters"] = model_inputs
    return request


def _retrieve_reqest_header(conn):
    """
    MSG Frame Format:

    | parameter_name |
    | content_type |
    | input data in bytes |
    """
    length = _retrieve_int(conn)
    if length == -1:
        return None

    header = {}
    header["name"] = _retrieve_buffer(conn, length)

    length = _retrieve_int(conn)
    header["value"] = _retrieve_buffer(conn, length)

    return header


def _retrieve_input_data(conn):
    """
    MSG Frame Format:

    | parameter_name |
    | content_type |
    | input data in bytes |
    """
    decode_req = os.environ.get("TS_DECODE_INPUT_REQUEST")
    length = _retrieve_int(conn)
    if length == -1:
        return None

    model_input = {}
    model_input["name"] = _retrieve_buffer(conn, length).decode("utf-8")

    length = _retrieve_int(conn)
    content_type = _retrieve_buffer(conn, length).decode("utf-8")
    model_input["contentType"] = content_type

    length = _retrieve_int(conn)
    value = _retrieve_buffer(conn, length)
    if content_type == "application/json" and (
        decode_req is None or decode_req == "true"
    ):
        try:
            model_input["value"] = json.loads(value.decode("utf-8"))

        except Exception as e:
            model_input["value"] = value
            logging.warning(
                "Failed json decoding of input data. Forwarding encoded payload",
                exc_info=True,
            )

    elif content_type.startswith("text") and (
        decode_req is None or decode_req == "true"
    ):
        try:
            model_input["value"] = value.decode("utf-8")

        except Exception as e:
            model_input["value"] = value
            logging.warning(
                "Failed utf-8 decoding of input data. Forwarding encoded payload",
                exc_info=True,
            )

    else:
        model_input["value"] = value

    return model_input
