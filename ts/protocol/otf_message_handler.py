

"""
OTF Codec
"""
import json
import logging
import struct
import sys
import os
import io
from builtins import bytearray
from builtins import bytes
import time
import torch

bool_size = 1
int_size = 4
END_OF_LIST = -1
LOAD_MSG = b'L'
PREDICT_MSG = b'I'
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
    msg += struct.pack('!i', len(resp_hdr_map))
    for k, v in resp_hdr_map.items():
        msg += struct.pack('!i', len(k.encode('utf-8')))
        msg += k.encode('utf-8')
        msg += struct.pack('!i', len(v.encode('utf-8')))
        msg += v.encode('utf-8')
    return msg


def create_predict_response(ret, req_id_map, message, code, context=None):
    """
    Create inference response.

    :param context:
    :param ret:
    :param req_id_map:
    :param message:
    :param code:
    :return:
    """
    msg = bytearray()
    msg += struct.pack('!i', code)

    buf = message.encode("utf-8")
    msg += struct.pack('!i', len(buf))
    msg += buf

    for idx in req_id_map:
        req_id = req_id_map.get(idx).encode('utf-8')
        msg += struct.pack("!i", len(req_id))
        msg += req_id

        # Encoding Content-Type
        if context is None:
            msg += struct.pack('!i', 0)  # content_type
        else:
            content_type = context.get_response_content_type(idx)
            if content_type is None or len(content_type) == 0:
                msg += struct.pack('!i', 0)  # content_type
            else:
                msg += struct.pack('!i', len(content_type))
                msg += content_type.encode('utf-8')

        # Encoding the per prediction HTTP response code
        if context is None:
            # status code and reason phrase set to none
            msg += struct.pack('!i', code)
            msg += struct.pack('!i', 0)  # No code phrase is returned
            # Response headers none
            msg += struct.pack('!i', 0)
        else:
            sc, phrase = context.get_response_status(idx)
            http_code = sc if sc is not None else 200
            http_phrase = phrase if phrase is not None else ""

            msg += struct.pack('!i', http_code)
            msg += struct.pack("!i", len(http_phrase))
            msg += http_phrase.encode("utf-8")
            # Response headers
            msg += encode_response_headers(context.get_response_headers(idx))

        if ret is None:
            buf = b"error"
            msg += struct.pack('!i', len(buf))
            msg += buf
        else:
            val = ret[idx]
            # NOTE: Process bytes/bytearray case before processing the string case.
            if isinstance(val, (bytes, bytearray)):
                msg += struct.pack('!i', len(val))
                msg += val
            elif isinstance(val, str):
                buf = val.encode("utf-8")
                msg += struct.pack('!i', len(buf))
                msg += buf
            elif isinstance(val, torch.Tensor):
                buff = io.BytesIO()
                torch.save(val, buff)
                buff.seek(0)
                val_bytes = buff.read()
                msg += struct.pack('!i', len(val_bytes))
                msg += val_bytes
            else:
                try:
                    json_value = json.dumps(val, indent=2).encode("utf-8")
                    msg += struct.pack('!i', len(json_value))
                    msg += json_value
                except TypeError:
                    logging.warning("Unable to serialize model output.", exc_info=True)
                    return create_predict_response(None, req_id_map, "Unsupported model output data type.", 503)

    msg += struct.pack('!i', -1)  # End of list
    return msg


def create_load_model_response(code, message):
    """
    Create load model response.

    :param code:
    :param message:
    :return:
    """
    msg = bytearray()
    msg += struct.pack('!i', code)

    buf = message.encode("utf-8")
    msg += struct.pack('!i', len(buf))
    msg += buf
    msg += struct.pack('!i', -1)  # no predictions

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
    msg = dict()
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

    request = dict()
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

    header = dict()
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

    model_input = dict()
    model_input["name"] = _retrieve_buffer(conn, length).decode("utf-8")

    length = _retrieve_int(conn)
    content_type = _retrieve_buffer(conn, length).decode("utf-8")
    model_input["contentType"] = content_type

    length = _retrieve_int(conn)
    value = _retrieve_buffer(conn, length)
    if content_type == "application/json" and (decode_req is None or decode_req == "true"):
        model_input["value"] = json.loads(value.decode("utf-8"))
    elif content_type.startswith("text") and (decode_req is None or decode_req == "true"):
        model_input["value"] = value.decode("utf-8")
    else:
        model_input["value"] = value
    return model_input
