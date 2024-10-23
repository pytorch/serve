"""
OTF Codec for functionality requiring importing torch
"""

import io
import json
import logging
import os
import struct
from builtins import bytearray, bytes

import torch

from ts.protocol.otf_message_handler import encode_response_headers
from ts.utils.util import deprecated


def create_predict_response(
    ret, req_id_map, message, code, context=None, ts_stream_next=False
):
    """
    Create inference response.

    :param context:
    :param ret:
    :param req_id_map:
    :param message:
    :param code:
    :return:
    """
    if str(os.getenv("LOCAL_RANK", 0)) != "0":
        return None

    msg = bytearray()
    msg += struct.pack("!i", code)

    buf = message.encode("utf-8")
    msg += struct.pack("!i", len(buf))
    msg += buf

    for idx in req_id_map:
        req_id = req_id_map.get(idx).encode("utf-8")
        msg += struct.pack("!i", len(req_id))
        msg += req_id

        if context is None:
            # Encoding Content-Type
            msg += struct.pack("!i", 0)  # content_type

            # Encoding the per prediction HTTP response code
            # status code and reason phrase set to none
            msg += struct.pack("!i", code)
            msg += struct.pack("!i", 0)  # No code phrase is returned
            # Response headers none
            msg += struct.pack("!i", 0)
        else:
            if ts_stream_next is True:
                context.set_response_header(idx, "ts_stream_next", "true")
            elif context.stopping_criteria:
                is_stop = context.stopping_criteria[idx](ret[idx])
                if is_stop is not None:
                    ts_stream_next = "false" if is_stop else "true"
                    context.set_response_header(idx, "ts_stream_next", ts_stream_next)
            elif "true" == context.get_response_headers(idx).get("ts_stream_next"):
                context.set_response_header(idx, "ts_stream_next", "false")

            content_type = context.get_response_content_type(idx)
            if content_type is None or len(content_type) == 0:
                msg += struct.pack("!i", 0)  # content_type
            else:
                msg += struct.pack("!i", len(content_type))
                msg += content_type.encode("utf-8")

            sc, phrase = context.get_response_status(idx)
            http_code = sc if sc is not None else 200
            http_phrase = phrase if phrase is not None else ""

            msg += struct.pack("!i", http_code)
            msg += struct.pack("!i", len(http_phrase))
            msg += http_phrase.encode("utf-8")
            # Response headers
            msg += encode_response_headers(context.get_response_headers(idx))

        if ret is None:
            buf = b"error"
            msg += struct.pack("!i", len(buf))
            msg += buf
        else:
            val = ret[idx]
            # NOTE: Process bytes/bytearray case before processing the string case.
            if isinstance(val, (bytes, bytearray)):
                msg += struct.pack("!i", len(val))
                msg += val
            elif isinstance(val, str):
                buf = val.encode("utf-8")
                msg += struct.pack("!i", len(buf))
                msg += buf
            elif isinstance(val, torch.Tensor):
                buff = io.BytesIO()
                torch.save(val, buff)
                buff.seek(0)
                val_bytes = buff.read()
                msg += struct.pack("!i", len(val_bytes))
                msg += val_bytes
            else:
                try:
                    json_value = json.dumps(val, indent=2).encode("utf-8")
                    msg += struct.pack("!i", len(json_value))
                    msg += json_value
                except TypeError:
                    logging.warning("Unable to serialize model output.", exc_info=True)
                    return create_predict_response(
                        None, req_id_map, "Unsupported model output data type.", 503
                    )

    msg += struct.pack("!i", -1)  # End of list
    return msg


@deprecated(
    version=1.0,
    replacement="ts.handler_utils.utils.send_intermediate_predict_response",
)
def send_intermediate_predict_response(ret, req_id_map, message, code, context=None):
    if str(os.getenv("LOCAL_RANK", 0)) != "0":
        return None
    msg = create_predict_response(ret, req_id_map, message, code, context, True)
    context.cl_socket.sendall(msg)
