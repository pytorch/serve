import os

from ts.context import Context
from ts.protocol.otf_message_handler import create_predict_response


def send_intermediate_predict_response(
    ret, req_id_map, message, code, context: Context
):
    if str(os.getenv("LOCAL_RANK", 0)) != "0":
        return None
    msg = create_predict_response(ret, req_id_map, message, code, context, True)
    context.cl_socket.sendall(msg)
