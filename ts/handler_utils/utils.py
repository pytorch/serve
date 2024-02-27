import importlib
import os

from ts.context import Context
from ts.protocol.otf_message_handler import create_predict_response


def import_class(class_name: str, module_prefix=None):
    if not class_name:
        raise ImportError(f"class name is not defined")

    module_name = ""
    arr = class_name.rsplit(".", maxsplit=1)
    if len(arr) == 2:
        module_name, class_name = arr
    else:
        class_name = arr[0]

    if module_prefix:
        module = (
            importlib.import_module(f"{module_prefix}.{module_name}")
            if len(module_name) > 0
            else importlib.import_module(module_prefix)
        )
    elif len(module_name) > 0:
        module = importlib.import_module(module_name)
    else:
        raise ImportError(f"module name is not defined.")

    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ImportError(f"class:{class_name} not found in module:{module_name}.")
    return model_class


def send_intermediate_predict_response(
    ret, req_id_map, message, code, context: Context
):
    if str(os.getenv("LOCAL_RANK", 0)) != "0":
        return None
    msg = create_predict_response(ret, req_id_map, message, code, context, True)
    context.cl_socket.sendall(msg)
