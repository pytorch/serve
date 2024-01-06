import importlib


def import_class(class_name: str, module_prefix=None):
    module_name = ""
    arr = class_name.rsplit(".", maxsplit=1)
    if len(arr) == 2:
        module_name, class_name = arr
    else:
        class_name = arr[0]

    if module_prefix is not None:
        module = importlib.import_module(f"{module_prefix}.{module_name}")
    else:
        module = importlib.import_module(module_name)

    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ImportError(f"class:{class_name} not found in module:{module_name}.")
    return model_class
