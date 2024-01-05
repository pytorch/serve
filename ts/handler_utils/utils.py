import importlib


def import_class(class_name: str, module_prefix=None):
    module_name, class_name = class_name.rsplit(".", maxsplit=1)
    if module_prefix is not None:
        module = importlib.import_module(f"{module_prefix}.{module_name}")
    model_class = getattr(module, class_name, None)
    if model_class is None:
        raise ImportError(f"{class_name} not found in {module_name}.")
    return model_class
