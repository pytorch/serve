"""
Util files for TorchServe
"""
import inspect


def list_classes_from_module(module, parent_class=None):
    """
    Parse user defined module to get all model service classes in it.

    :param module:
    :param parent_class:
    :return: List of model service class definitions
    """

    # Parsing the module to get all defined classes
    classes = [cls[1] for cls in inspect.getmembers(module, lambda member: inspect.isclass(member) and
                                                    member.__module__ == module.__name__)]
    # filter classes that is subclass of parent_class
    if parent_class is not None:
        return [c for c in classes if issubclass(c, parent_class)]

    return classes
