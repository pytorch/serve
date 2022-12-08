"""
Custom error class for Metrics Cache.
"""
import yaml


class GeneralMetricsCacheError(Exception):
    """
    Metrics Cache Error wrapper class to be utilized in Metric Cache classes.
    """

    def __init__(self, message):
        super().__init__(f"Generic error: {message}")


class MetricsCacheIOError(IOError):
    """
    IO Error for reading in file
    """

    def __init__(self, message):
        super().__init__(f"Error reading file: {message}")


class MetricsCachePyYamlError(yaml.YAMLError):
    """
    PyYaml error for parsing file
    """

    def __init__(self, message):
        super().__init__(f"Error parsing file: {message}")


class MetricsCacheTypeError(TypeError):
    """
    Type error wrapper
    """

    def __init__(self, message):
        super().__init__(f"{message}")


class MetricsCacheValueError(ValueError):
    """
    Value error wrapper
    """

    def __init__(self, message):
        super().__init__(f"{message}")


class MetricsCacheKeyError(KeyError):
    """
    Key error wrapper
    """

    def __init__(self, message):
        super().__init__(f"{message}")
