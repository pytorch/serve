"""
Enums for the different metric types
"""
import enum


class MetricTypes(enum.Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
