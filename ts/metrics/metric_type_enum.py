"""
Enums for the different metric types
"""
import enum


class MetricTypes(enum.Enum):
    counter = "counter"
    gauge = "gauge"
    histogram = "histogram"
