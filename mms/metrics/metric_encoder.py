

"""
Metric Encoder class for json dumps
"""

import json
from json import JSONEncoder

from mms.metrics.dimension import Dimension
from mms.metrics.metric import Metric


class MetricEncoder(JSONEncoder):
    """
    Encoder class for json encoding Metric Object
    """
    def default(self, obj):  # pylint: disable=arguments-differ, method-hidden
        """
        Override only when object is of type Metric
        """
        if isinstance(obj, (Metric, Dimension)):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)
