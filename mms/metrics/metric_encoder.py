# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

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
