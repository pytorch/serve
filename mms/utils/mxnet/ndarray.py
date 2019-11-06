# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
NDArray utils
"""
import numpy as np
import mxnet as mx


def top_probability(data, labels, top=5):
    """Get top probability prediction from NDArray.

    Parameters
    ----------
    data : NDArray
        Data to be predicted
    labels : List
        List of class labels

    Returns
    -------
    List
        List of probability: class pairs in sorted order
    """
    dim = len(data.shape)
    if dim > 2:
        data = mx.nd.array(
            np.squeeze(data.asnumpy(), axis=tuple(range(dim)[2:])))
    sorted_prob = mx.nd.argsort(data[0], is_ascend=False)
    # pylint: disable=deprecated-lambda
    top_prob = map(lambda x: int(x.asscalar()), sorted_prob[0:top])
    return [{'probability': float(data[0, i].asscalar()), 'class': labels[i]}
            for i in top_prob]
