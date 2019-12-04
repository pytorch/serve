# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the 'License'). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the 'license' file accompanying this file. This file is
# distributed on an 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
from typing import Iterable  # noqa ignore=F401 imported but unused

import numpy as np
from six import BytesIO, StringIO

JSON = 'application/json'
CSV = "text/csv"
OCTET_STREAM = "application/octet-stream"
ANY = '*/*'
NPY = 'application/x-npy'
UTF8_TYPES = [JSON, CSV]


def _array_to_json(array_like):  # type: (np.array or Iterable or int or float) -> str
    """Convert an array-like object to JSON.

    To understand better what an array-like object is see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

    Args:
        array_like (np.array or Iterable or int or float): array-like object to be converted to JSON.

    Returns:
        (str): object serialized to JSON
    """

    def default(_array_like):
        if hasattr(_array_like, 'tolist'):
            return _array_like.tolist()
        return json.JSONEncoder().default(_array_like)

    return json.dumps(array_like, default=default)


def _array_to_npy(array_like):  # type: (np.array or Iterable or int or float) -> object
    """Convert an array-like object to the NPY format.

    To understand better what an array-like object is see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

    Args:
        array_like (np.array or Iterable or int or float): array-like object to be converted to NPY.

    Returns:
        (obj): NPY array.
    """
    buffer = BytesIO()
    np.save(buffer, array_like)
    return buffer.getvalue()


def _array_to_csv(array_like):  # type: (np.array or Iterable or int or float) -> str
    """Convert an array-like object to CSV.

    To understand better what an array-like object is see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

    Args:
        array_like (np.array or Iterable or int or float): array-like object to be converted to CSV.

    Returns:
        (str): object serialized to CSV
    """
    stream = StringIO()
    np.savetxt(stream, array_like, delimiter=',', fmt='%s')
    return stream.getvalue()


_encoder_map = {NPY: _array_to_npy,
                CSV: _array_to_csv,
                JSON: _array_to_json}


def encode(array_like, content_type):
    # type: (np.array or Iterable or int or float, str) -> np.array
    """Encode an array-like object in a specific content_type to a numpy array.

    To understand better what an array-like object is see:
    https://docs.scipy.org/doc/numpy/user/basics.creation.html#converting-python-array-like-objects-to-numpy-arrays

    Args:
        array_like (np.array or Iterable or int or float): to be converted to numpy.
        content_type (str): content type to be used.

    Returns:
        (np.array): object converted as numpy array.
    """
    try:
        encoder = _encoder_map[content_type]
        return encoder(array_like)
    except KeyError:
        raise Exception(content_type)
