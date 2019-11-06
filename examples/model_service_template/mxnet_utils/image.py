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
Image utils
"""
import base64
import sys
from io import BytesIO

import mxnet as mx
import numpy as np
from PIL import Image
from mxnet import image as img


def transform_shape(img_arr, dim_order='NCHW'):
    """
    Rearrange image NDArray shape to 'NCHW' or 'NHWC' which
    is valid for MXNet model input.
    Input image NDArray should has dim_order of 'HWC'.

    :param img_arr: NDArray
        Image in NDArray format with shape (channel, width, height)
    :param dim_order: str
        Output image dimension order. Valid values are 'NCHW' and 'NHWC'

    :return: NDArray
        Image in NDArray format with dim_order shape
    """
    assert dim_order in 'NCHW' or dim_order in 'NHWC', "dim_order must be 'NCHW' or 'NHWC'."
    if dim_order == 'NCHW':
        img_arr = mx.nd.transpose(img_arr, (2, 0, 1))
    output = mx.nd.expand_dims(img_arr, axis=0)
    return output


def read(buf, flag=1, to_rgb=True, out=None):
    """
    Read and decode an image to an NDArray.
    Input image NDArray should has dim_order of 'HWC'.

    Note: `imread` uses OpenCV (not the CV2 Python library).
    MXNet must have been built with USE_OPENCV=1 for `imdecode` to work.

    :param buf: str/bytes or numpy.ndarray
        Binary image data as string or numpy ndarray.
    :param flag:  {0, 1}, default 1
        1 for three channel color output. 0 for grayscale output.
    :param to_rgb:  bool, default True
        True for RGB formatted output (MXNet default).
        False for BGR formatted output (OpenCV default).
    :param out:  NDArray, optional
        Output buffer. Use `None` for automatic allocation.
    :return: NDArray
        An `NDArray` containing the image.

    Example
    -------
    >>> buf = open("flower.jpg", 'rb').read()
    >>> image.read(buf)
    <NDArray 224x224x3 @cpu(0)>
    """
    return img.imdecode(buf, flag, to_rgb, out)


def write(img_arr, flag=1, output_format='jpeg', dim_order='CHW'):
    """
    Write an NDArray to a base64 string.

    :param img_arr: NDArray
        Image in NDArray format with shape (channel, width, height).
    :param flag: {0, 1}, default 1
        1 for three channel color output. 0 for grayscale output.
    :param output_format: str
        Output image format.
    :param dim_order: str
        Input image dimension order. Valid values are 'CHW' and 'HWC'
    :return: str
        Image in base64 string format
    """
    assert dim_order in 'CHW' or dim_order in 'HWC', "dim_order must be 'CHW' or 'HWC'."
    if dim_order == 'CHW':
        img_arr = mx.nd.transpose(img_arr, (1, 2, 0))
    if flag == 1:
        mode = 'RGB'
    else:
        mode = 'L'
        img_arr = mx.nd.reshape(img_arr, (img_arr.shape[0], img_arr.shape[1]))
    img_arr = img_arr.astype(np.uint8).asnumpy()
    image = Image.fromarray(img_arr, mode)
    output = BytesIO()
    image.save(output, format=output_format)
    output.seek(0)
    if sys.version_info[0] < 3:
        return base64.b64encode(output.getvalue())
    else:
        return base64.b64encode(output.getvalue()).decode("utf-8")


def resize(src, new_width, new_height, interp=2):
    """
    Resizes image to new_width and new_height.
    Input image NDArray should has dim_order of 'HWC'.

    :param src: NDArray
        Source image in NDArray format
    :param new_width: int
        Width in pixel for resized image
    :param new_height: int
        Height in pixel for resized image
    :param interp: int
        interpolation method for all resizing operations

        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    :return: NDArray
        An `NDArray` containing the resized image.
    """
    return img.imresize(src, new_width, new_height, interp)


def fixed_crop(src, x0, y0, w, h, size=None, interp=2):
    """
    Crop src at fixed location, and (optionally) resize it to size.
    Input image NDArray should has dim_order of 'HWC'.

    :param src: NDArray
        Input image
    :param x0: int
        Left boundary of the cropping area
    :param y0 : int
        Top boundary of the cropping area
    :param w : int
        Width of the cropping area
    :param h : int
        Height of the cropping area
    :param size : tuple of (w, h)
        Optional, resize to new size after cropping
    :param interp : int, optional, default=2
        Interpolation method. See resize for details.
    :return: NDArray
        An `NDArray` containing the cropped image.
    """
    return img.fixed_crop(src, x0, y0, w, h, size, interp)


def color_normalize(src, mean, std=None):
    """
    Normalize src with mean and std.

    :param src : NDArray
        Input image
    :param mean : NDArray
        RGB mean to be subtracted
    :param std : NDArray
        RGB standard deviation to be divided
    :return: NDArray
        An `NDArray` containing the normalized image.
    """
    src = src.astype(np.float32)
    return img.color_normalize(src, mean, std)
