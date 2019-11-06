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
Dimension class for model server metrics
"""


class Dimension(object):
    """
    Dimension class defining key value pair
    """
    def __init__(self, name, value):
        """
        Constructor for Dimension class

        Parameters
        ----------
        name: str
            NAme of dimension
        value : str
           Unique Value of dimension
        """
        self.name = name
        self.value = value

    def __str__(self):
        """
        Return a string value
        :return:
        """
        return "{}:{}".format(self.name, self.value)

    def to_dict(self):
        """
        return an dictionary
        """
        return {'Name': self.name, 'Value': self.value}
