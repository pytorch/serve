# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# pylint: disable=missing-docstring
import json


class Publisher(object):
    """
    Publisher object is a part of Manifest.json
    """

    def __init__(self, author, email):
        self.author = author
        self.email = email
        self.pub_dict = self.__to_dict__()

    def __to_dict__(self):
        pub_dict = dict()
        pub_dict['author'] = self.author
        pub_dict['email'] = self.email

        return pub_dict

    def __str__(self):
        return json.dumps(self.pub_dict)

    def __repr__(self):
        return json.dumps(self.pub_dict)
