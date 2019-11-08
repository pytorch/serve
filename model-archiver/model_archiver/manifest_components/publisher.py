

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
