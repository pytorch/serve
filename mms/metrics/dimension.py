

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
