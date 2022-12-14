
"""
Module to define Unit mappings
"""


class Units(object):
    """
    Define a unit of elements
    """

    def __init__(self):
        self.units = {
            'ms': "Milliseconds",
            's': 'Seconds',
            'percent': 'Percent',
            'count': 'Count',
            'MB': 'Megabytes',
            'GB': 'Gigabytes',
            'kB': 'Kilobytes',
            'B': 'Bytes',
            '': 'unit',
            None: 'unit',
        }
