

"""
This module does the following:
a. Starts model-server.
b. Creates end-points based on the configured models.
c. Exposes standard "ping" and "api-description" endpoints.
d. Waits for servicing inference requests.
"""
from . import version

__version__ = version.__version__
