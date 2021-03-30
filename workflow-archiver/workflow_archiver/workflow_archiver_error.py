
"""
Workflow Archiver Error
"""


class WorkflowArchiverError(Exception):
    """
    Error for Workflow Archiver module
    """
    def __init__(self, message):
        super().__init__(message)
