# pylint: disable=missing-docstring
import json


class Workflow(object):
    """
    Workflow is a part of the manifest.json. It defines the properties of the workflow such as name, handler file etc.
    """

    def __init__(self, workflow_name, spec_file, handler):

        self.workflow_name = workflow_name
        self.spec_file = spec_file.split("/")[-1]
        self.handler = handler.split("/")[-1]

        self.workflow_dict = self.__to_dict__()

    def __to_dict__(self):
        workflow_dict = dict()

        workflow_dict['workflowName'] = self.workflow_name

        workflow_dict['specFile'] = self.spec_file

        workflow_dict['handler'] = self.handler

        return workflow_dict

    def __str__(self):
        return json.dumps(self.workflow_dict)

    def __repr__(self):
        return json.dumps(self.workflow_dict)
