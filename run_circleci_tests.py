#!/usr/bin/env python
"""
- This script helps to execute circleci jobs in a container on developer's local machine.
- The script accepts workflow(mandatory), job(optional) and executor(optional) arguments.
- The script used circleci cli's process command to generate a processed yaml.
- The processed yaml, is parsed and tweaked to generate a new transformed yaml.
- The transformed yaml contains a single job, which is merged and ordered list of job steps
from the specified and required parent jobs.
"""

# Make sure you have following dependencies installed on your local machine
# 1. PyYAML (pip install PyYaml)
# 2. CircleCI cli from - https://circleci.com/docs/2.0/local-cli/#installation
# 3. docker

from collections import OrderedDict
from functools import reduce

import subprocess
import sys
import copy
import argparse
import yaml

parser = argparse.ArgumentParser(description='Execute circleci jobs in a container \
                                                on your local machine')
parser.add_argument('workflow', type=str, help='Workflow name from config.yml')
parser.add_argument('-j', '--job', type=str, help='Job name from config.yml')
parser.add_argument('-e', '--executor', type=str, help='Executor name from config.yml')
args = parser.parse_args()

workflow = args.workflow
job = args.job
executor = args.executor

CCI_CONFIG_FILE = '.circleci/config.yml'
PROCESSED_FILE = '.circleci/processed.yml'
XFORMED_FILE = '.circleci/xformed.yml'
CCI_CONFIG = {}
PROCESSED_CONFIG = {}
XFORMED_CONFIG = {}
XFORMED_JOB_NAME = 'ts_xformed_job'
BLACKLISTED_STEPS = ['persist_to_workspace', 'attach_workspace', 'store_artifacts']

# Read CircleCI's config
with open(CCI_CONFIG_FILE) as fstream:
    try:
        CCI_CONFIG = yaml.safe_load(fstream)
    except yaml.YAMLError as err:
        print(err)

# Create processed YAML using circleci cli's 'config process' commands
PROCESS_CONFIG_CMD = 'circleci config process {} > {}'.format(CCI_CONFIG_FILE, PROCESSED_FILE)
print("Executing command : ", PROCESS_CONFIG_CMD)
subprocess.check_call(PROCESS_CONFIG_CMD, shell=True)

# Read the processed config
with open(PROCESSED_FILE) as fstream:
    try:
        PROCESSED_CONFIG = yaml.safe_load(fstream)
    except yaml.YAMLError as err:
        print(err)

# All executors available in the config file
available_executors = list(CCI_CONFIG['executors'])

# All jobs available under the specified workflow
jobs_in_workflow = PROCESSED_CONFIG['workflows'][workflow]['jobs']


def get_processed_job_sequence(processed_job_name):
    """ Recursively iterate over jobs in the workflow to generate an ordered list of parent jobs """
    jobs_in_sequence = []

    job_dict = next((jd for jd in jobs_in_workflow \
                    if isinstance(jd, dict) and processed_job_name == list(jd)[0]), None)
    if job_dict:
        # Find all parent jobs, recurse to find their respective ancestors
        parent_jobs = job_dict[processed_job_name].get('requires', [])
        for pjob in parent_jobs:
            jobs_in_sequence += get_processed_job_sequence(pjob)

    return jobs_in_sequence + [processed_job_name]


def get_jobs_to_exec(job_name):
    """ Returns a dictionary of executors and a list of jobs to be executed in them  """
    jobs_dict = {}
    executors = [executor] if executor else available_executors

    for exectr_name in executors:
        if job_name is None:
            # List of all job names(as string)
            jobs_dict[exectr_name] = map(lambda j: j if isinstance(j, str) \
                else list(j)[0], jobs_in_workflow)
            # Filter processed job names as per the executor
            # "job_name-executor_name" is a convention set in config.yml
            # pylint: disable=cell-var-from-loop
            jobs_dict[exectr_name] = filter(lambda j: exectr_name in j, jobs_dict[exectr_name])
        else:
            # The list might contain duplicate parent jobs due to multiple fan-ins like config
            #     - Remove the duplicates
            # "job_name-executor_name" is a convention set in config.yml
            jobs_dict[exectr_name] = \
                OrderedDict.fromkeys(get_processed_job_sequence(job_name + '-' + exectr_name))
        jobs_dict[exectr_name] = list(jobs_dict[exectr_name])

    return jobs_dict


# jobs_to_exec is a dict, with executor(s) as the key and list of jobs to be executed as its value
jobs_to_exec = get_jobs_to_exec(job)


def get_jobs_steps(steps, job_name):
    """ Merge all the steps from list of jobs to execute """
    job_steps = PROCESSED_CONFIG['jobs'][job_name]['steps']
    filtered_job_steps = list(filter(lambda step: list(step)[0] not in BLACKLISTED_STEPS, \
                                     job_steps))
    return steps + filtered_job_steps


result_dict = {}

for exectr, jobs in jobs_to_exec.items():
    merged_steps = reduce(get_jobs_steps, jobs, [])

    # Create a new job, using the first job as a reference
    # This ensures configs like executor, environment, etc are maintained from the first job
    first_job = jobs[0]
    xformed_job = copy.deepcopy(PROCESSED_CONFIG['jobs'][first_job])

    # Add the merged steps to this newly introduced job
    xformed_job['steps'] = merged_steps

    # Create a duplicate config(transformed) with the newly introduced job as the only job in config
    XFORMED_CONFIG = copy.deepcopy(PROCESSED_CONFIG)
    XFORMED_CONFIG['jobs'] = {}
    XFORMED_CONFIG['jobs'][XFORMED_JOB_NAME] = xformed_job

    # Create a transformed yaml
    with open(XFORMED_FILE, 'w+') as fstream:
        yaml.dump(XFORMED_CONFIG, fstream)

    try:
        # Locally execute the newly created job
        # This newly created job has all the steps (ordered and merged from steps in parent job(s))
        LOCAL_EXECUTE_CMD = 'circleci local execute -c {} --job {}'.format(XFORMED_FILE, \
                                                                           XFORMED_JOB_NAME)
        print('Executing command : ', LOCAL_EXECUTE_CMD)
        result_dict[exectr] = subprocess.check_call(LOCAL_EXECUTE_CMD, shell=True)
    except subprocess.CalledProcessError as err:
        result_dict[exectr] = err.returncode

# Clean up, remove the processed and transformed yml files
CLEANUP_CMD = 'rm {} {}'.format(PROCESSED_FILE, XFORMED_FILE)
print('Executing command : ', CLEANUP_CMD)
subprocess.check_call(CLEANUP_CMD, shell=True)

# Print job execution details
for exectr, retcode in result_dict.items():
    colorcode, status = ('\033[0;37;42m', 'successful') if retcode == 0 \
        else ('\033[0;37;41m', 'failed')
    print("{} Job execution {} using {} executor \x1b[0m".format(colorcode, status, exectr))

# Exit as per overall status
SYS_EXIT_CODE = 0 if all(retcode == 0 for exectr, retcode in result_dict.items()) else 1
sys.exit(SYS_EXIT_CODE)
