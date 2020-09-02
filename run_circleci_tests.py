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

import subprocess
import sys
import copy
import argparse
import yaml


CCI_CONFIG_FILE = '.circleci/config.yml'
PROCESSED_FILE = '.circleci/processed.yml'
XFORMED_FILE = '.circleci/xformed.yml'
XFORMED_JOB_NAME = 'ts_xformed_job'


def create_processed_config(cci_config_file, processed_file):
    """ Create processed YAML using circleci cli's 'config process' command """
    process_config_cmd = 'circleci config process {} > {}'.format(cci_config_file, processed_file)
    print("Executing command : ", process_config_cmd)
    subprocess.check_call(process_config_cmd, shell=True)


def get_config(config_file):
    """ Read CircleCI config YAMLs as dictionaries """
    with open(config_file) as fstream:
        try:
            return yaml.safe_load(fstream)
        except yaml.YAMLError as err:
            print(err)


def get_available_executors(config):
    """ Returns list of all executors available in the config file """
    return list(config['executors'])


def get_all_jobs_in_workflow(processed_cfg, wrkflw):
    """ All jobs available under the specified workflow """
    return processed_cfg['workflows'][wrkflw]['jobs']


def get_processed_job_sequence(procsd_jb_name, all_jbs_in_wrkflw):
    """ Recursively iterate over jobs in the workflow to generate an ordered list of parent jobs """
    jobs_in_sequence = []

    job_dict = next((jd for jd in all_jbs_in_wrkflw \
                     if isinstance(jd, dict) and procsd_jb_name == list(jd)[0]), None)
    if job_dict:
        # Find all parent jobs, recurse to find their respective ancestors
        parent_jobs = job_dict[procsd_jb_name].get('requires', [])
        for pjob in parent_jobs:
            jobs_in_sequence += get_processed_job_sequence(pjob, all_jbs_in_wrkflw)

    return jobs_in_sequence + [procsd_jb_name]


def get_jobs_to_exec(job_name, all_jobs_in_wrkflw, executr, avlbl_executrs):
    """ Returns a dictionary of executors and a list of jobs to be executed in them  """
    jobs_dict = {}
    executors = [executr] if executr else avlbl_executrs

    for exectr_name in executors:
        if job_name is None:
            # List of all job names(as string)
            jobs_dict[exectr_name] = map(lambda j: j if isinstance(j, str) \
                                            else list(j)[0], all_jobs_in_wrkflw)
            # Filter processed job names as per the executor
            # "job_name-executor_name" is a convention set in config.yml
            # pylint: disable=cell-var-from-loop
            jobs_dict[exectr_name] = filter(lambda j: exectr_name in j, jobs_dict[exectr_name])
        else:
            # The list might contain duplicate parent jobs due to multiple fan-ins like config
            #     - Remove the duplicates
            # "job_name-executor_name" is a convention set in config.yml
            jobs_dict[exectr_name] = \
                OrderedDict.fromkeys(get_processed_job_sequence(job_name + '-' + exectr_name, \
                                                                all_jobs_in_wrkflw))
        jobs_dict[exectr_name] = list(jobs_dict[exectr_name])

    return jobs_dict


def get_merged_jobs_steps(jobs, processed_cfg):
    """ Merge and filter steps from all jobs """
    blocked_steps = ['persist_to_workspace', 'attach_workspace', 'store_artifacts']
    merged_steps = []
    for jname in jobs:
        steps = processed_cfg['jobs'][jname]['steps']
        merged_steps += list(filter(lambda step: list(step)[0] not in blocked_steps, steps))
    return merged_steps


def create_transformed_job(jobs, processed_cfg):
    """ Create a new transformed job which has all the steps merged from input jobs  """
    merged_steps = get_merged_jobs_steps(jobs, processed_cfg)

    # Create a new job, using the first job as a reference to ensure
    # This ensures configs like executor, environment, etc are maintained from the first job
    first_job = jobs[0]
    xformed_job = copy.deepcopy(processed_cfg['jobs'][first_job])

    # Add the merged steps to this newly introduced job
    xformed_job['steps'] = merged_steps

    return xformed_job


def create_transformed_config(xformed_jb, processed_cfg, xformed_jb_name, xformed_file):
    """ Create a duplicate config(transformed) with the
    newly introduced job as the only job in config """
    xformed_config = copy.deepcopy(processed_cfg)
    xformed_config['jobs'] = {}
    xformed_config['jobs'][xformed_jb_name] = xformed_jb

    # Create a transformed yaml
    with open(xformed_file, 'w+') as fstream:
        yaml.dump(xformed_config, fstream)


def execute_job(jbs_to_exec, processed_cfg, xformed_job_name, xformed_file):
    """ Create transformed job & config, use circleci cli's local execute
    to execute the transformed job """
    result_dict = {}
    for exectr, jobs in jbs_to_exec.items():
        xformed_job = create_transformed_job(jobs, processed_cfg)
        create_transformed_config(xformed_job, processed_cfg, xformed_job_name, xformed_file)

        try:
            # Locally execute the newly created job. This newly created job has all the steps
            # (ordered and merged from steps in parent job(s))
            local_execute_cmd = 'circleci local execute -c {} --job {} \
                                --env AWS_ACCESS_KEY_ID=`aws configure get aws_access_key_id` \
                                --env AWS_SECRET_ACCESS_KEY=`aws configure get aws_secret_access_key`' \
                .format(xformed_file, xformed_job_name)
            print('Executing command : ', local_execute_cmd)
            result_dict[exectr] = subprocess.check_call(local_execute_cmd, shell=True)
        except subprocess.CalledProcessError as err:
            result_dict[exectr] = err.returncode
    return result_dict


def print_result(res):
    """ Print job execution details """
    for exectr, retcode in res.items():
        colorcode, status = ('\033[0;37;42m', 'successful') if retcode == 0 \
            else ('\033[0;37;41m', 'failed')
        print("{} Job execution {} using {} executor \x1b[0m".format(colorcode, status, exectr))


def cleanup(processed_file, xformed_file):
    """ Clean up, remove the processed and transformed YAML files """
    cleanup_cmd = 'rm {} {}'.format(processed_file, xformed_file)
    print('Executing command : ', cleanup_cmd)
    subprocess.check_call(cleanup_cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Execute circleci jobs in a \
                                                    container on your local machine')
    parser.add_argument('workflow', type=str, help='Workflow name from config.yml')
    parser.add_argument('-j', '--job', type=str, help='Job name from config.yml')
    parser.add_argument('-e', '--executor', type=str, help='Executor name from config.yml')
    args = parser.parse_args()

    workflow = args.workflow
    job = args.job
    executor = args.executor

    create_processed_config(CCI_CONFIG_FILE, PROCESSED_FILE)
    cci_config = get_config(CCI_CONFIG_FILE)
    processed_config = get_config(PROCESSED_FILE)
    available_executors = get_available_executors(cci_config)
    all_jobs_in_workflow = get_all_jobs_in_workflow(processed_config, workflow)
    jobs_to_exec = get_jobs_to_exec(job, all_jobs_in_workflow, executor, available_executors)
    result = execute_job(jobs_to_exec, processed_config, XFORMED_JOB_NAME, XFORMED_FILE)
    print_result(result)
    cleanup(PROCESSED_FILE, XFORMED_FILE)

    # Exit as per overall status
    SYS_EXIT_CODE = 0 if all(retcode == 0 for exectr, retcode in result.items()) else 1
    sys.exit(SYS_EXIT_CODE)
