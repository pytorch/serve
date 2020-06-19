# TorchServe Regression Tests

This folder contains nightly regression tests execututed against TorchServe master.These tests use [POSTMAN](https://www.postman.com/downloads/) for exercising all the Management & Inference APIs.

### Latest Test Run Status

![Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiS1QvY3lIUEdUb3hZVWNnbmJ2SEZCdExRNmNkNW9EVk1ZaFNldEk4Q0h3TU1qemwzQ29GNW0xMGFhZkxpOFpiMjUrZVVRVDUrSkh2ZDhBeFprdW5iNjRRPSIsIml2UGFyYW1ldGVyU3BlYyI6IjlvcjRqSTNMTmNhcExZbUwiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master)

[Latest Test Logs](https://torchserve-regression-test.s3.amazonaws.com/public/latest/test_exec.log)

[Test Logs By Date](https://torchserve-regression-test.s3.amazonaws.com/public/TS-NIGHTLY-REGRESSION-2020-05-20/tmp/test_exec.log) - Update Date in URL

### Running the test manually.

Clone Torch Serve Repo & Build the Docker Image for the execition env.

```
git clone https://github.com/pytorch/serve
cd serve/docker
./build_image.sh
```

This would build a docker Image with a pytorch/torchserve:latest in which we would run our Regression Tests.

```
docker run -it --user root pytorch/torchserve:latest /bin/bash
```

In the Docker CLI execute the following cmds.

```
apt-get update 
apt-get install -y git wget sudo 
git clone https://github.com/pytorch/serve
cd serve 
```
To execute tests on master run: 

`./test/regression_tests.sh `

To execute tests on different run: 

`./test/regression_tests.sh <branch_name>`


You can view the logs for Test execution & the Torch serve in the /tmp dir.

```
cat /tmp/test_exec.log
cat /tmp/ts.log 
```

### Adding tests

To add to the tests, import a collection (in /postman) to Postman and add new requests.
Specifically to test for inference against a new model
* Open /postman/inference_data.json
* Add new json object with the new model url and payload.

![POSTMAN UI](screenshot/postman.png)

Afterwards, export the collection as a v2.1 collection and replace the existing exported collection.
To add a new suite of tests, add a new collection to /postman and update regression_tests.sh to run the new collection and buldsepc.yml to keep track of the report.
