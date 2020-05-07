# TorchServe Regression Tests

This folder contains nightly regression tests execututed against TorchServe master.

These tests use [POSTMAN](https://www.postman.com/downloads/) for exercising all the Management & Inference APIs.

### Latest Test Run Status

![Build Status](https://codebuild.us-east-1.amazonaws.com/badges?uuid=eyJlbmNyeXB0ZWREYXRhIjoiS1QvY3lIUEdUb3hZVWNnbmJ2SEZCdExRNmNkNW9EVk1ZaFNldEk4Q0h3TU1qemwzQ29GNW0xMGFhZkxpOFpiMjUrZVVRVDUrSkh2ZDhBeFprdW5iNjRRPSIsIml2UGFyYW1ldGVyU3BlYyI6IjlvcjRqSTNMTmNhcExZbUwiLCJtYXRlcmlhbFNldFNlcmlhbCI6MX0%3D&branch=master)

[Test Logs](https://torchserve-regression-test.s3.amazonaws.com/torch-serve-regression-test/tmp/test_exec.log)

### Running the test manually.

Install npm, newman cli & html reporter
```
sudo apt-get install npm
npm install -g newman
npm install -g newman-reporter-html
```

Then, run `./test/regression_tests.sh` from the torchserve root. 


### Adding tests

To add to the tests, import a collection (in /postman) to Postman and add new requests. 
Afterwards, export the collection as a v2.1 collection and replace the existing exported collection.
To add a new suite of tests, add a new collection to /postman and update regression_tests.sh to run the new collection and buldsepc.yml to keep track of the report.
