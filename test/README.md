# TorchServe API Tests

## Adding tests

The tests are Postman collections which are ran via Newman with AWS CodeBuild. 
To add to the tests, import a collection (in /postman) to Postman and add new requests. 
Afterwards, export the collection as a v2.1 collection and replace the existing exported collection.

To add a new suite of tests, add a new collection to /postman and update regression_tests.sh to run the new collection and buldsepc.yml to keep track of the report.