Use following properties to add inference test case in inference_data.json

Mandatory properties
----
**url:** Model url.

**model_name:** Model name of the model url.

**worker:** Number workers of the model.

**synchronous:** Synchronous call for register api, this must be true.

**file:** Relative file location of the input file

Response validation
----

Only http status code [200] gets verified to indicate test case success.

If you want to validate the response body refer optional properties below.

Optional properties for response content validation
----

Following properties should be used only when you to want validate the response content/body.

**content-type:** Can be set to "text/plain" or "application/json".
For "text/plain", content-type response message is compared with expected string.
For "application/json", content-type response json is compared with expected json object.

**validator:**: Validator can be either "default_json" or "image_classification".
"default_json" is a basic json comparator where key value pairs are expected to be constant, however the order may change. By default, this will be used for content validation even if you don't specify any validator for content type json.

"image_classification" is a custom comparator for json structure given in following section. In this case, prediction scores may vary hence tolerance is used while comparing scores.

**Note:**
If expected output from your model's inference request is json with a different structure (compared to image classfication above] then you will have to add a custom comparator with name validate_<new_validator> and add entry for <new_validator>:validate_<new_validator> in `validators`
 objects in `inference_api_test_collection.json`.

**expected:** Expected string or json object based on content-type.

**Note:**
At present binary types are not supported as expected type.

**tolerance:** Tolerance value in percentage, it is used to compare prediction scores in json object with a tolerance factor.

Sample expected output for "image_classification" in json.
```json
[
            {
                "tabby":0.2752002477645874
            },
            {
                "lynx":0.2546876072883606
            },
            {
                "tiger_cat":0.24254210293293
            },
            {
                "Egyptian_cat":0.2213735282421112
            },
            {
                "cougar":0.0022544863168150187
            }
        ]
```
For above image classifiction inference response, here is the test case -
```json
[{
        "url":"https://torchserve.pytorch.org/mar_files/squeezenet1_1.mar",
        "model_name":"squeezenet1_1",
        "worker":1,
        "synchronous":"true",
        "file":"../examples/image_classifier/kitten.jpg",
        "content-type":"application/json",
        "validator": "image_classification",
        "expected":[
            {
                "tabby":0.2752002477645874
            },
            {
                "lynx":0.2546876072883606
            },
            {
                "tiger_cat":0.24254210293293
            },
            {
                "Egyptian_cat":0.2213735282421112
            },
            {
                "cougar":0.0022544863168150187
            }
        ],
        "tolerance":1
    }
]
```

In above example the tolerance is 1 which means — Probability scores can be +/-1% of expected value.
ex: if expected score of tabby is 0.25 then the accepted value can range from 0.2475
to 0.2525 (1% of 0.25 is 0.0025).
