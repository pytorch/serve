Following are the properties to be included in inference_data.json for adding a new testcase.

Mandatory properties
----
**url:** Model url.

**model_name:** Model name of the model url.

**worker:** Number workers of model.

**synchronous:** Synchronous call for register api, this must be true.

**file:** relative file location of input file

Response validation
----

Only http status code [200] gets verified to indicate test case success.

If you want to validate response body refer optional properties below.

Optional properties
----

Use these properties only if you want to validate response content/body.

**output_type:** Can be set to "text" or "json".
For text output type respone message is compared with expected String.
For json output type response json is compared with expected json object.

**json_type:**: Currently json_type can be either "default_json" or "image_classification".
"default_json" is a basic json comparator where key value pairs are expected to be constant, however the order may change.
"image_classification" is a custom comparator for json structure given below, here prediction scores may vary hence
tolerance is used while comparing scores.

**Note:**
If expected output from your model's inference request is json with a different structure then you will have to add a custom
 function with name validate_<new_json_type> and add entry for <new_json_type>:validate_<new_json_type> in `json_tests`
 objects in `inference_api_test_collection.json`.

**expected:** Expected string or json object based on output type.
**Note:**
At present binary types are not supported as expected type.

**tolerance:** Tolerence value in percentage, it is used to compare prediction scores in json object with a tolerance factor.

sample expected output for "image_classification" in json.
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
For above image classifiction inference response, here is the test case
```json
[{
        "url":"https://torchserve.s3.amazonaws.com/mar_files/squeezenet1_1.mar",
        "model_name":"squeezenet1_1",
        "worker":1,
        "synchronous":"true",
        "file":"../examples/image_classifier/kitten.jpg",
        "output_type":"json",
        "json_type": "image_classification",
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

In above example the tolerance is 1 which means - Probablity scores can be +/-1% of expected value.
eg:  if expected score of tabby is 0.25 then the accepted value can range from 0.2475
to 0.2525 (1% of 0.25 is 0.0025).



