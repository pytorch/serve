import numpy as np
from tensorboard_plugin_wit._vendor.tensorflow_serving.apis import classification_pb2
from tensorboard_plugin_wit._vendor.tensorflow_serving.apis import inference_pb2
from tensorboard_plugin_wit._vendor.tensorflow_serving.apis import regression_pb2



# An example to receive the result from iris_handler.py
# Note that the file name `custom_wit_predict_fn.py` and the function name
# "custom_predict_fn" and the number of parameter should be exact the same.
# And this file should be put in the same folder at where you launched your
# local TensorBoard server.

def custom_predict_fn(examples, serving_bundle):
  """A custom function that is consumed by the What-If-Tool. This function is
  responsible for decoding the protocal buffer formatted inputs `examples` and
  returning the result based on the input.
  The values that users filled in the WIT setting page can be accessed in 
  serving_bundle's attributes, which includes inference_address, model_name,
  model_version. They can be used to compile the URL for your inference server.
  Args:
    examples: A list of examples that matches the model spec.
    serving_bundle: A `ServingBundle` object that contains the information to
      make the serving request.

  Returns:
    A ClassificationResponse or RegressionResponse proto.
  """

  if len(examples) == 0:
    return
  import pycurl
  from urllib.parse import urlencode

  crl = pycurl.Curl() 

  # The API for torchserve
  crl.setopt(crl.URL, serving_bundle.inference_address+"/predictions/"+serving_bundle.model_name)

  # Convert examples to list of lists
  examples_lol = []

  # May need some experience of TensorFlow TFRecord and tf.Example here:
  id_to_name = {0:"setosa", 1:"versicolor", 2:"virginica"}
  for example in examples:
    x0 = example.features.feature["sepal_length"].float_list.value[0]
    x1 = example.features.feature["sepal_width"].float_list.value[0]
    x2 = example.features.feature["petal_length"].float_list.value[0]
    x3 = example.features.feature["petal_width"].float_list.value[0]
    target = example.features.feature["species"].bytes_list.value[0]  # not used
    examples_lol.append([x0, x1, x2, x3])

  # Encode the data in JSON format and send request to torchserve
  # Since the example in iris_handler.py has: json.loads(data[0]["input"]
  # So you should use "input" as the key.
  data = {'input': examples_lol}
  pf = urlencode(data)
  crl.setopt(crl.POSTFIELDS, pf)

  response = crl.perform_rs() 
  import json
  result_for_each_example = json.loads(response)

  crl.close()

  # Now you have the result_for_each_example returned from the server,
  # encode it in classification_pb2 protobuf format to make WIT happy.
  # Also there is a regression_pb2 format can be used.

  # The following code extracts the score and class ID from the JSON
  # response and fills the "label" and "score" in the protofuf. You have to
  # extract these two values according to how you implement your server's
  # response.
  # In iris_handler.py, I added torch.exp() to convert the log_softmax 
  # score to probability before returning the result. The conversion
  # can be made in this file too.

  # p.s. If Each label's score for an given input is smaller than 0.5,
  # WIT will ignore the assigned label and use the label "0". (Bug or Feature? :))

  classifications_for_all_examples = []
  for result in result_for_each_example:
    classifications_for_single_example = []
    for score, clsid in zip(*result):  # score for each class
      classifications_for_single_example.append(classification_pb2.Class(label=id_to_name[clsid], score=score))
    classifications_for_all_examples.append(classification_pb2.Classifications(classes=classifications_for_single_example))
  classification_results = classification_pb2.ClassificationResult(classifications=classifications_for_all_examples)
  result = classification_pb2.ClassificationResponse(result=classification_results)
  return result

