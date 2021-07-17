import numpy as np
import json
import pycurl
from urllib.parse import urlencode


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
    A list of list (for classification) or a list of numbers (for regression)
  """

  if len(examples) == 0:
    return
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
  # Since the example in iris_handler.py says: json.loads(data[0]["input"]
  # So you should use "input" as the key.
  data = {'input': examples_lol}
  pf = urlencode(data)
  crl.setopt(crl.POSTFIELDS, pf)

  response = crl.perform_rs()
  result_for_each_example = json.loads(response)
  # A list of list. Each inner list contains the probability of each class.

  crl.close()
  return result_for_each_example

