import numpy as np
from tensorboard_plugin_wit._vendor.tensorflow_serving.apis import classification_pb2
from tensorboard_plugin_wit._vendor.tensorflow_serving.apis import inference_pb2
from tensorboard_plugin_wit._vendor.tensorflow_serving.apis import regression_pb2



def custom_predict_fn(examples, serving_bundle):
  """Send an RPC request to the Servomatic prediction service.

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
  from io import BytesIO 

  b_obj = BytesIO() 
  crl = pycurl.Curl() 
  # Set URL value
  crl.setopt(crl.URL, serving_bundle.inference_address+"/predictions/"+serving_bundle.model_name)

  # convert examples to list of lists
  examples_lol = []
  # [[0, 1, 0], [1.1, 2.0, -1]]

  name_to_id = {"setosa":0, "versicolor":1, "virginica":2}
  id_to_name = {0:"setosa", 1:"versicolor", 2:"virginica"}
  for example in examples:
    x0 = example.features.feature["sepal_length"].float_list.value[0]
    x1 = example.features.feature["sepal_width"].float_list.value[0]
    x2 = example.features.feature["petal_length"].float_list.value[0]
    x3 = example.features.feature["petal_width"].float_list.value[0]
    target = example.features.feature["species"].bytes_list.value[0]
    examples_lol.append([x0, x1, x2, x3])
  # print(examples_lol)
  data = {'input': examples_lol}
  pf = urlencode(data)
  crl.setopt(crl.POSTFIELDS, pf)

  response = crl.perform_rs() 
  # print(response)
  import json
  result_for_each_example = json.loads(response)
  # print(crl.getinfo(crl.HTTP_CODE))

  crl.close()

  classifications_for_all_examples = []
  for result in result_for_each_example:
    classifications_for_single_example = []
    for score, clsid in zip(*result):
      classifications_for_single_example.append(classification_pb2.Class(label=str(clsid), score=score))
    classifications_for_all_examples.append(classification_pb2.Classifications(classes=classifications_for_single_example))
  classification_results = classification_pb2.ClassificationResult(classifications=classifications_for_all_examples)
  result = classification_pb2.ClassificationResponse(result=classification_results)
  return result

