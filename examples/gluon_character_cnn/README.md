# Character-level CNN Model in Gluon trained using Amazon Product Dataset

In this example, we show how to create a service which classifies a review into product type using [Character-level Convolutional Network Model (CNN) model](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf) model by Yann LeCunn. This model is trained on [Amazon product data](http://jmcauley.ucsd.edu/data/amazon/) and training detail can be found in a detailed tutorial from Thomas Delteil on [Character CNN training.](https://github.com/ThomasDelteil/CNN_NLP_MXNet).


# Step by step to create service

## Step 1 - Download the Gluon Char CNN model file, model parameter and classification labels file to "/tmp/crepe"

```bash
# Create a model directory
mkdir /tmp/crepe

# Download the model file
curl -O https://raw.githubusercontent.com/awslabs/mxnet-model-server/master/examples/gluon_character_cnn/gluon_crepe.py

# Download the parameters
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/mms-char-cnn-files/crepe_gluon_epoch6.params

# Download classification labels file
curl -O https://s3.amazonaws.com/model-server/model_archive_1.0/examples/mms-char-cnn-files/synset.txt

# Move required files to the following folder
mv crepe_gluon_epoch6.params gluon_crepe.py synset.txt /tmp/crepe
```

## Step 2 - Look at the Gluon model/service  file

For Gluon models on MMS, the models are defined, within the MMS service file, the skeletal structure of the file looks like follows.

```python
class GluonCrepe(HybridBlock):
    """
    Hybrid Block gluon Crepe model
    """
    def __init__(self, classes=7, **kwargs):
      ## Define model below
      pass

class CharacterCNNService(object):
    """
    Gluon Character-level Convolution Service
    """

    def __init__(self):
        # The 69 characters as specified in the paper
        self.ALPHABET = list("abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}")
        # Map Alphabets to index
        self.ALPHABET_INDEX = {letter: index for index, letter in enumerate(self.ALPHABET)}
        # max-length in characters for one document
        self.FEATURE_LEN = 1014
        self.initialized = False

    def initialize(self, params):
        self.net = GluonCrepe()
        self.param_filename = "crepe_gluon_epoch6.params"
        self.model_name = params.manifest["model"]["modelName"]

        gpu_id = params.system_properties.get("gpu_id")
        model_dir = params.system_properties.get("model_dir")

        synset_file = os.path.join(model_dir, "synset.txt")
        param_file_path = os.path.join(model_dir, self.param_filename)
        if not os.path.isfile(param_file_path):
            raise OSError("Parameter file not found {}".format(param_file_path))
        if not os.path.isfile(synset_file):
            raise OSError("synset file not available {}".format(synset_file))

        self.ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)

        self.net.load_parameters(param_file_path, self.ctx)

        self.labels = [line.strip() for line in open(synset_file).readlines()]
        self.initialized = True
        self.net.hybridize(static_shape=True, static_alloc=True)

        # define preprocess, inference and postprocess methods
```

As shown, the Gluon model derives from the basic gluon hybrid block. Gluon hybrid blocks, provide performance of a symbolic model with a imperative model. More on Gluon, hybrid blocks [here](https://gluon.mxnet.io/chapter07_distributed-learning/hybridize.html).
The fully defined service file can be found under [gluon_crepe.py](gluon_crepe.py), we define `preprocess`, `inference`, `postprocess` methods in this file.

The input size is, limited to 1014, characters as mentioned in the paper. The output is of shape [0,7] as we classify the reviews into seven product categories. Both the input and output are passed on as 'application/json' based text content.

# Step 3 - Prepare synset.txt with list of class names

[synset.txt](synset.txt) is where we define list of all classes detected by the model. The pre-trained Character-level CNN model used in the example is trained to detect 7 classes including Books, CDs_and_Vinyl, Movies_and_TV and more. See synset.txt file for list of all classes.

The list of classes in synset.txt will be loaded by MMS as list of labels in inference logic.


## Step 4 - Export model files with mxnet-model-export CLI tool

With model file together with signature and  files in the model folder, we are ready to export them to MMS model file.

```bash
model-archiver --model-name crepe -f --model-path /tmp/crepe/ --handler gluon_crepe:crepe_inference --runtime python --export-path /tmp
```

A packaged model can be downloaded from [here.](https://s3.amazonaws.com/model-server/model_archive_1.0/examples/mms-char-cnn-files/crepe.mar)

## Step 5 - Establish inference service

`crepe.mar` file is created by exporting model files. We also defined custom service under gluon_crepe.py. We are ready to establish the Character-level CNN inference service:

```bash
mxnet-model-server --models crepe.mar --model-store /tmp
```

The endpoint is on localhost and port 8080. You can change them by passing --host and --port when establishing the service.

## Test inference service

Now we can send post requests to the endpoint we just established.


The key values of application/json input are 'review_title', 'review'. This can be a different value or combined to a single input , to achieve this preprocess method in gluon_crepe.py needs to be modified.

Let's take up a movie, review

```bash
curl -X POST http://127.0.0.1:8080/predictions/crepe -F "data=[{\"review_title\":\"Inception is the best\",\"review\": \"great direction and story\"}]"
```
Prediction result will be:

```json
{
  "confidence": {
    "Clothing_Shoes_and_Jewelry": 0.004,
    "Home_and_Kitchen": 0.001,
    "Sports_and_Outdoors": 0.001,
    "CDs_and_Vinyl": 0.038,
    "Movies_and_TV": 0.59,
    "Cell_Phones_and_Accessories": 0.0,
    "Books": 0.362
  },
  "predicted": "Movies_and_TV"
 }
```

Let's try another review, this time for a music album.

```bash
curl -X POST http://127.0.0.1:8080/predictions/crepe -F "data=[{\"review_title\":\"fantastic quality\",\"review\": \"quality sound playback\"}]"
```

Prediction result will be:

```json
{
  "confidence": {
    "Clothing_Shoes_and_Jewelry": 0.028,
    "Home_and_Kitchen": 0.012,
    "Sports_and_Outdoors": 0.028,
    "CDs_and_Vinyl": 0.727,
    "Movies_and_TV": 0.118,
    "Cell_Phones_and_Accessories": 0.068,
    "Books": 0.015
  },
  "predicted": "CDs_and_Vinyl"
}
```

References
1. [Character-level CNN](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf)
2. [How to train Character-level CNN on gluon](https://github.com/ThomasDelteil/TextClassificationCNNs_MXNet)
3. [Web Demo of Character-level CNN on gluon](https://thomasdelteil.github.io/TextClassificationCNNs_MXNet/)
