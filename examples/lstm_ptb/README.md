# Sequence to Sequence inference with LSTM network trained on PenTreeBank data set

In this example, we show how to create a service which generates sentences with a pre-trained LSTM model with deep model server. This model is trained on [PenTreeBank data](https://catalog.ldc.upenn.edu/ldc99t42) and training detail can be found in [MXNet example](https://github.com/apache/incubator-mxnet/tree/master/example/rnn).

This model uses [MXNet Bucketing Module](https://mxnet.incubator.apache.org/how_to/bucketing.html) to deal with variable length input sentences and generates output sentences with the same length as inputs.

# Step by step to create service

## Step 1 - Download the pre-trained LSTM model files, signature file and vocabulary dictionary file

```bash
cd mxnet-model-server/examples/lstm_ptb

curl -O https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb-symbol.json
curl -O https://s3.amazonaws.com/model-server/models/lstm_ptb/lstm_ptb-0100.params
curl -O https://s3.amazonaws.com/model-server/models/lstm_ptb/vocab_dict.txt
curl -O https://s3.amazonaws.com/model-server/models/lstm_ptb/signature.json
```

## Step 2 - Verify signature file

In this example, provided mxnet_vision_service.py template assume there is a `signature.json` file that describes input parameter and shape.

After [Step 1](#step-1---download-the-pre-trained-lstm-model-files,-signature-file-and-vocabulary-dictionary-file) there should be a signature file in the lstm_ptb folder. Verify that this file exists before proceeding further.

The signature file looks as follows.

```json
{
  "inputs": [
    {
      "data_name": "data",
      "data_shape": [
        1,
        60
      ],
     ...
    }
  ]
}
```
Input data shape is (1, 60). For sequence to sequence models, the inputs can be variable length sequences. In the signature file the input shape should be set to the maximum length of the input sequence, which is the default bucket key. The bucket sizes are defined when training the model. In this example valid bucket sizes are 10, 20, 30, 40, 50 and 60. Default bucket key is the maximum value which is 60. 
Check [bucketing module tutorials](https://mxnet.incubator.apache.org/faq/bucketing.html) if you want to know more about the bucketing module in MXNet.

## Step 3 - Check vocabulary dictionary file

[vocab_dict.txt](https://s3.amazonaws.com/model-server/models/lstm_ptb/vocab_dict.txt) is to store word to integer indexing information. In this example, each line in the text file represents a (word, index) pair. This file can be in different format and requires different customized parsing methods respectively.

## Step 4 - Create custom service class

We provide custom service class template code in [model_service_template](../model_service_template) folder:
1. [model_handler.py](../model_service_template/model_handler.py) - A generic based service class.
2. [mxnet_utils](../model_service_template/mxnet_utils) - A python package that contains utility classes.

```bash
cd mxnet-model-server/examples

cp model_service_template/model_handler.py lstm_ptb/
cp -r model_service_template/mxnet_utils lstm_ptb/
```

In this example, we need to implement `preprocess`, `inference` and `postprocess` methods in a custom service class. Implementation details are in [lstm_ptb_service.py](lstm_ptb_service.py).

## Step 5 - Package the model with `model-archiver` CLI utility

In this step, we package the following:
1. pre-trained MXNet Model we downloaded in Step 1.
2. '[signature.json](signature.json)' file we prepared in step 2.
3. '[vocab_dict.txt](vocab_dict.txt)' file we prepared in step 3.
4. custom model service files we prepared in step 4.

We use `model-archiver` command line utility (CLI) provided by MMS.
Install `model-archiver` in case you have not:

```bash
pip install model-archiver
```

This tool creates a .mar file that will be provided to MMS for serving inference requests. In following command line, we specify 'lstm_ptb_service:handle' as model archive entry point.

```bash
cd mxnet-model-server/examples
model-archiver --model-name lstm_ptb --model-path lstm_ptb --handler lstm_ptb_service:handle
```

## Step 6 - Start the Inference Service

Start the inference service by providing the 'lstm_ptb.mar' file we created in Step 5.

By default, the server is started on the localhost at port 8080.

```bash
cd mxnet-model-server

mxnet-model-server --start --model-store examples --models lstm_ptb.mar
```

## Test inference service

Now we can send post requests to the endpoint we just established.

Since the entire range of vocabularies in the training set is only 10,000, you may not get very good results with arbitrary test sentences. Instead, we recommend that you test with sentences from the [PTB test data set](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/ptb/ptb.test.txt). That being said, if you try some random text you should know that any word that isn't in that 10k vocabulary is encoded with an "invalid label" of 0. This will create a prediction result of '\n'. Note that in PTB data set, person name is represented by `<unk>`.

The key value of application/json input is 'input_sentence'. This can be a different value and preprocess method in lstm_ptb_service.py needs to be modified respectively. 

```bash
curl -X POST http://127.0.0.1:8080/predictions/lstm_ptb -H "Content-Type: application/json" -d '[{"input_sentence": "on the exchange floor as soon as ual stopped trading we <unk> for a panic said one top floor trader"}]'
```

Prediction result will be:

```json
{
  "prediction": "the <unk> 's the the as the 's the the 're to a <unk> <unk> <unk> analyst company trading at "
}
```

Let's try another sentence:

```bash
curl -X POST http://127.0.0.1:8080/predictions/lstm_ptb -H "Content-Type: application/json" -d '[{"input_sentence": "while friday '\''s debacle involved mainly professional traders rather than investors it left the market vulnerable to continued selling this morning traders said "}]'
```

Prediction result will be:

```json
{
  "prediction": "the 's stock were <unk> in <unk> say than <unk> were will to <unk> to to the <unk> the week \n \n \n \n \n \n \n \n \n \n "
}
```

References
1. [How to use MXNet bucketing module](https://mxnet.incubator.apache.org/how_to/bucketing.html)
2. [LSTM trained with PennTreeBank data set](https://github.com/apache/incubator-mxnet/tree/master/example/rnn)
