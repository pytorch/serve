### MultiModal (MMF) Framework

Multi modality learning helps the AI solutions to get signals from different input sources such as language, video, audio and combine their results to improve the inferences.


[MultiModal (MMF) framework](https://ai.facebook.com/blog/announcing-mmf-a-framework-for-multimodal-ai-models/)  is a modular deep learning framework for vision and language multimodal research. MMF provides  starter code for several multimodal challenges, including the Hateful Memes, VQA, TextVQA, and TextCaps challenges. You can learn more about MMF from their [website](https://mmf.readthedocs.io/en/latest/?=IwAR3P8zccSXqNt1XCCUv4Ysq0qkD515T6K9JnhUwpNcz0zzRl75FNSio9REU) a [Github](https://github.com/facebookresearch/mmf?fbclid=IwAR2OZi-8rQaxO3uwLxwvvvr9cuY8J6h0JP_g6BBM-qM7wpnNYEZEmWOQ6mc).


In the following, we first show how to serve the MMF model with Torchserve using a pre-trained MMF model for activity recognition, then, we will discuss the details of the custom handler and how to train your activity recognition model in MMF.

### Serving Activity Recognition MMF Model with Torchserve

This section, we have the trained MMF model for activity recognition, it can be served in production with [Torchserve](https://github.com/pytorch/serve).

To serve a model using Torchserve, we need to bundle the model artifacts and a handler into a mar file which is an archive format that torchserve uses to serve our model, model_archiver package does this step. The mar file will get extracted in a temp directory and the Path will be added to the PYTHONPATH.

#### Requirements

Install [Torchserve](https://github.com/pytorch/serve)


Install [MMF](https://mmf.sh/docs#install-from-source-recommended) from source or install latest MMF version from GitHub.

`pip install git+https://github.com/facebookresearch/mmf.git`

If you installed using pip, then you need install Pyav :

`pip install av`

MMF currently is using Transformers 3.4.0, in case you have other version installed in your environment, this would be the best instead of installing it directly, add the MMF Github and 'av', in the requirements.txt and pass it to the model archiver using -r flag. You can read more about serving models with third party dependencies [here](https://github.com/pytorch/serve/tree/master/docs/use_cases.md#serve-custom-models-with-third-party-dependency).

***Note, MMF currently does not support Pytorch 1.10, please make sure you are using Pytorch 1.9***

#### Getting started on Serving

To make the mar file, we will use model_archiver as follows.

After training the MMF model, the final checkpoints are saved in the mmf/save/ directory, where we need to use it as the serialized file for model-archiver. The [Charades action labels](https://mmfartifacts.s3-us-west-2.amazonaws.com/charades_action_lables.csv) along with the config file that covers all the settings of this experiment (can be found in mmf/save/config.yaml in case of training) are passed as extra files, they are getting used in the handler. Please make sure to change the label and config files accordingly in the handler (Lines #53, #68) if you are passing different ones.

**You can simply download the mar file as follows:**

`wget https://mmfartifacts.s3-us-west-2.amazonaws.com/MMF_activity_recognition.mar`

 If mar file is downloaded then skip this and move to the next step. The other option is to download a pre-trained model, along with labels and config for this example and package them to a mar file.

```
wget https://mmfartifacts.s3-us-west-2.amazonaws.com/mmf_transformer_Charades_final.pth
wget https://mmfartifacts.s3-us-west-2.amazonaws.com/charades_action_lables.csv
wget https://mmfartifacts.s3-us-west-2.amazonaws.com/config.yaml
```

```
torch-model-archiver --model-name MMF_activity_recognition --version 1.0 --serialized-file mmf_transformer_Charades_final.pth  --handler handler.py --extra-files "charades_action_lables.csv,config.yaml"
```

Running the above commands will result in MMF_activity_recognition mar in the current directory.

Note as MMF uses torch.cuda.current_device() to decide if inputs are on correct device, we used device context manager in the handler. This means you won't be able to set the number_of_gpu to zero in this example, basically to serve this example on cpu, you will need to run on a cpu instance or masking the cuda devices using export CUDA_VISIBLE_DEVICES="".

The **next step** is to make a model_store and move the .mar file to it:

```
mkdir model_store
mv MMF_activity_recognition.mar model_store
```

Now we can start serving our model:

```
torchserve --start --model-store model_store --disable-token-auth  --enable-model-api
curl -X POST "localhost:8081/models?model_name=MMF_activity_recognition&url=MMF_activity_recognition.mar&batch_size=1&max_batch_delay=5000&initial_workers=1&synchronous=true"
```

Sending inference request using

The examples of video and info.json is available here for demonstration purposes. **Please make sure to set the correct path to the video, info.json and model name in the request.py (e.g. $(path)/video.mp4, $(path)/video.info.json).**

```
wget https://mmfartifacts.s3-us-west-2.amazonaws.com/372CC.mp4

Python request.py
```

This will write the results in response.txt in the current directory.

#### Tochserve Custom Handler

For the activity recognition MMF model, we need to provide a custom handler. The handler generally extends the [Base handler](https://github.com/pytorch/serve/tree/master/ts/torch_handler/base_handler.py). The [handler](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition/handler.py) for MMF model, needs to load and initialize the model in the initialize method and then in the preprocess, mimics the logic in dataset processors to make a sample form the input video and its related text ( preprocess the video and make the related tensors to video, audio and text). The inference method runs the preprocessed samples through the  MMF model and sends the outputs to the post process method.

To initialize the MMF model in the [initialization method](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition/handler.py#L65), there are few points to consider. We need to load [config](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition/handler.py#L68)  using OmegaConfing, then [setup_very_basic_config](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition/handler.py#L70)  function from mmf.utils.logger and [setup_imports](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition/handler.py#L71)  from  mmf.utils.env need to be called to setup the environment for loading the model. Finally to load the [model](https://github.com/pytorch/serve/tree/master/examples/MMF-activity-recognition/handler.py#L72), we pass the model config to the  [MMFTransformer](https://github.com/facebookresearch/mmf/tree/master/mmf/models/mmf_transformer.py) model.


### Activity Recognition from Videos using MMF

We are going to present an example of activity recognition on Charades video dataset. Basically in this example three modalities will be used to classify the activity from the video, image, audio and text. Images are extracted frames from the video, and audio is extracted from the video and text is the captions related to frames in the video. In this case,  embedding for each of the modalities are captured and then [MMFTransformer](https://github.com/facebookresearch/mmf/tree/master/mmf/models/mmf_transformer.py) from the model zoo has been used for fusion of the embeddings.

 There are a number of steps based on the discussed concepts on MMF website:

1. Define a new dataset, "Charades" and registering the dataset builder with MMF.
2. Define a model for our training, which is MMFTransformer model in this case.
3. Set up configs for dataset, model, the experiment (configs for training job).

In the following we discuss each of the steps in more details.

#### New Dataset

In this example Charades dataset has been used which is a video dataset added in the [dataset zoo](https://github.com/facebookresearch/mmf/tree/master/mmf/datasets/builders/charades). We can define a new dataset in MMF by following this [guide](https://mmf.sh/docs/tutorials/dataset). **To add a new dataset**, we need to define a new dataset class which extends the Basedataset class from  mmf.datasets.base_dataset, where we need to override three methods, __init__, __getitem__ and __len__. These methods basically define how to initialize ( set the path to the dataset), get each item from the dataset and then provide the length of the dataset. The Charades dataset class can be found [here](https://github.com/facebookresearch/mmf/tree/master/mmf/datasets/builders/charades/dataset.py#L16).  Also, we are able to set the [processors](https://github.com/facebookresearch/mmf/tree/master/mmf/configs/datasets/charades/defaults.yaml#L22) in the dataset config file and initialize them in the dataset class.

The **next step** is to define a dataset builder class which extends the "BaseDatasetBuilder" class from mmf.datasets.base_dataset_builder. In this class essentially we need to override three methods, __init__, __build__ and __load__. Where in the __init __ method, the dataset class name is set (as we defined in the previous step), the __build__ method, is responsible for downloading the dataset and __load__ method is taking care of   loading the dataset, builds an object of class inheriting "BaseDataset" which contains your dataset logic and returns it. The dataset builder code is also available [here](https://github.com/facebookresearch/mmf/tree/master/mmf/datasets/builders/charades/builder.py).

**Final step** is to register the dataset builder with mmf, where we can use the registry function as decorator, such as @registry.register_builder("charades").

#### Model Definition

To train a multimodal model, we need to define a model that will take the features from our modalities (using modality encoders) as inputs and trains for the task in hand, where in this example is activity recognition. For activity recognition basically the task is classification on activity labels. In this example, [MMFTransformer](https://github.com/facebookresearch/mmf/tree/master/mmf/models/mmf_transformer.py) is used that extend [base transformers](https://github.com/facebookresearch/mmf/tree/master/mmf/models/transformers/base.py) from mmf.models.transformers.base model available in the MMF model zoo.

 Generally, any defined MMF model class requires to extend the BaseModel from mmf.models.base_model, where we need to pass the configs in the __init__ method and implement the build and forward method. Init method takes the related config and build method builds all the essential module used in the model including encoders.

#### Configurations

As indicated in [MMF docs](https://mmf.sh/docs/notes/configuration), there are separate config files for datasets, models and experiments, all the configs can be found in config directory. For this example, [dataset config](https://github.com/facebookresearch/mmf/tree/master/mmf/configs/datasets/charades/defaults.yaml) sets the path to different sets (train/val/test), and processors and their related parameters. Similarly, [model config](https://github.com/facebookresearch/mmf/tree/master/mmf/configs/models/mmf_transformer/defaults.yaml) can set the specifics for the model including in hierarchical mode , different modalities, the encoder used in each modality, the model head type, loss function and etc. [Experiment config](https://github.com/facebookresearch/mmf/tree/master/projects/mmf_transformer/configs/charades/direct.yaml) is where setting for experiment such as optimizer specifics, scheduler,  evaluation metrics, training parameters such as batch-size, number of iterations and so on can be configured.

#### Running the experiment

To train the activity recognition MMF model as we discussed, the [Charades dataset](https://github.com/facebookresearch/mmf/tree/master/mmf/datasets/builders/charades/dataset.py#L16) has been added to MMF and we use  [MMFTransformer](https://github.com/facebookresearch/mmf/tree/master/mmf/models/mmf_transformer.py#L34) model. [Config files](https://github.com/facebookresearch/mmf/tree/master/mmf/configs) for the model, dataset and experiment has been set as well. Now we need to run the following command to start training the model, the following command starts with downloading the datasets if it is not available and continues with training and evaluation loops. Settings in the command are self-explanatory. Note that setting run_type define the type of run and if you want to train, then add "train". If set to "val", it runs inference on the validation set. Finally, if it is set to "test", it runs inference on the test set. You can combine different schemes  with _. For example, setting it to train_val_test, will train and run inference on Val and test as well.

```
mmf_run config=projects/mmf_transformer/configs/charades/direct.yaml  run_type=train_val dataset=charades model=mmf_transformer training.batch_size=4 training.num_workers=1 training.find_unused_parameters=True training.log_interval=100 training.max_updates=5000
```

Settings for each of the training parameters can be specified from command line as well. At the end of the training, the checkpoints are saved in mmf/save directory. We will use the saved checkpoints in the  for serving the model.
