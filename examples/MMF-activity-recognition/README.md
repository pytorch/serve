### MultiModal (MMF) Framework

Multi modality learning helps the AI solutions to get signals from different input sources such as language, video, audio and combine their result to imporve the inferences. 



[MultiModal (MMF) framework](https://ai.facebook.com/blog/announcing-mmf-a-framework-for-multimodal-ai-models/)  is a modular deep learning framework for vision and language multimodal research. MMF provides  starter code for several multimodal challenges, including the Hateful Memes, VQA, TextVQA, and TextCaps challenges. You can learn more about MMF from their [website](https://mmf.readthedocs.io/en/latest/?fbclid=IwAR3P8zccSXqNt1XCCUv4Ysq0qkD515T6K9JnhUwpNcz0zzRl75FNSio9REU) a [Github](https://github.com/facebookresearch/mmf?fbclid=IwAR2OZi-8rQaxO3uwLxwvvvr9cuY8J6h0JP_g6BBM-qM7wpnNYEZEmWOQ6mc). 

MMF is using [OmegaConfig](https://github.com/omry/omegaconf?fbclid=IwAR1jxgSCJUzKqnbVI46vdgPIv9psaBlkwjYAox5xtm4c4TtwTPpzaWhCL_k) for different configuration and settings, where OmegaConf is a hierarchical configuration system, which support merging configurations from multiple sources (YAML config files, dataclasses/objects and CLI arguments). 

MMF has a [model zoo](https://mmf.sh/docs/notes/model_zoo) providing a number of pretrained models such as [VilBert](https://arxiv.org/abs/1908.02265)  and MMF Transformer, and a compelmenty  [Dataset zoo](https://mmf.sh/docs/notes/dataset_zoo). 

MMF is multi-tasking framework which means essentialy you are able to use multiple dataasets for your training job on different tasks / models at the same time. Here is where the modularity and config system of MMF can picture a clear logic for your multi-task training job. 

The diagram below shows the MMF flow for a visual question answering task. 

![](/Users/hamidnazeri/Documents/Screen Shot 2021-04-09 at 4.39.55 PM.png)

### Key Concepts

There are some basic terminology and cocnepts that helps you to get onboarded with MMF easiser. 

**Datasets** has their own config files where you can set the specifics of the dataset, all the existing dataset configs in the zoo can be found [here](https://github.com/facebookresearch/mmf/tree/master/mmf/configs/datasets). Also, one can add a new dataset where the tutorial can be found [here](https://mmf.sh/docs/tutorials/dataset) , in this example Charades dataset has been added to the MMF.

 Basically, MMF is dataset agnostic and each dataset would require 4 components to be added, Dataset builder, Default configuration, Dataset class, Dataset metric, details of how to create these components are explained in the tutorial. 

**Models** similar to datasets they have their own config files, where you can set the specifics, including what modalities you are using, the encoders for each of the models, for example if you are using image as a modality, you can set "Resnet" as the image encoder or Bert model for your text encoder. The list of existing models and their config files in zoo can be found [here](https://github.com/facebookresearch/mmf/tree/master/mmf/configs/models). 

**Registery** is playing a centeral role in MMF where all the models, tasks,  dataset and necessary information for running a workload in the MMF is registered here. The functions in the registery class can be used as decorates to register different classes, for example to [resgiter the dataset builder](https://github.com/facebookresearch/mmf/blob/master/mmf/datasets/builders/clevr/builder.py#L19) that will register your dataset with MMF. 

**Configuration** as mentioned before, MMF uses OmegaConfig that provides a hierarchical configuration and for each datasets, model and experiment there is seperate config file that can be found or places under MMF/configs. Also, at any place in the MMF code base you can access the configs by registry.get('config') and all the attributes can be accessed using 'dot' notation, for example registry.get('config').training.max_updates. Also, you are able to use override the config values by setting it from command line, for example for using DataParallel in case of having multi-gpus you can just pass the training.data_parallel True at the end of your command. 

**Processors** are used to keep data processing pipelines as similar as possible for different datasets and allow code reusability. You can think of the processors similar to transforms provided in torchvision, where it takes a sample from convert it to a useable form by the model.  Processors take in a dict with keys corresponding to data they need and return back a dict with processed data.  Processors are initialized as member variables of the dataset and can be used while generating samples. This helps keep processors independent of the rest of the logic by fixing the signatures they require. You can write you own processors beside the defaults exisiting in the MMF by following the guide [here](https://mmf.sh/api/lib/datasets/processors.html). 

**SampleList** is a list of sample and how you pass a batch of data, models integrated with MMF receive a `SampleList` as an argument which makes the trainer unopinionated about the models as well as the datasets.

### Installation

You can install either from source or using python wheels, further details can be found in the  [installation guide](https://mmf.sh/docs/). It is recommneded to make a virtual enviroment for installation.

Install from source:

`git clone https://github.com/facebookresearch/mmf.git`
`cd mmf`
`pip install --editable .`

Pip install:

`pip install --upgrade --pre mmf`

and for latest from MMF github repo :

`pip install git+https://github.com/facebookresearch/mmf.git`



### Activity Recognition from Videos

We are going to present an example of activity recognition on Charades video dataset. Basically in this example three modalities will be used to classify the activity from the video, image, audio and text. Images are extracted frames from the video, and audio is extracted from the vido and text is the captions related to frames in the video. In this case,  embedding for each of the modalities are captured and then [MMFTransformer](https://github.com/facebookresearch/mmf/blob/master/mmf/models/mmf_transformer.py) from the model zoo has been used for fusion of the embeddings. 

 There a number of steps based on the discussed concepts in previous section :

1. Define a new dataset, "Charades" and resgitering the dataset builder with MMF.
2.  Define a model for our training, which is essentially a wrapper setting the configs for MMFTransformer model.
3. Set up configs for dataset, model, the experiment (configs for training job) and zoo(?)

In the following we discuss each of the steps in more details. 

#### New Dataset

In this example Charades dataset has been used which is a video dataset not availble in the [dataset zoo]().  As mentioned before, we can define a new dataset in MMF by following this [guide](https://mmf.sh/docs/tutorials/dataset). **First,** we need to define a new dataset class which exteneds the Basedataset class from  mmf.datasets.base_dataset, where we need to override three methods, __init__, __getitem__ and __len__. These methods basially define how initialize ( set the path to the dataset), get each item from the dataset and then providing the length of the dataset. The Charades dataset class can be found [here](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/datasets/builders/charades/dataset.py#L16).  Also, we are able to set the processors in the dataset class as well, where we can define and set the processors for different modalities. In this Charades dataset example provides here, a fix number of frames from each video clip will be extracted. 

The **next step** is to define a dataset builder class which extending the "BaseDatasetBuilder" class from mmf.datasets.base_dataset_builder. In this class essentially we need to override three methods, __init__, __build__ and __load__. Where in the __init __ method, the dataset class name is set (as we defined in the previous step), the __build__ method, is responsible for downloading the dataset and __load__ method is taking care of   loading the dataset, builds an object of class inheriting "BaseDataset" which contains your dataset logic and returns it. The dataset builder code is also availble [here](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/datasets/builders/charades/builder.py).

**Final step** is to register the dataset builder with mmf, where we can use the resgistery function as decorator, such as @registry.register_builder("charades"). 

#### Model Defnition

To train a multimodal model, we need to define a model that will take the features from our modalities (using modality encoders) as inputs and trains for the task in hand, where in this example is activity recognition. For activity recognition basically the task is classification on activity labels. In this example, [MMFTransformer](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/models/mmf_transformer.py) is used that extend [base tranformers](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/models/transformers/base.py) from mmf.models.transformers.base model available in the MMF model zoo.

 Generally, any defined MMF model class requires to extend the BaseModel from mmf.models.base_model, where we need to pass the configs in the __init__ method and impelement the build  and forward method. Init method takes the realted config and build method builds all the essential module used in the model including encoders. 

#### Configurations

As discussed before in MMF, there are seprate config files for datasets, models and experiments, all the configs can be found in config directory. For this example, [dataset config](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/configs/datasets/charades/defaults.yaml) sets the path to different sets (train/val/test), and processors and their related parameters. Similarly, [model config](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/configs/models/mmf_transformer/defaults.yaml) can set the specifics for the model including in hierachical mode , different modalities, the encoder used in each modality, the model head type, loss funciton and etc. [Experiment config](https://github.com/facebookresearch/mmf/blob/video_datasets/projects/mmf_transformer/configs/charades/direct.yaml) is where setting for experiment such as optimizer specifics, schedular,  evaluation metrics, training parameters such as batch sizem, number of iterations and so on can be configured. 

#### Running the experiment

To train the activity recognition MMF model as we discussed, the [Charades dataset](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/datasets/builders/charades/dataset.py#L16) has been added to MMF and we use  [MMFTransformer](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/models/mmf_transformer.py#L34) model. [Config files](https://github.com/facebookresearch/mmf/tree/video_datasets/mmf/configs) for the model, dataset and experiment has been set as well. Now we need to run the following command to start tarining the model.

```
mmf_run config=projects/mmf_transformer/configs/charades/direct.yaml model=mmf_transformer dataset=charades training.batch_size=32  training.max_updates=500 val=300
training.max_updates=1500
```

Settings for each of the training parameters can be specified from command line as well. At the end of the training, the checkpoints are saved in mmf/save directory. We will use the saved checkpoints in the next step for serving the model.

#### Serving Activity Recognition MMF Model with Torchserve

Now, we have the trained MMF model for activity recognition, it can be served in production with [Torchserve](https://github.com/pytorch/serve). 

To serve a model using Torchserve, we need to bundle the model artifacts and a handler into a .mar file which is an archive format that torchserve uses to serve our model, model_archiver package does this step. The .mar file will get extracted in a temp directory and the Path will be added to the PYTHONPATH.

##### Requirements

Install [Torchserve](https://github.com/pytorch/serve)
Install [MMF](https://github.com/facebookresearch/mmf/tree/video_datasets)

##### Tochserve Custom Handler

For the activity recognition MMF model, we need to provide a custom handler. The handler generally extends the [Base handler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py). The [handler](https://github.com/pytorch/serve/blob/adding_MMF_example/examples/MMF-activity-recognition/handler.py) for MMF model, needs to load and intialize the model in the initialize method and then in the preprocess, mimics the logic in dataset processors to make a sample form the input video and its realted text ( preprocess the video and make the related tensors to video,audio and text). The inference method run the preprocessed samples through the  MMF model and send the outputs to the post process method. 

To intialize the MMF model in the [intialization method](https://github.com/pytorch/serve/blob/adding_MMF_example/examples/MMF-activity-recognition/handler.py#L65), there are few poinst to consider. We need to load config (Line#91) using OmegaConfing, then setup_very_basic_config() (Line#93) function from mmf.utils.logger    and setup_imports()  (Line#95) from  mmf.utils.env need to be called to setup the enviroment for loading the model. Finally to load the model (Line#96), we pass the model config to the  [MMFTransformer](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/models/mmf_transformer.py) model. 

##### Getting started on Serving

To make the .mar file in the current setting we will use model_archiver as follow. ( if you have already have downloaded the .mar file you can skip this step)

Make sure to compress configs, datasets and models folder.

```
torch-model-archiver --model-name MMF_model --version 1.0 --serialized-file $(path to checkpoints) --handler handler.py --extra-files "charades.csv,mmf_utils.py,transforms.py,$(path to the aggreagted config file,saved in mmf/save/config.yaml)"
```

charades.csv has been created using the [notebook](https://github.com/apsdehal/mmf_video_audio/blob/mmft_audio_video_disney/data/how_to_use_mickeynetics.ipynb) and is available here as well, which contains the labels from the dataset to be used for setting the model ouput shape and also index to class mapping. Running this will result in MMF_model.mar in the current directory.

The next step is to make a model_store and move the .mar file to it:

```
mkdir model_store
mv MMF_model.mar model_store
```

Now we can start serving our model:

```
torchserve --start --model-store model_store
curl -X POST "localhost:8081/models?model_name=MMF_model&url=MMF_model.mar&batch_size=1&max_batch_delay=5000&initial_workers=1&synchronous=true"
```

Sending inference request using

The examples of video and info.json is avalilbe here for demonstration purposes. **Please make sure to set the corret path to the video and info.json in the request.py.**

```
Python request.py
```

This will write the results in response.txt in the current directory.

