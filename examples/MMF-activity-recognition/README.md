### MultiModal (MMF) Framework

Multi modality learning helps the AI solutions to get signals from different input sources such as language, video, audio and combine their result to imporve the inferences. 


[MultiModal (MMF) framework](https://ai.facebook.com/blog/announcing-mmf-a-framework-for-multimodal-ai-models/)  is a modular deep learning framework for vision and language multimodal research. MMF provides  starter code for several multimodal challenges, including the Hateful Memes, VQA, TextVQA, and TextCaps challenges. You can learn more about MMF from their [website](https://mmf.readthedocs.io/en/latest/?fbclid=IwAR3P8zccSXqNt1XCCUv4Ysq0qkD515T6K9JnhUwpNcz0zzRl75FNSio9REU) a [Github](https://github.com/facebookresearch/mmf?fbclid=IwAR2OZi-8rQaxO3uwLxwvvvr9cuY8J6h0JP_g6BBM-qM7wpnNYEZEmWOQ6mc). 

In the following, we will showcase an example from MMF repo for activity recognition from videos and how to serve the MMF model using Torchserve. 


### Activity Recognition from Videos

We are going to present an example of activity recognition on Charades video dataset. Basically in this example three modalities will be used to classify the activity from the video, image, audio and text. Images are extracted frames from the video, and audio is extracted from the vido and text is the captions related to frames in the video. In this case,  embedding for each of the modalities are captured and then [MMFTransformer](https://github.com/facebookresearch/mmf/blob/master/mmf/models/mmf_transformer.py) from the model zoo has been used for fusion of the embeddings. 

 There are a number of steps based on the discussed concepts on MMF webiste:

1. Define a new dataset, "Charades" and resgitering the dataset builder with MMF.
2. Define a model for our training, which is MMFTransformer model in this case.
3. Set up configs for dataset, model, the experiment (configs for training job).

In the following we discuss each of the steps in more details. 

#### New Dataset

In this example Charades dataset has been used which is a video dataset added in the [dataset zoo](https://github.com/facebookresearch/mmf/tree/master/mmf/datasets/builders/charades). We can define a new dataset in MMF by following this [guide](https://mmf.sh/docs/tutorials/dataset). **To add a new dataset**, we need to define a new dataset class which exteneds the Basedataset class from  mmf.datasets.base_dataset, where we need to override three methods, __init__, __getitem__ and __len__. These methods basially define how initialize ( set the path to the dataset), get each item from the dataset and then providing the length of the dataset. The Charades dataset class can be found [here](https://github.com/facebookresearch/mmf/blob/master/mmf/datasets/builders/charades/dataset.py#L16).  Also, we are able to set the [processors](https://github.com/facebookresearch/mmf/blob/master/mmf/configs/datasets/charades/defaults.yaml#L22) in the dataset config file and intialize them in the dataset class. 

The **next step** is to define a dataset builder class which extending the "BaseDatasetBuilder" class from mmf.datasets.base_dataset_builder. In this class essentially we need to override three methods, __init__, __build__ and __load__. Where in the __init __ method, the dataset class name is set (as we defined in the previous step), the __build__ method, is responsible for downloading the dataset and __load__ method is taking care of   loading the dataset, builds an object of class inheriting "BaseDataset" which contains your dataset logic and returns it. The dataset builder code is also availble [here](https://github.com/facebookresearch/mmf/blob/master/mmf/datasets/builders/charades/builder.py).

**Final step** is to register the dataset builder with mmf, where we can use the resgistery function as decorator, such as @registry.register_builder("charades"). 

#### Model Defnition

To train a multimodal model, we need to define a model that will take the features from our modalities (using modality encoders) as inputs and trains for the task in hand, where in this example is activity recognition. For activity recognition basically the task is classification on activity labels. In this example, [MMFTransformer](https://github.com/facebookresearch/mmf/blob/master/mmf/models/mmf_transformer.py) is used that extend [base tranformers](https://github.com/facebookresearch/mmf/blob/master/mmf/models/transformers/base.py) from mmf.models.transformers.base model available in the MMF model zoo.

 Generally, any defined MMF model class requires to extend the BaseModel from mmf.models.base_model, where we need to pass the configs in the __init__ method and impelement the build and forward method. Init method takes the realted config and build method builds all the essential module used in the model including encoders. 

#### Configurations

As indicated in [MMF docs](https://mmf.sh/docs/notes/configuration), there are seprate config files for datasets, models and experiments, all the configs can be found in config directory. For this example, [dataset config](https://github.com/facebookresearch/mmf/blob/master/mmf/configs/datasets/charades/defaults.yaml) sets the path to different sets (train/val/test), and processors and their related parameters. Similarly, [model config](https://github.com/facebookresearch/mmf/blob/master/mmf/configs/models/mmf_transformer/defaults.yaml) can set the specifics for the model including in hierachical mode , different modalities, the encoder used in each modality, the model head type, loss funciton and etc. [Experiment config](https://github.com/facebookresearch/mmf/blob/video_datasets/projects/mmf_transformer/configs/charades/direct.yaml) is where setting for experiment such as optimizer specifics, schedular,  evaluation metrics, training parameters such as batch sizem, number of iterations and so on can be configured. 

#### Running the experiment

To train the activity recognition MMF model as we discussed, the [Charades dataset](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/datasets/builders/charades/dataset.py#L16) has been added to MMF and we use  [MMFTransformer](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/models/mmf_transformer.py#L34) model. [Config files](https://github.com/facebookresearch/mmf/tree/video_datasets/mmf/configs) for the model, dataset and experiment has been set as well. Now we need to run the following command to start tarining the model, the following command starts with downloaing the datasets if it is not availble and continues with training and evaluation loops. Settings in the command are self-explanatory. Note that setting run_type defne the type of run and if you want to train, then add "train". If set to "val", it runs inference on validation set. Finally, if it is set to "test", it runs inference on test set. You can combine different schemes  with _. For example, setting it to train_val_test, will train and run inference on Val and test as well.

```
mmf_run config=projects/mmf_transformer/configs/charades/direct.yaml  run_type=train_val dataset=charades model=mmf_transformer training.batch_size=4 training.num_workers=1 training.find_unused_parameters=True training.log_interval=100 training.max_updates=5000
```

Settings for each of the training parameters can be specified from command line as well. At the end of the training, the checkpoints are saved in mmf/save directory. We will use the saved checkpoints in the next step for serving the model.

### Serving Activity Recognition MMF Model with Torchserve

Now, we have the trained MMF model for activity recognition, it can be served in production with [Torchserve](https://github.com/pytorch/serve). 

To serve a model using Torchserve, we need to bundle the model artifacts and a handler into a .mar file which is an archive format that torchserve uses to serve our model, model_archiver package does this step. The .mar file will get extracted in a temp directory and the Path will be added to the PYTHONPATH.

#### Requirements

Install [Torchserve](https://github.com/pytorch/serve)


Install [MMF](https://github.com/facebookresearch/mmf/tree/video_datasets)

#### Tochserve Custom Handler

For the activity recognition MMF model, we need to provide a custom handler. The handler generally extends the [Base handler](https://github.com/pytorch/serve/blob/master/ts/torch_handler/base_handler.py). The [handler](https://github.com/pytorch/serve/blob/adding_MMF_example/examples/MMF-activity-recognition/handler.py) for MMF model, needs to load and intialize the model in the initialize method and then in the preprocess, mimics the logic in dataset processors to make a sample form the input video and its realted text ( preprocess the video and make the related tensors to video,audio and text). The inference method run the preprocessed samples through the  MMF model and send the outputs to the post process method. 

To intialize the MMF model in the [intialization method](https://github.com/pytorch/serve/blob/adding_MMF_example/examples/MMF-activity-recognition/handler.py#L65), there are few poinst to consider. We need to load config (Line#91) using OmegaConfing, then setup_very_basic_config (Line#93) function from mmf.utils.logger and setup_imports (Line#95) from  mmf.utils.env need to be called to setup the enviroment for loading the model. Finally to load the model (Line#96), we pass the model config to the  [MMFTransformer](https://github.com/facebookresearch/mmf/blob/video_datasets/mmf/models/mmf_transformer.py) model. 

#### Getting started on Serving

To make the .mar file in the current setting we will use model_archiver as follow.

After training the MMF model, the final checkpoints are saved in the mmf/save/ directory, where we need to use it as the serialized file for model-archiver, a pretrained checkpoint can be downloaded from [here](https://mmfartifacts.s3-us-west-2.amazonaws.com/mmf_transformer_Charades_final.pth). The Charades action labels along with the config file that covers all the setting of this experiment (can be found in mmf/save/config.yaml) are passed as extra files, they are getting used in the handler. Please make sure to change the label and config files accordingly in the handler (Lines #68, #84) if you are passing different ones. A ready mar file can be found [here](https://mmfartifacts.s3-us-west-2.amazonaws.com/MMF_model.mar) as well.

```
torch-model-archiver --model-name MMF_model --version 1.0 --serialized-file $(path to checkpoints) --handler handler.py --extra-files "charades_action_lables.csv,$(path to the aggreagted config file,saved in mmf/save/config.yaml)"
```

charades_action_lables.csv has been created using the [notebook](https://github.com/pytorch/serve/blob/adding_MMF_example/examples/MMF-activity-recognition/Generting_Charades_action_lables.ipynb) and is available here as well, which contains the labels from the dataset to be used for mapping predictions to labels. Running this will result in MMF_model.mar in the current directory.

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

