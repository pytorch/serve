# Serverless Inference with MMS on FARGATE

This is self-contained step by step guide that shows how to create launch and server your deep learning models with MMS 
in a production setup. 
In this document you will learn how to launch MMS with AWS Fargate, in order to achieve a serverless inference.

## Prerequisites

Even though it is fully self-contained we do expect the reader to have some knowledge about the following topics:

* [MMS](https://github.com/awslabs/mxnet-model-server)
* [What is Amazon Elastic Container Service (ECS)](https://aws.amazon.com/ecs)
* [What is Fargate](https://aws.amazon.com/fargate)
* [What is Docker](https://www.docker.com/) and how to use containers

Since we are doing inference, we need to have a pre-trained model that we can use to run inference. 
For the sake of this article, we will be using 
[SqueezeNet model](https://github.com/awslabs/mxnet-model-server/blob/master/docs/model_zoo.md#squeezenet_v1.1). 
In short, SqueezeNet is a model that allows you to recognize objects in a picture. 

Now that we have the model chosen, let's discuss at a high level what our pure-container based solution will look like:

![architecture](https://s3.amazonaws.com/mxnet-model-server/mms-github-docs/MMS+with+Fargate+Article/AWS+Fargate+MMS.jpg)

In this document we are going to walk you through all the steps of setting up MMS 1.0 on Amazon Fargate services.
The steps in this process are as follows:

1. Familiarize yourself with MMS containers
2. Create a SqueezeNet task definition (with the docker container of MMS) 
3. Create AWS Fargate cluster
4. Create Application Load Balancer
5. Create Squeezenet Fargate service on the cluster
6. Profit!

Let the show begin...

## Familiarize Yourself With Our Containers 

With the current release of [MMS, 1.0](https://github.com/awslabs/mxnet-model-server/releases/tag/v1.0.0), 
Official pre-configured, optimized container images of MMS are provided on [Docker hub](https://hub.docker.com).

* [awsdeeplearningteam/mxnet-model-server](https://hub.docker.com/r/awsdeeplearningteam/mxnet-model-server)

```bash
docker pull awsdeeplearningteam/mxnet-model-server

# for gpu image use following command:
docker pull awsdeeplearningteam/mxnet-model-server:latest-gpu
```
In our article we are going to use the official CPU container image.

One major constraint for using Fargate service is that there is currently no support for GPU on Fargate.

The model-server container comes with a configuration file pre-baked inside the container.
It is highly recommended that you understand all the parameters of the MMS configuration file.
Familiarize yourself with the 
[MMS configuration](https://github.com/awslabs/mxnet-model-server/blob/master/docs/configuration.md) and 
[configuring MMS Container docs](https://github.com/awslabs/mxnet-model-server/blob/master/docker/README.md).
When you want to launch and host your custom model, you will have to update this configuration. 

In this tutorial, we will be use the squeezenet model from the following S3 link.

```
https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
```

Since MMS can consume model files from S3 buckets, we wouldn't need to bake the containers with the actual model files.

The last question that we need to address: how we should be starting our MMS within our container. 
And the answer is very simple, you just need to set the following 
[ENTRYPOINT](https://docs.docker.com/engine/reference/builder/#entrypoint):

```bash
mxnet-model-server --start --models https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar
```

You will now have a running container serving squeezenet model.

At this point, you are ready to start creating actual task definition.

**Note**: To start multiple models with the model-server, you could run the following command with multiple model names
```bash
# Example, following command starts model server with Resnet-18 and Squeezenet V1 models
$ mxnet-model-server --start --models https://s3.amazonaws.com/model-server/model_archive_1.0/squeezenet_v1.1.mar https://s3.amazonaws.com/model-server/model_archive_1.0/resnet-18.mar

```


## Create an AWS Fargate task to serve SqueezeNet model

This is the first step towards getting your own "inference service" up and running in a production setup. 

1. Login to the AWS console and go to the Elastic Cloud Service -> Task Definitions and Click “Create new Task Definition”:

![task def](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/1_Create_task_definition.png)

2. Now you need to specify the type of the task, you will be using the Fargate task:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/2_Select_Fargate.png)

3. The task requires some configuration, let's look at it step by step. First set the name:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3_Config_1.png)

Now is important part, you need to create a [IAM role](https://aws.amazon.com/iam) that will be used to publish metrics to CloudWatch:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/Task+Execution+IAM+Role+.png)

The containers are optimized for 8 vCPUs, however in this example you are going to use slightly smaller task with 4 vCPUs and 8 GB of RAM:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/cpu+and+ram.png)

4. Now it is time to configure the actual container that the task should be executing.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/container+step+1.png)
<br></br>
*Note:* If you are using a [custom container](https://github.com/awslabs/mxnet-model-server/blob/master/docs/mms_on_fargate.md#customize-the-containers-to-serve-your-custom-deep-learning-models), make sure to first upload your container to Amazon ECR or Dockerhub and replace the link in this step with the link to your uploaded container.

5. The next task is to specify the port mapping. You need to expose container port 8080. 
This is the port that the MMS application inside the container is listening on. 
If needed it can be configured via the config [here](https://github.com/awslabs/mxnet-model-server/blob/master/docker/mms_app_cpu.conf#L40).

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/port+8080.png)

Next, you will have to configure the health-checks. This is the command that ECS should run to find out whether MMS is running within the container or not. MMS has a pre-configured endpoint `/ping`
that can be used for health checks. Configure ECS to reach that endpoint at `http://127.0.0.1:8080/ping` using the `curl` command as shown below:

```bash
curl, http://127.0.0.1:8080/ping
```

The healthcheck portion of your container configuration should look like the image below:

![](https://s3.amazonaws.com/mxnet-model-server/mms-github-docs/MMS+with+Fargate+Article/add+container+healthcheck.png)

After configuring the health-checks, you can go onto configuring the environment, with the entry point that we have discussed earlier:

![](https://s3.amazonaws.com/mxnet-model-server/mms-github-docs/MMS+with+Fargate+Article/environtment.png)

Everything else can be left as default. So feel free to click `Create` to create your very first AWS Fargate-task. 
If everything is ok, you should now be able to see your task in the list of task definitions.

In ECS, `Services` are created to run Tasks. A service is in charge of 
running multiple tasks and making sure the that required number of tasks are always running, 
restarting un-healthy tasks, adding more tasks when needed. 
 
 To have your `inference service` accessible over the Internet, you would need to configure a load-balancer (LB). This LB will  be 
 in charge of serving the traffic from the Internet and redirecting it to these newly created tasks. 
 Let's create an Application Load Balancer now:

## Create a Load Balancer

AWS supports several different types of Load Balancers:

* Application Load Balancer: works on the level 7 of the OSI model (effectively with the HTTP/HTTPS protocols)
* TCP Load Balancer 

For your cluster you are going to use application load balancer.
1. Login to the EC2 Console.
2. Go to the “Load balancers” section.
3. Click on Create new Load Balancer.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/1__Create_Load_Balancer.png)

5. Choose Application Load Balancer.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/2__HTTP_HTTPS+.png)

6. Set all the required details. **Make a note of the VPC of the LB**. This is important since the LB's VPC and the ECS
cluster's VPC need to be same for them to communicate with each other.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3_2_Listeners_and_AZ+.png)

7. Next is configuring the security group. This is also important. Your security group should:

* Allow inbound connections for port 80 (since this is the port on which LB will be listening on)
* LBs security group needs to be added to the AWS Fargate service's security group, so that all the traffic from LB is accepted
by your "inference service". 

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/4.+Configure+Security+groups.png)

8. Routing configuration is simple. Here you need to create a “target group”. 
But, in your case the AWS Fargate service, that you will create later, will automatically create a target group.  
Therefore you will create dummy “target group” that you will delete after the creation of the LB. 

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/5.+Configure+Routing+(DUmmy).png)

9. Nothing needs to be done for the last two steps. `Finish` the creation and ...
10. Now you are ready to remove dummy listener and target group

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/8__Delete_the_dummy_listener.png)

Now that you are `done-done-done` with the Load Balancer creation, lets move onto creating our Serverless inference service.

## Creating an ECS Service to launch our AWS Fargate task

1. Go to Elastic Container Service → Task Definitions and select the task definitions name. Click on actions and select create service.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/1.+Go+to+task+definitions.png)

2. There are two important things on the first step (apart from naming):

* Platform version: It should be set to 1.1.0 .
* Number of tasks that the service should maintain as healthy all of the time, for this example you will set this to 3.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/number+of+tasks.png)

3. Now it is time to configure the VPC and the security group. **You should use the same VPC that was used for the LB (and same subnets!).**

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.1+Use+the+existing+VPC+Edit+sg.png)

4. As for the security group, it should be either the same security group as you had for the LB, or the one that accepts traffic from the LBs security group.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.2+SG+Use+existing.png)

5. Now you can connect your service to the LB that was created in the previous section. Select the "Application Load Balancer" and set the LB name:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.3+Add+load+balancing.png)

6. Now you need to specify which port on the LB our service should be listening on:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.4+Configure+load+blancer.png)

7. You are not going to use service discovery now, so uncheck it:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.2.5+Next.png)

8. In this document, we are not using auto-scaling options. For an actual production system, it is advisable to have this configuration setup.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/3.3+Auto+scaling.png)

9. Now you are `done-done-done` creating a running service. You can move to the final chapter of the journey, which is testing the service you created. 

## Test your service

First find the DNS name of your LB. It should be in `AWS Console -> Service -> EC2 -> Load Balancers` and click on the LB that you created.

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/lb_dns.png)

Now you can run the health checks using this load-balancer public DNS name, to verify that your newly created service is working:

```bash
curl InfraLb-1624382880.us-east-1.elb.amazonaws.com/ping 
```

```text
http://infralb-1624382880.us-east-1.elb.amazonaws.com/ping
{
    "status": "Healthy!"
}
```

And now you are finally ready to run our inference! Let's download an example image:
```bash
curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg
```

The image:

![](https://s3.amazonaws.com/mms-github-assets/MMS+with+Fargate+Article/kitten.jpg)

The output of this query would be as follows,

```bash
curl -X POST InfraLb-1624382880.us-east-1.elb.amazonaws.com/predictions/squeezenet_v1.1 -F "data=@kitten.jpg"
```

```text
{
      "prediction": [
    [
      {
        "class": "n02124075 Egyptian cat",
        "probability": 0.8515275120735168
      },
      {
        "class": "n02123045 tabby, tabby cat",
        "probability": 0.09674164652824402
      },
      {
        "class": "n02123159 tiger cat",
        "probability": 0.03909163549542427
      },
      {
        "class": "n02128385 leopard, Panthera pardus",
        "probability": 0.006105933338403702
      },
      {
        "class": "n02127052 lynx, catamount",
        "probability": 0.003104303264990449
      }
    ]
  ]
}
```

## Instead of a Conclusion

There are a few things that we have not covered here and which are very useful, such as:

* How to configure auto-scaling on our ECS cluster.
* Running A/B testing of different versions of the model with the Fargate Deployment concepts.

Each of the above topics require their own articles, so stay tuned!!

## Authors

* Aaron Markham
* Vamshidhar Dantu 
* Viacheslav Kovalevskyi (@b0noi) 
