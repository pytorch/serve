# Cloudformation Templates
Torchserve provides configurable cloudformation templates to spin up AWS instances running torchserve.

*Following instructions requires you have [aws-cli](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html) installed as a prerequisite*

## Single EC2 instance
* To spinup a single EC2 instance running Torchserve use the `ec2.yaml` template
* Run the following command with the an ec2-keypair, and optionally an instance type (default: c5.4xlarge)
```
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
cd cloudformation/
aws cloudformation create-stack \
  --stack-name torchserve \
  --region us-west-2 \
  --template-body file://ec2.yaml \
  --capabilities CAPABILITY_IAM \
  --parameters ParameterKey=KeyName,ParameterValue=<ec2-keypair-name> \
               ParameterKey=InstanceType,ParameterValue=<instance-type>
```

* Once the cloudformation stack creation is complete, you can get the **TorchServeManagementURL** and **TorchServeInferenceURL** of the instance from the cloudformation output tab on AWS console and test with the following commands
```
> curl --insecure -X POST "<TorchServeManagementURL>/models?initial_workers=1&synchronous=false&url=https://torchserve.s3.amazonaws.com/mar_files/squeezenet1_1.mar"
{
  "status": "Processing worker updates..."
}

> curl --insecure "<TorchServeInferenceURL>/ping"
{
  "status": "Healthy"
}

> curl --insecure "<TorchServeManagementURL>/models"
{
    "models": [
        {
            "modelName": "squeezenet1_1",
            "modelUrl": "https://torchserve.s3.amazonaws.com/mar_files/squeezenet1_1.mar"
        }
     ]
}

> curl -O https://s3.amazonaws.com/model-server/inputs/kitten.jpg

> curl --insecure "<TorchServeInferenceURL>/predictions/squeezenet1_1" -T kitten.jpg
[
    {
        "tabby": 0.2752002477645874
    },
    {
        "lynx": 0.2546876072883606
    },
    {
        "tiger_cat": 0.24254210293293
    },
    {
        "Egyptian_cat": 0.2213735282421112
    },
    {
        "cougar": 0.0022544863168150187
    }
]
```
* Once the instance is up and running, TorchServe logs are published to cloudwatch under the LogGroup=`<stack-name>/<ec2-instance-id>/TorchServe` e.g. `torchserve/i-0649487ecbe691676/TorchServe`

* If you have to stop or restart torchserve, you'll have to ssh into the host

```
ssh -i <ec2-keypair-name> ubuntu@<ec2-dns>
```

```
cd /
sudo bash
export PATH="/home/ubuntu/miniconda/bin:$PATH"
conda init bash
# IMPORTANT: You may need to close and restart your shell after running 'conda init'.
conda activate torchserve
torchserve --stop
torchserve --start --model-store ./model_store --ts-config /etc/torchserve/config.properties
```

* To terminate the instance and delete the stack you can run `aws cloudformation delete-stack --stack-name <stack-name>`

