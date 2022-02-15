Note: the following benchmarking suite requires an AWS setup, and is an expensive operation involving several high-compute ec2 instances.

This apache-bench based benchmark aims to do a multiple batch-size based inference benchmarking as per provided config in a *.yaml file. Each config represents one model.

Check out a sample vgg11 model config at the path: `tests/suite/vgg11.yaml`

### Setup

* Ensure you have access to an AWS account i.e. [setup](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) your environment such that awscli can access your account via either an IAM user or an IAM role. An IAM role is recommended for use with AWS. For the purposes of testing in your personal account, the following managed permissions should suffice: <br>
-- [AmazonEC2ContainerRegistryFullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess) <br>
-- [AmazonEC2FullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonEC2FullAccess) <br>
-- [AmazonS3FullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/AmazonS3FullAccess) <br>
-- [IAMFullAccess](https://console.aws.amazon.com/iam/home#policies/arn:aws:iam::aws:policy/IAMFullAccess) (or at the least iam:passrole). <br>
-- [CloudWatchFullAccess](https://console.aws.amazon.com/iam/home#/policies/arn:aws:iam::aws:policy/CloudWatchFullAccess$jsonEditor) <br>

* [Create](https://docs.aws.amazon.com/cli/latest/reference/ecr/create-repository.html) an ECR repository with the name “torchserve-benchmark” in the us-west-2 region, e.g.
```
aws ecr create-repository --repository-name torchserve-benchmark --region us-west-2
```
If you'd like to use your own repo, edit the `config.yaml` file at `serve/benchmarks/automated/tests/suite/benchmark/config.yaml`
* Ensure you have [docker](https://docs.docker.com/get-docker/) client set-up on your system - osx/ec2
* Adjust the following global variables to your preference in the file `serve/benchmarks/automated/tests/suite/benchmark/config.yaml` <br>
-- iam_instance_profile :this role is attached to all ec2 instances created as part of the benchmarking process. Create this as described [here](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html#create-iam-role). Default role name is 'EC2Admin'.<br>
Use the following commands to create a new role if you don't have one you can use.
1. Create the trust policy file `ec2-admin-trust-policy.json` and add the following content:
```
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "ec2.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```
2. Create the EC2 role as follows:
```
aws iam create-role --role-name EC2Admin --assume-role-policy-document file://ec2-admin-trust-policy.json
```
3. Add the permissions to the role as follows:
```
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/IAMFullAccess --role-name EC2Admin
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2FullAccess --role-name EC2Admin
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess --role-name EC2Admin
aws iam attach-role-policy --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess --role-name EC2Admin
```
-- s3_bucket_benchmark_artifacts :all temporary benchmarking artifacts including server logs will be stored in this bucket. Note that this bucket must be in the same account, or the credentials being used should have read and write access to the bucket. <br>
Use the following command to create a new S3 bucket if you don't have one you can use.
```
aws s3api create-bucket --bucket <torchserve-benchmark> --region us-west-2 --create-bucket-configuration LocationConstraint=us-west-2
```
-- default_docker_dev_ecr_repo :docker image used for benchmarking will be pushed to this repo <br>
* If you're running this setup on an EC2 instance, please ensure that the instance's security group settings 'allow' inbound ssh port 22. Refer [docs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/security-group-rules.html).

*The following steps assume that the current working directory is serve/.*

1. Create or use any python virtual environment
```
sudo apt-get install python3-venv
python3 -m venv bvenv
source bvenv/bin/activate
# Ensure you have the latest pip
pip3 install -U pip
```
2. Install requirements for the benchmarking 
```
pip install -r test/benchmark/requirements.txt
```
3. Make sure that you've setup AWS account correctly
```
aws sts get-caller-identity
```
4. The automation scripts uses the ts-config from the following location: `benchmarks/config.properties`. Make changes to this file in the current local folder to use this across all the runs.
5. The simplest way to run a benchmark is to spin-up the ec2 instance type of your choice (must be a DLAMI), and run the benchmark with `--local-execution`, this will run through the models located in `benchmarks/automated/tests/suite/`, and execute benchmarks against these on the current instance. 
Start the benchmark run as follows (run this a pseudo shell such as tmux or screen, as this is a long-running script):
```
python benchmarks/automated/run_benchmark.py --local-execution
```
7. Another method is to execute the above command from your desktop terminal, **without** the argument `local-exeuction`. This will cause the instance types mentioned in the `<model>.yaml` files to be spun up. For each of the model config files under `benchmarks/automated/tests/suite/`, e.g., vgg11.yaml, set the list of instance types you want to test on:
```
instance_types:
  - c4.4xlarge
  - p3.8xlarge"
```
Start the benchmark run as follows:
```
python benchmarks/automated/run_benchmark.py --local-execution
```

The final benchmark report will be available in markdown format as `report.md` in the `serve/` folder. 

**Example report for vgg11 model**


### Benchmark report

**vgg11 | eager_mode | c5.18xlarge | batch size 1**
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg11 | 100 | 1000 | 0 | 2.05 | 47419 | 54745 | 58781 | 48852.156 | 0.0 | 589.16 | 709.42 | 709.42 | 1905.05 | 1904.91 | 44589.48 | 1.09 | 

**vgg11 | eager_mode | c5.18xlarge | batch size 8**
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg11 | 100 | 1000 | 0 | 8.11 | 12205 | 13162 | 14772 | 12334.135 | 0.0 | 3431.05 | 3525.94 | 3525.94 | 3872.42 | 3872.04 | 7958.16 | 53.27 | 

**vgg11 | eager_mode | c5.18xlarge | batch size 4**
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg11 | 100 | 1000 | 0 | 5.55 | 17891 | 18936 | 19965 | 18017.484 | 0.0 | 2304.79 | 2412.98 | 2412.98 | 2820.51 | 2820.24 | 14423.28 | 52.17 | 

**vgg11 | eager_mode | c5.18xlarge | batch size 2**
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg11 | 100 | 1000 | 0 | 3.51 | 28732 | 29900 | 30531 | 28520.545 | 0.0 | 748.97 | 1431.84 | 1431.84 | 2226.96 | 2226.79 | 25045.02 | 49.16 | 

**vgg11 | scripted_mode | c5.18xlarge | batch size 1**
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg11 | 100 | 1000 | 0 | 2.06 | 48058 | 50794 | 51760 | 48618.091 | 0.0 | 874.51 | 1012.23 | 1012.23 | 1900.22 | 1900.11 | 44363.84 | 1.07 | 

**vgg11 | scripted_mode | c5.18xlarge | batch size 4**
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg11 | 100 | 1000 | 0 | 5.50 | 18055 | 19159 | 19844 | 18171.083 | 0.0 | 2230.75 | 2316.4 | 2316.4 | 2846.7 | 2846.34 | 14550.68 | 51.29 | 

**vgg11 | scripted_mode | c5.18xlarge | batch size 3**
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg11 | 100 | 1000 | 1 | 4.51 | 22138 | 23074 | 23792 | 22165.721 | 0.1 | 1804.03 | 2160.02 | 2160.02 | 2597.17 | 2597.01 | 18563.88 | 50.2 | 

**vgg11 | scripted_mode | c5.18xlarge | batch size 2**
 | Benchmark |Model |Concurrency |Requests |TS failed requests |TS throughput |TS latency P50 |TS latency P90 |TS latency P99 |TS latency mean |TS error rate |Model_p50 |Model_p90 |Model_p99 |predict_mean |handler_time_mean |waiting_time_mean |worker_thread_mean |
 |--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | 
 | AB | vgg11 | 100 | 1000 | 0 | 3.47 | 28765 | 29849 | 30488 | 28781.227 | 0.0 | 1576.24 | 1758.28 | 1758.28 | 2249.52 | 2249.34 | 25210.43 | 46.77 | 


## Features of the automation:

3. To run a test containing a specific string, use the `--run-only` flag. Note that the argument is 'string matched' i.e. if the test-name contains the supplied argument as a substring, the test will run. 
```
# To run mnist test
python benchmarks/automated/run_benchmark.py --run-only mnist

# To run fastrcnn test
python benchmarks/automated/run_benchmark.py --run-only fastrcnn

# To run bert_neuron and bert_cpu
python benchmarks/automated/run_benchmark.py --run-only bert_cpu

# To run vgg11 test
python benchmarks/automated/run_benchmark.py --run-only vgg11

# To run vgg16 test
python benchmarks/automated/run_benchmark.py --run-only vgg16

# To run multiple:
python benchmarks/automated/run_benchmark.py --run-only vgg11 vgg16 bert_cpu
```

4. You can benchmark a specifc branch of the torchserve github repo by specifying the flag `--use-torchserve-branch` e.g., 
```
python benchmarks/automated/run_benchmark.py --use-torchserve-branch issue_1115
```

5. Once the docker image is built, you may choose to not have it re-built by passing the argument `--skip-docker-build` e.g.,
```
python benchmarks/automated/run_benchmark.py --skip-docker-build
```

6. If you do not wish to benchmark on different instance types specified in the model config `*.yaml`, you may pass the argument `--local-execution`. In this case, the instance types specified in the model config `*.yaml` file are *ignored*. Also, in this case, all the model benchmarks will be performed *sequentially* in order to 
```
python benchmarks/automated/run_benchmarks.py --local-execution
```