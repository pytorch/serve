# Digit recognition model with MNIST dataset using a Kubernetes cluster

In this example, we show how to use a pre-trained custom MNIST model to performing real time Digit recognition with TorchServe.
We will be serving the model using a Kubernetes cluster deployed using [minikube](https://minikube.sigs.k8s.io/docs/start/).

The inference service would return the digit inferred by the model in the input image.

We used the following pytorch example to train the basic MNIST model for digit recognition :
https://github.com/pytorch/examples/tree/master/mnist

## Serve an MNIST model on TorchServe docker container

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path/serve

 ### Create a torch model archive using the torch-model-archiver utility to archive the above files.

  ```
  torch-model-archiver --model-name mnist --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/image_classifier/mnist/mnist_handler.py
  ```

  ### Move .mar file into model_store directory

  ```
  mkdir model_store
  mv mnist.mar model_store/
  ```

  ### Start kubernetes cluster

  We start the cluster mounting the location of `serve` to `/host`

  The following command works if torchserve is under $HOME/serve
  ```
  minikube start --mount-string="$HOME/serve:/host" --mount
  ```

  ### Deploy the cluster

  In this example, we are launching a cluster with a single pod.
  We are exposing ports 8080 and 8081
  We are also mapping the the `model_store` directory created on host to
  `/home/model-server/model-store` on the container

  ```
  kubectl apply -f kubernetes/examples/mnist/deployment.yaml
  ```

  Make sure the pod is running

  ```
  kubectl get pods
  ```
  shows the output
  ```
  NAME                      READY   STATUS    RESTARTS   AGE
  ts-def-5c95fdfd57-m446t   1/1     Running   0          58m

  ```

  ### Create a Service
  We create a service to send inference request to the pod.
  We are using `NodePort` so that the cluster can be accessed by the outside world.

  ```
  kubectl apply -f kubernetes/examples/mnist/service.yaml
  ```

  Verify the service is running

  ```
  kubectl get svc
  ```
  shows the output
  ```

    NAME         TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)                         AGE
    ts-def       NodePort    10.109.14.120   <none>        8080:30160/TCP,8081:30302/TCP   59m

  ```

  ### Make cluster accessible by localhost

  We use kubectl port-forward to make the cluster accessible from the local machine. This will run in the background. Make sure to kill the process when the test is done.

  ```
  kubectl port-forward svc/ts-def 8080:8080 8081:8081 &
  ```

  ### Register the model on TorchServe using the above model archive file

  ```
  curl -X POST "localhost:8081/models?model_name=mnist&url=mnist.mar&initial_workers=4"
  ```

  If this succeeeds, you will see a message like below

  ```
  {
  "status": "Model \"mnist\" Version: 1.0 registered with 4 initial workers"
  }
  ```

  ### Run digit recognition inference

  ```
  curl http://127.0.0.1:8080/predictions/mnist -T examples/image_classifier/mnist/test_data/0.png
  ```

   The output in this case will be a `0`


  ### Delete the cluster

  ```
  minikube stop
  minikube delete
  ```
