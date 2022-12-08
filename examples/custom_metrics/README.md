# Monitoring Torchserve custom metrics with mtail metrics exporter and prometheus

In this example, we show how to use a pre-trained custom MNIST model and export the custom metrics using mtail and prometheus

We used the following pytorch example to train the basic MNIST model for digit recognition : https://github.com/pytorch/examples/tree/master/mnist

Run the commands given in following steps from the parent directory of the root of the repository. For example, if you cloned the repository into /home/my_path/serve, run the steps from /home/my_path

## Steps

- Step 1: In this example we introduce a new custom metric `SizeOfImage` in the custom handler and export it using mtail.

  ```python
  def preprocess(self, data):
    metrics = self.context.metrics
    input = data[0].get('body')
    metrics.add_size('SizeOfImage', len(input) / 1024, None, 'kB')
    return ImageClassifier.preprocess(self, data)
  ```

  Refer: [Custom Metrics](https://github.com/pytorch/serve/blob/master/docs/metrics.md#custom-metrics-api)
  Refer: [Custom Handler](https://github.com/pytorch/serve/blob/master/docs/custom_service.md#custom-handlers)

- Step 2: Create a torch model archive using the torch-model-archiver utility to archive the above files.

  ```bash
  torch-model-archiver --model-name mnist --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler examples/custom_metrics/mnist_handler.py
  ```

- Step 3: Register the model on TorchServe using the above model archive file.

  ```bash
  mkdir model_store
  mv mnist.mar model_store/
  torchserve --start --model-store model_store --models mnist=mnist.mar
  ```

- Step 4: Install [mtail](https://github.com/google/mtail/releases)

  ```bash
  wget https://github.com/google/mtail/releases/download/v3.0.0-rc47/mtail_3.0.0-rc47_Linux_x86_64.tar.gz
  tar -xvzf mtail_3.0.0-rc47_Linux_x86_64.tar.gz
  chmod +x mtail
  ```

- Step 5: Create a mtail program. In this example we using a program to export default custom metrics.

  Refer: [mtail Programming Guide](https://google.github.io/mtail/Programming-Guide.html).

- Step 6: Start mtail export by running the below command

  ```bash
  ./mtail --progs examples/custom_metrics/torchserve_custom.mtail --logs logs/model_metrics.log
  ```

  The mtail program parses the log file extracts info by matching patterns and presents as JSON, Prometheus and other databases. https://google.github.io/mtail/Interoperability.html

- Step 7: Make Inference request

  ```bash
  curl http://127.0.0.1:8080/predictions/mnist -T examples/image_classifier/mnist/test_data/0.png
  ```

  The inference request logs the time taken for prediction to the model_metrics.log file.
  Mtail parses the file and is served at 3903 port

  `http://localhost:3903`

- Step 8: Sart Prometheus with mtailtarget added to scarpe config

  - Download [Prometheus](https://prometheus.io/download/)

  - Add mtail target added to scrape config in the config file

  ```yaml
  scrape_configs:
    # The job name is added as a label `job=<job_name>` to any timeseries scraped from this config.
    - job_name: "prometheus"

      # metrics_path defaults to '/metrics'
      # scheme defaults to 'http'.

      static_configs:
        - targets: ["localhost:9090", "localhost:3903"]
  ```

  - Start Prometheus with config file

  ```bash
  ./prometheus --config.file prometheus.yml
  ```

  The exported logs from mtail are scraped by prometheus on 3903 port.
