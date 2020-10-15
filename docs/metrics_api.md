# Prometheus Metrics Plugin Endpoint

TorchServe does not provide any inbuilt metrics API, however a metrics API which returns Prometheus formatted metrics is added as a part of the plugin.
This plugin can be extended by the users or a new metrics endpoint API plugin can also be added. For more details see [here](metrics_plugin.md).

## How to use Prometheus plugin endpoint?
1. Build jar of the plugin.
2. Add the jar to Java classpath manually or specify the "plugins_path" property in config.properties for torchserve start command.
3. Use metrics API to get metrics in Prometheus format  

**Note:** In TorchServe, the endpoints are divided in three categories Inference, Management and Metrics. 
The Prometheus Metric plugin is added as Metric Endpoint. Metric Endpoints listen on port 8082 and only accessible from localhost by default. 
By default these Metric endpoints are enabled however same can be disabled by setting `enable_metrics_endpoints=false` 
in torchserve config.properties file. To change the default setting, see [TorchServe Configuration](configuration.md).  


```shell script
cd ${TORCHSEVE_HOME}

# to run the test cases in plugins
plugins/gradlew -p plugins test

# use command below to generate jars in ${TORCHSEVE_HOME}plugins/build/plugins folder
plugins/gradlew -p plugins clean bS

# OR use command below
python setup.py  build_plugins -p endpoints

# create a config.properties file and add plugins_path=${TORCHSEVE_HOME}/plugins/build/plugins
vi config.properties

# start the server
torchserve --start --ts-config config.properties --model-store frontend/modelarchive/src/test/resources/models

# Fire same APIs to generate metric values
curl -X POST "http://localhost:8081/models?url=noop.mar"
curl -X PUT "http://localhost:8081/models/noop?min_worker=3&synchronous=true"
curl http://localhost:8080/predictions/noop

```

Use command as shown below to get the metrics:


```console

curl -X GET http://localhost:8082/metrics
# HELP ts_queue_latency_milliseconds Cumulative Queue duration in milliseconds
# TYPE ts_queue_latency_milliseconds histogram
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="0.005",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="0.01",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="0.025",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="0.05",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="0.075",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="0.1",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="0.25",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="0.5",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="0.75",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="1.0",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="2.5",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="5.0",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="7.5",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="10.0",} 3.0
ts_queue_latency_milliseconds_bucket{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",le="+Inf",} 3.0
ts_queue_latency_milliseconds_count{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",} 3.0
ts_queue_latency_milliseconds_sum{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",} 0.0
# HELP memory_used System Memory used
# TYPE memory_used gauge
memory_used{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",} 3302.43359375
```

```console
curl -X GET "http://localhost:8082/metrics?name[]=ts_queue_latency_milliseconds_sum&name[]=memory_used" --globoff
# HELP ts_queue_latency_milliseconds Cumulative Queue duration in milliseconds
# TYPE ts_queue_latency_milliseconds histogram
ts_queue_latency_milliseconds_sum{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",model_name="noop",model_version="null",} 0.0
# HELP memory_used System Memory used
# TYPE memory_used gauge
memory_used{uuid="b5b23535-9d55-4564-8865-4fda3ce8aa2c",} 3540.0
```

#### Prometheus server

To view these metrics on a Prometheus server, download and install using the instructions [here](https://prometheus.io/download/#prometheus). Create a minimal `prometheus.yml` config file as below and run `./prometheus --config.file=prometheus.yml`.

```yaml
global:
  scrape_interval:     15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
    - targets: ['localhost:9090']
  - job_name: 'torchserve'
    static_configs:
    - targets: ['localhost:8082'] #TorchServe metrics endpoint
```
Navigate to http://localhost:9090/ on a browser to execute queries and create graphs 

<img width="1231" alt="PrometheusServer" src="https://user-images.githubusercontent.com/880376/86984450-806fc680-c143-11ea-9ae2-f2ef42f24f4c.png">

#### Grafana

Once you have the Torchserve and Prometheus servers running, you can further [setup](https://prometheus.io/docs/visualization/grafana/) Grafana, point it to Prometheus server and navigate to http://localhost:3000/ to create dashboards and graphs.

You can use command given below to start Grafana - 
`sudo systemctl daemon-reload && sudo systemctl enable grafana-server && sudo systemctl start grafana-server`

<img width="1220" alt="Screen Shot 2020-07-08 at 5 51 57 PM" src="https://user-images.githubusercontent.com/880376/86984550-c4fb6200-c143-11ea-9434-09d4d43dd6d4.png">
