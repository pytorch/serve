# sockeye-serving
This example shows how to serve Sockeye models for machine translation.
The custom handler is implemented in `sockeye_service.py`.
Since Sockeye has many dependencies, it's convenient to use Docker.
For simplicity, we'll use a pre-trained model and make some assumptions about how we preprocess the data.

## Getting Started With Docker
Pull the latest Docker image:
```bash
docker pull jwoo11/sockeye-serving
```

Download the example model archive file (MAR).
This is a ZIP archive containing the parameter files and scripts needed to run translation for a particular language:
* https://www.dropbox.com/s/pk7hmp7a5zjcfcj/zh.mar?dl=0

Extract the MAR file to `/tmp/models`.
 We'll use this directory as a bind mount for Docker:
```bash
unzip -d /tmp/models/zh zh.mar
```

Start the server:
```bash
docker run -itd --name mms -p 8080:8080 -p 8081:8081 -v /tmp/models:/opt/ml/model jwoo11/sockeye-serving serve
```

Now we can load the model using the management API provided by `mxnet-model-server`:
```bash
curl -X POST "http://localhost:8081/models?synchronous=true&initial_workers=1&url=zh"
```
Get the status of the model with the following:
```bash
curl -X GET "http://localhost:8081/models/zh"
```
```json
{
  "modelName": "zh",
  "modelUrl": "zh",
  "runtime": "python3",
  "minWorkers": 1,
  "maxWorkers": 1,
  "batchSize": 1,
  "maxBatchDelay": 100,
  "workers": [
    {
      "id": "9000",
      "startTime": "2019-01-26T00:49:10.431Z",
      "status": "READY",
      "gpu": false,
      "memoryUsage": 601395200
    }
  ]
}
```

To translate text, use the inference API. Notice that the port is different from above. 
```bash
curl -X POST "http://localhost:8080/predictions/zh" -H "Content-Type: application/json" \
    -d '{ "text": "我的世界是一款開放世界遊戲，玩家沒有具體要完成的目標，即玩家有超高的自由度選擇如何玩遊戲" }'
```

The translation quality depends on the model. Apparently, this one needs more training:
```json
{
  "translation": "in my life was a life of a life of a public public, and a public, a time, a video, a play, which, it was a time of a time of a time."
}
```

For more information on MAR files and the built-in REST APIs, see:
* https://github.com/awslabs/mxnet-model-server/tree/master/docs
