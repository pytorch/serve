#!/bin/bash

# torchserve --start --ncs --ts-config /home/model-server/config.properties

curl -X POST "http://localhost:8081/models?url=rnnt.mar";
curl -X PUT "http://localhost:8081/models/rnnt?min_worker=1"