# Caching with Redis database

We will build a minimal working example that uses a Redis server to cache the input/output of a custom handler.

The example will be based on the [MNIST classifier example](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist).

### Pre-requisites

- Redis is installed on your system. Follow the [Redis getting started guide](https://redis.io/docs/getting-started/) to install Redis.

  Start a Redis server using (the server will be started on `localhost:6379` by default):
  ```bash
  redis-server
  # optionally specify the port:
  # redis-server --port 6379
  ```
- The [Python Redis interface](https://github.com/redis/redis-py) is installed:
    ```bash
    pip install redis
    ```

### Using the `ts.utils.redis_cache.handler_cache` decorator

The decorator's usage is similar to that of the built-in `functools.lru_cache`.

A typical usage would be:
```python
from ts.utils.redis_cache import handler_cache

class SomeHandler(BaseHandler):
    def __init__(self):
        ...
        self.handle = handler_cache(host='localhost', port=6379, db=0, maxsize=128)(self.handle)
```
See [mnist_handler_cached.py](https://github.com/pytorch/serve/tree/master/examples/redis_cache/mnist_handler_cached.py) for a minimal concrete example.

### Package and serve the model as usual

Execute commands from the project root:
```bash
torch-model-archiver --model-name mnist --version 1.0 --model-file examples/image_classifier/mnist/mnist.py --serialized-file examples/image_classifier/mnist/mnist_cnn.pt --handler  examples/redis_cache/mnist_handler_cached.py
mkdir -p model_store
mv mnist.mar model_store/
torchserve --start --model-store model_store --models mnist=mnist.mar --ts-config examples/image_classifier/mnist/config.properties
```

Run inference using:
```bash
curl http://127.0.0.1:8080/predictions/mnist -T examples/image_classifier/mnist/test_data/0.png
# The second call will return the cached result
curl http://127.0.0.1:8080/predictions/mnist -T examples/image_classifier/mnist/test_data/0.png
```

### Breif note on performance
The input and output are both serialized (by pickle) before being put into the cache.
The output also needs to be retrieved and deserialized at a cache hit.

If the input and/or output are very large objects, these serialization process might take a while and longer keys take longer to compare.
