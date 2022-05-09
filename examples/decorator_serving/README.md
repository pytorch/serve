# Decorator Based Serving

The traditional way of serving models in torchserve is to follow instructions from the [getting started guide](https://github.com/pytorch/serve/blob/master/docs/getting_started.md)

Traditionally this includes a few steps
1. Create a model `handler.py`
2. Package your model into a `.mar` file
3. Setup a `config.properties`
4. Call `torchserve --start`

This makes it challenging to leverage torchserve in notebook look like environments since the workflow is entirely done within shell files.

As an experimental alternative we're introducing decorator based model serving where in the same file you can create a minimal handler. At a bare minimum you need an `inference()` function in some `Handler`

```python
from ts.torch_handler.base_handler import BaseHandler

@serve
class YourHandler(BaseHandler):
    def inference():
        # Run your code here
```

`@serve` here will make a local copy of the handler class to your disk and package up the model and start torchserve with some reasonable defaults

However you can also do any sort of advanced configuration you're used to doing from your `config.properties` also directly from a decorator

For example let's say you'd like to overwrite the default serving endpoint you can just run

```python        
@serve(inference_http_port=8080, management_port=8081)
class YourHandler(BaseHandler):
    def inference():
        # Run your code here
```

But you can also easily change model specific configurations like batch size just as easily

```python        
@serve(inference_http_port=8080, management_port=8081, batch_size=32)
class YourHandler(BaseHandler):
    def inference():
        # Run your code here
```



For the full list of supported configs you can check out `serve/ts/utils/serve_decorator.py`