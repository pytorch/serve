# TorchServe token authorization API

## Setup
1. Download the jar files from [Maven](https://mvnrepository.com/artifact/org.pytorch/torchserve-endpoint-plugin) 
2. Enable token authorization by adding the `--plugins-path /path/to/the/jar/files` flag at start up with the path leading to the downloaded jar files.

## Configuration
1. Torchserve will enable token authorization if the plugin is provided. Expected log statement `[INFO ] main org.pytorch.serve.servingsdk.impl.PluginsManager - Loading plugin for endpoint token`
2. In the current working directory a file `key_file.json` will be generated.
    1. Example key file:

```python
  {
  "management": {
    "key": "B-E5KSRM",
    "expiration time": "2024-02-16T21:12:24.801167Z"
  },
  "inference": {
    "key": "gNRuA7dS",
    "expiration time": "2024-02-16T21:12:24.801148Z"
  },
  "API": {
    "key": "yv9uQajP"
  }
}
```

3. There are 3 keys and each have a different use.
    1. Management key: Used for management APIs. Example:
    `curl http://localhost:8081/models/densenet161 -H "Authorization: Bearer I_J_ItMb"`
    2. Inference key: Used for inference APIs. Example:
    `curl http://127.0.0.1:8080/predictions/densenet161 -T examples/image_classifier/kitten.jpg -H "Authorization: Bearer FINhR1fj"`
    3. API key: Used for the token authorization API. Check section 4 for API use.
4. The plugin also includes an API in order to generate a new key to replace either the management or inference key.
    1. Management Example:
    `curl localhost:8081/token?type=management -H "Authorization: Bearer m4M-5IBY"` will replace the current management key in the key_file with a new one and will update the expiration time.
    2. Inference example:
    `curl localhost:8081/token?type=inference -H "Authorization: Bearer m4M-5IBY"`

    Users will have to use either one of the APIs above.

5. When users shut down the server the key_file will be deleted.


## Customization
Torchserve offers various ways to customize the token authorization to allow owners to reach the desired result.
1. Time to expiration is set to default at 60 minutes but can be changed in the config.properties by adding `token_expiration_min`. Ex:`token_expiration_min=30`
2. The token authorization code is consolidated in the plugin and thus can be changed without impacting the frontend or end result. The only thing the user cannot change is:
    1. The urlPattern for the plugin must be 'token' and the class name must not change
    2. The `generateKeyFile`, `checkTokenAuthorization`, and `setTime` functions return type and signature must not change. However, the code in the functions can be modified depending on user necessity.

## Notes
1. DO NOT MODIFY THE KEY FILE. Modifying the key file might impact reading and writing to the file thus preventing new keys from properly being displayed in the file.
2. 3 tokens allow the owner with the most flexibility in use and enables them to adapt the tokens to their use. Owners of the server can provide users with the inference token if users should not mess with models. The owner can also provide owners with the management key if owners want users to add and remove models.
