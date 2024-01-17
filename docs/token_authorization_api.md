# TorchServe token authorization API

## Customer Use
1. Enable token authorization by adding the provided plugin at start using the `--plugin-path` command.
2. Torchserve will enable token authorization if the plugin is provided. In the model server home folder a file `key_file.txt` will be generated.
    1. Example key file:

    `Management Key: aadJv_R6 --- Expiration time: 2024-01-16T22:23:32.952499Z`

    `Inference Key: poZXAlqe --- Expiration time: 2024-01-16T22:23:50.621298Z`

    `API Key: xryL_Vzs`
3. There are 3 keys and each have a different use.
    1. Management key: Used for management apis. Example:
    `curl http://localhost:8081/models/densenet161 -H "Authorization: Bearer aadJv_R6"`
    2. Inference key: Used for inference apis. Example:
    `curl http://127.0.0.1:8080/predictions/densenet161 -T examples/image_classifier/kitten.jpg -H "Authorization: Bearer poZXAlqe"`
    3. API key: Used for the token authorization api. Check section 4 for api use.
    4. 3 tokens allow the owner with the most flexibility in use and enables them to adapt the tokens to their use. Owners of the server can provide users with the inference token if users should not mess with models. The owner can also provide owners with the management key if owners want users to add and remove models.
4. The plugin also includes an api in order to generate a new key to replace either the management or inference key.
    1. Management Example:
    `curl localhost:8081/token?type=management -H "Authorization: Bearer xryL_Vzs"` will replace the current management key in the key_file with a new one and will update the expiration time.
    2. Inference example:
    `curl localhost:8081/token?type=inference -H "Authorization: Bearer xryL_Vzs"`

    Users will have to use either one of the apis above.

5. When users shut down the server the key_file will be deleted.


## Customization
Torchserve offers various ways to customize the token authorization to allow owners to reach the desired result.
1. Time to expiration is set to default at 60 minutes but can be changed in the config.properties by adding `token_expiration`. Ex:`token_expiration=30`
2. The token authorization code is consolidated in the plugin and thus can be changed without impacting the frontend or end result. The only thing the user cannot change is:
    1. The urlPattern for the plugin must be 'token' and the class name must not change
    2. The `generateKeyFile`, `checkTokenAuthorization`, and `setTime` functions return type and signature must not change. However, the code in the functions can be modified depending on user necessity.

## Notes
1. DO NOT MODIFY THE KEY FILE. Modifying the key file might impact reading and writing to the file thus preventing new keys from properly being displayed in the file.
