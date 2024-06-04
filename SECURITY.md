# Security Policy

## Supported Versions

| Version | Supported          |
|---------| ------------------ |
| 0.11.0   | :white_check_mark: |


## How we do security


As much as possible, TorchServe relies on automated tools to do security scanning. In particular, we support:

1. Dependency Analysis: Using Dependabot
2. Docker Scanning: Using Snyk
3. Code Analysis: Using CodeQL

## Important Security Guidelines

1. TorchServe listens on the following ports
    1. HTTP - `8080`, `8081`, `8082`
    2. gRPC - `7070`, `7071`

    These ports are accessible to `localhost` by default.  The addresses can be configured by following the guides for
    [HTTP](https://github.com/pytorch/serve/blob/master/docs/configuration.md#configure-torchserve-listening-address-and-port) and
    [gRPC](https://github.com/pytorch/serve/blob/master/docs/configuration.md#configure-torchserve-grpc-listening-addresses-and-ports).
    TorchServe does not prevent users from configuring the address to be of any value, including the wildcard address `0.0.0.0`.
    Please be aware of the security risks of configuring the address to be `0.0.0.0`, this will give all addresses(including publicly accessible addresses, if any)
    on the host, access to the TorchServe endpoints listening on the ports shown above.
2. By [default](https://github.com/pytorch/serve/blob/master/docker/Dockerfile), TorchServe's Docker image is configured to expose the ports `8080`, `8081`, `8082`, `7070`, `7071` to the host. When starting the container,
   map the ports exposed by the container to `localhost` ports or a specific IP address, as shown in this [security guideline](https://github.com/pytorch/serve/blob/master/docker/README.md#security-guideline).

3. Be sure to validate the authenticity of the `.mar` file being used with TorchServe.
    1. A `.mar` file being downloaded from the internet from an untrustworthy source may have malicious code, compromising the integrity of your application.
    2. TorchServe executes the arbitrary python code packaged in the `mar` file. Make sure that you've either audited that the code you're using is safe and/or is from a source that you trust.
    3. TorchServe supports custom [plugins](https://github.com/pytorch/serve/tree/master/plugins) and [handlers](https://github.com/pytorch/serve/blob/master/docs/custom_service.md).
       These can be utilized to extend TorchServe functionality to perform runtime security scanning using tools such as:
        - Clamd: https://pypi.org/project/clamd/
        - VirusTotal: https://virustotal.github.io/vt-py/
        - Fickling: https://github.com/trailofbits/fickling
    4. Running TorchServe inside a container environment and loading an untrusted `.mar` file does not guarantee isolation from a security perspective.
4. By default, TorchServe allows you to register models from all URLs. Make sure to set `allowed_urls` parameter in config.properties to restrict this. You can find more details in the [configuration guide](https://pytorch.org/serve/configuration.html#other-properties).
    - `use_env_allowed_urls=true` is required in config.properties to read `allowed_urls` from environment variable.
5. Enable SSL:

    TorchServe supports two ways to configure SSL:
    1. Using a keystore
    2. Using private-key/certificate files

    You can find more details in the [configuration guide](https://pytorch.org/serve/configuration.html#enable-ssl).
6. Prepare your model against bad inputs and prompt injections. Some recommendations:
    1. Pre-analysis: check how the model performs by default when exposed to prompt injection (e.g. using [fuzzing for prompt injection](https://github.com/FonduAI/awesome-prompt-injection?tab=readme-ov-file#tools)).
    2. Input Sanitation: Before feeding data to the model, sanitize inputs rigorously. This involves techniques such as:
        - Validation: Enforce strict rules on allowed characters and data types.
        - Filtering: Remove potentially malicious scripts or code fragments.
        - Encoding: Convert special characters into safe representations.
        - Verification: Run tooling that identifies potential script injections (e.g. [models that detect prompt injection attempts](https://python.langchain.com/docs/guides/safety/hugging_face_prompt_injection)).
7. If you intend to run multiple models in parallel with shared memory, it is your responsibility to ensure the models do not interact or access each other's data. The primary areas of concern are tenant isolation, resource allocation, model sharing and hardware attacks.
8. TorchServe supports token authorization: check [documentation](https://github.com/pytorch/serve/blob/master/docs/token_authorization_api.md) for more information.

## Reporting a Vulnerability

If you find a vulnerability please report it to https://www.facebook.com/whitehat and aws-security@amazon.com
