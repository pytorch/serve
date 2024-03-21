# Security Policy

## Supported Versions

| Version | Supported          |
|---------| ------------------ |
| 0.9.0   | :white_check_mark: |
| 0.10.0  | :white_check_mark: |


## How we do security

TorchServe as much as possible relies on automated tools to do security scanning, in particular we support
1. Dependency Analysis: Using Dependabot
2. Docker Scanning: Using Snyk
3. Code Analysis: Using CodeQL

## Important Security Guidelines

1. TorchServe listens on the following ports
    1. HTTP - `8080`, `8081`, `8082`
    2. gRPC - `7070`, `7071`

    These ports are accessible to `localhost` by default.  The address can be configured by following the [guide](https://pytorch.org/serve/configuration.html#configure-torchserve-listening-address-and-port)
    TorchServe does not prevent users from configuring the address to be `0.0.0.0`. Please be aware of the security risks if you use `0.0.0.0`
2. TorchServe's Docker image is configured to expose the ports `8080`, `8081`, `8082`, `7070`, `7071` to the host by [default](https://github.com/pytorch/serve/blob/master/docker/Dockerfile). When these ports are mapped to the host, make sure to use `localhost` or a specific ip address.

3. Be sure to validate the authenticity of the `.mar` file being used with TorchServe.
    1. A `.mar` file being downloaded from the internet from an untrustworthy source may have malicious code, compromising the integrity of your application
    2. TorchServe executes arbitrary python code packaged in the `mar` file. Make sure that you've either audited that the code you're using is safe and/or is from a source that you trust
4. By default TorchServe allows you to register models from all URLs. Make sure to set `allowed_urls` parameter in config.properties to restrict this. You can find more details in the [configuration guide](https://pytorch.org/serve/configuration.html#other-properties)
    - `use_env_allowed_urls=true` is required in config.properties to read `allowed_urls` from environment variable
5. Enable SSL:

    TorchServe supports two ways to configure SSL:
    1. Using a keystore
    2. Using private-key/certificate files

    You can find more details in the [configuration guide](https://pytorch.org/serve/configuration.html#enable-ssl)
6. TorchServe supports token authorization: check [documentation](https://github.com/pytorch/serve/blob/master/docs/token_authorization_api.md) for more information. 




## Reporting a Vulnerability

If you find a serious vulnerability please report it to opensource@meta.com and torchserve@amazon.com
