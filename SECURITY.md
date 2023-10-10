# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.8.2   | :white_check_mark: |


## How we do security

TorchServe as much as possible relies on automated tools to do security scanning, in particular we support
1. Dependency Analysis: Using Dependabot
2. Docker Scanning: Using Snyk
3. Code Analysis: Using CodeQL

## Important Security Guidelines

1. TorchServe listens on the following ports
    1. HTTP - `8080`, `8081`, `8082`
    2. gRPC - `7070`, `7071`

    These ports are accessible to `localhost` by default.  The address can be configured by following the [guide](https://github.com/pytorch/serve/blob/master/docs/configuration.md#configure-torchserve-listening-address-and-port)
    TorchServe does not prevent users from configuring the address to be `0.0.0.0`. Please be aware of the security risks if you use `0.0.0.0`
2. TorchServe's Docker image is configured to expose the ports `8080`, `8081`, `8082`, `7070`, `7071` to the host by default(https://github.com/pytorch/serve/blob/master/docker/Dockerfile). When these ports are mapped to the host, make sure to use `localhost` or a specific ip address.

3. Be sure to validate the authenticity of the `.mar` file being used with TorchServe.
    1. A `.mar` file being downloaded from the internet from an untrusted source may have malicious code, compromising the integrity of your application
    2. TorchServe executes arbitrary python code packaged in the `mar` file. Make sure that you've either audited that the code you're using is safe and/or is from a source that you trust
4. By default TorchServe allows you to register models from all URLs. Make sure to set `allowed_urls` parameter in config.properties to restrict this. You can find more details in the [configuration guide](https://github.com/pytorch/serve/blob/master/docs/configuration.md#other-properties)
    - `use_env_allowed_urls=true` is required in config.properties to read `allowed_urls` from environment variable





## Reporting a Vulnerability

If you find a serious vulnerability please report it to opensource@meta.com and torchserve@amazon.com
