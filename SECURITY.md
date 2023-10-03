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
2. TorchServe's Docker image is configured to listen to `localhost` by [default](https://github.com/pytorch/serve/blob/master/docker/config.properties)





## Reporting a Vulnerability

If you find a serious vulnerability please report it to opensource@meta.com and torchserve@amazon.com
