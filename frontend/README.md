TorchServe REST API endpoint
==============================

## Quick Start

### Building frontend

You can build frontend using gradle:

```sh
$ cd frontend
$ ./gradlew build
```

You will find a jar file in frontend/server/build/libs file.

### Starting frontend

Frontend web server using a configuration file to control the behavior of the frontend web server.
A sample config.properties can be found in frontend/server/src/test/resources/config.properties.
This configuration will load a noop workflow by default. The noop workflow file is located at frontend/archive/src/test/resources/workflow/noop-v0.1.workflow.

#### Start Query service:

```sh
cd frontend/server
../gradlew startServer
```

#### Stop Query service:
```sh
cd frontend/server
../gradlew killServer
```
