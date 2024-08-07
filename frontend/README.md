TorchServe REST API endpoint
==============================

## Quick Start

### Building frontend

You can build the frontend using gradle:

```sh
$ cd frontend
$ ./gradlew -p frontend clean assemble
```

And you can build and test the frontend by running

```sh
$ cd frontend
$ ./gradlew build
```

You will find a jar file in frontend/server/build/libs file.

To continuously test your frontend changes during development (e.g. through out pytest integration tests in test/pytest) without continuously reinstalling TS you can create a symbolic link (ln -s) from ts/frontend/model-server.jar to frontend/server/build/libs/server-1.0.jar. That way you changes get picked up (after calling `./gradlew -p frontend clean assemble`) when you start TS.

When you create a PR with your changes it can happen that you see a formatting error during the CI testing.
To fix the format simply run this command and commit the changes:
```sh
$ cd frontend
$ ./gradlew format
```
### Starting frontend

Frontend web server using a configuration file to control the behavior of the frontend web server.
An sample config.properties can be found in frontend/server/src/test/resources/config.properties.
This configure will load a noop model by default. The noop model file is located in frontend/modelarchive/src/test/resources/model/noop-v0.1.model.

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
