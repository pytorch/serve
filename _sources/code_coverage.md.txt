#### To execute unit testing and generate code coverage report follow these steps:

```bash
cd serve/frontend
./gradlew clean build
```

The above command executes the TorchServe frontend build suite which consists of the following :

* checkstyle
* findbugs
* PMD
* UT

The reports can be accessed at the following path :

```
serve/frontend/server/build/reports
```