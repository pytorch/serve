# GitHub Actions for TorchServe

## Steps to create github actions

1. Create a new workflow file under `.github/workflows`
2. Specify the name of the workflow

    `name: Run Regression Tests on CPU`
3. Specify when the workflow should trigger. Some common examples are as follows

    - Trigger manually
        ```
        on: workflow_dispatch
        ```
    - Trigger on push to a branch
        ```
        on:
            push:
                branches:
                - master
        ```
    - Trigger on pull request
        ```
        on:
          pull_request:
            branches:
            - master
        ```
    - Trigger nightly
        ```
        on:
            # run every day at 2:15am
            schedule:
                - cron:  '15 02 * * *'
        ```

4. Start defining the job to be run
    ```
    jobs:
        regression-cpu:
    ```

    Everything after this is aligned to `jobs`

5. Define where the job is to run
    - Specify which OS/machine to be run on
    ```
        runs-on: ubuntu-20.04
    ```
    ```
        runs-on: [self-hosted, ci-gpu]
    ```
    - Specify in terms of a matrix. This would run on `ubuntu-20.04` and `macOS-latest`. `fail-fast` indicates to fail the job when the first one fails
    ```
        runs-on: ${{ matrix.os }}
        strategy:
        fail-fast: false
        matrix:
            os: [ubuntu-20.04, macOS-latest]
    ```
    -  This would create 2 runs. One run on `ci-gpu` with CUDA 11.6 and a second run on `ci-gpu` with CUDA 11.7
    ```
        runs-on: [self-hosted, ci-gpu]
            strategy:
            fail-fast: false
            matrix:
                cuda: ["cu116", "cu117"]
    ```

6. Specify the commands to be executed for the run

    1. Setup python
    ```
        - name: Setup Python 3.8
            uses: actions/setup-python@v3
            with:
            python-version: 3.8
            architecture: x84
    ```
    2. Setup Java
    ```
        - name: Setup Java 17
            uses: actions/setup-java@v3
            with:
            distribution: 'zulu'
            java-version: '17'
    ```
    3. Checkout TorchServe
    ```
        - name: Checkout TorchServe
            uses: actions/checkout@v3
    ```

    4. Specify the shell commands/scripts to be run. Examples:

        1.
        ```
            - name: Install dependencies
                run: python ts_scripts/install_dependencies.py --environment=dev
        ```
        2.

        ```
            - name: Upload codecov
                if: matrix.os == 'ubuntu-20.04'
                run : |
                curl -Os https://uploader.codecov.io/latest/linux/codecov
                chmod +x codecov
                ./codecov
        ```
