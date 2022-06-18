# How to keep the original root directory of the extra files

In this example, we show how to deal with the extra files that the structure is complicated or contain same repetitive file names.

## Quick start

1. *[Recommended]: For easy access, copy your extra files to one directory. Eg:*
    ```shell
    $ tree -L 3 examples/mar_extra_root
    examples/mar_extra_root
    ├── my_lib1
    │   ├── my_file.py
    │   └── my_module
    │       └── my_file1.py
    ├── my_lib2
    │   ├── my_file.py
    │   └── my_module
    │       └── my_file1.py
    └── ...
    ```
2. Check your handler file as [mnist_handler_extra.py](mnist_handler_extra.py);
    ```python
    from my_lib1.my_file import func as func1
    from my_lib2.my_file import func as func2

    func1()
    func2()
    ...
    ```
    **Note:** need to use absolute imports.

3. Generate the .mar file:
    ```shell
    $ torch-model-archiver \
        --model-name mnist \
        --version 1.0 \
        --model-file      examples/image_classifier/mnist/mnist.py \
        --serialized-file examples/image_classifier/mnist/mnist_cnn.pt \
        --handler         'examples/mar_extra_root/mnist_handler_extra.py' \
        --extra-files     'examples/mar_extra_root/my_lib1,examples/mar_extra_root/my_lib2' \
        --keep-extra-root
    ```

## Normal Mar file

Usually we don't need to append extra files:
```shell
$ torch-model-archiver \
  --model-name mnist \
  --version 1.0 \
  --model-file      examples/image_classifier/mnist/mnist.py \
  --serialized-file examples/image_classifier/mnist/mnist_cnn.pt \
  --handler         examples/image_classifier/mnist/mnist_handler.py
```

The structure of the mar file as follows:
```shell
$ unzip -l mnist.mar    
Archive:  mnist.mar
  Length      Date    Time    Name
---------  ---------- -----   ----
     1273  2022-06-19 01:57   mnist_handler.py
  4800893  2022-06-19 01:57   mnist_cnn.pt
      757  2022-06-19 01:57   mnist.py
      265  2022-06-19 01:57   MAR-INF/MANIFEST.json
---------                     -------
  4803188                     4 files
```


### Without `--keep-extra-root` option

Maybe it works:
```shell
$ torch-model-archiver \
    --model-name mnist \
    --version 1.0 \
    --model-file      examples/image_classifier/mnist/mnist.py \
    --serialized-file examples/image_classifier/mnist/mnist_cnn.pt \
    --handler        'examples/mar_extra_root/mnist_handler_extra.py' \
    --extra-files    'examples/mar_extra_root/my_lib1'
```

The structure of the mar file as follows:
```shell
$ unzip -l mnist.mar
Archive:  mnist.mar
  Length      Date    Time    Name
---------  ---------- -----   ----
     1376  2022-06-19 02:22   mnist_handler_extra.py
  4800893  2022-06-19 02:22   mnist_cnn.pt
       58  2022-06-19 01:32   my_file.py
      757  2022-06-19 02:22   mnist.py
       59  2022-06-19 01:32   my_module/my_file1.py
      271  2022-06-19 02:22   MAR-INF/MANIFEST.json
---------                     -------
  4803414                     6 files
```
*Note: `my_lib1` has been dropped, `my_file.py` and `my_module/` are under the root dir.* 

*Note: This mar file contains `mnist_handler_extra.py`, and it requires `my_lib2` dir to run.* 

### Use `--keep-extra-root` option

Somtimes we don't want to break the existing structure in extra files. Then work with the `--keep-extra-root`.

```shell
$ torch-model-archiver \
    --model-name mnist \
    --version 1.0 \
    --model-file      examples/image_classifier/mnist/mnist.py \
    --serialized-file examples/image_classifier/mnist/mnist_cnn.pt \
    --handler         'examples/mar_extra_root/mnist_handler_extra.py' \
    --extra-files     'examples/mar_extra_root/my_lib1,examples/mar_extra_root/my_lib2' \
    --keep-extra-root
```

```shell
$ unzip -l mnist.mar
Archive:  mnist.mar
  Length      Date    Time    Name
---------  ---------- -----   ----
     1376  2022-06-19 02:06   mnist_handler_extra.py
  4800893  2022-06-19 02:06   mnist_cnn.pt
      757  2022-06-19 02:06   mnist.py
       58  2022-06-19 01:32   my_lib2/my_file.py
       59  2022-06-19 01:32   my_lib2/my_module/my_file1.py
       58  2022-06-19 01:32   my_lib1/my_file.py
       59  2022-06-19 01:32   my_lib1/my_module/my_file1.py
      271  2022-06-19 02:06   MAR-INF/MANIFEST.json
---------                     -------
  4803531                     8 files
```

**WARN!** If we don't modify files and run command with `my_lib1` and `my_lib2`, error may occur:
```shell
$ torch-model-archiver \
    --model-name mnist \
    --version 1.0 \
    --model-file      examples/image_classifier/mnist/mnist.py \
    --serialized-file examples/image_classifier/mnist/mnist_cnn.pt \
    --handler         examples/mar_extra_root/mnist_handler_extra.py \
    --extra-files     'examples/mar_extra_root/my_lib1,examples/mar_extra_root/my_lib2'
...
FileExistsError: [Errno 17] File exists: '/tmp/mnist/my_module'
```
## Refer

- [MNIST](https://github.com/pytorch/serve/blob/master/examples/image_classifier/mnist/README.md).
