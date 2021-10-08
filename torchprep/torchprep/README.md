# Torchprep

A CLI tool to prepare your Pytorch models for efficient inference. The only prerequisite is a model trained and saved with `torch.save(model_name, model_path)`. See `example.py` for an example.

## Install from source

```sh
pip install -r requirements.txt
cd torchprep
poetry install
```

## Install from Pypi

```sh
pip install torchprep
```

## Usage

```sh
torchprep quantize --help
```

### Example

```sh
python example.py
torchprep quantize models/resnet152.pt int8 --profile 64,3,7,7
```

## Uploading to Pypi

### Create binaries

To create binaries and test them out locally

```sh
poetry build
pip install --user /path/to/wheel
```

### Upload to Pypi

```sh
poetry config pypi-token.pypi <SECRET_KEY>
poetry publish --build
```

## Coming soon
* Reduce parameter count by 1/3 `torchprep distill model.pt 1/3`
* Automated release with github actions