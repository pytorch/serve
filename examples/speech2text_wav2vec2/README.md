## Speech2Text Wav2Vec2 example:
In this example we will use a pretrained Wav2Vec2 model for Speech2Text using the `transformers` library: https://huggingface.co/docs/transformers/model_doc/wav2vec2 and serve it using torchserve.

### Prerequisites
Apart from the usual dependencies as shown here: `https://github.com/pytorch/serve/blob/master/docs/getting_started.md`, we need to install `torchaudio` and `transformers`.

You can install these into your current environment or follow these steps which should give you all necessary prerequisites from scratch:
* Install miniconda: https://docs.conda.io/en/latest/miniconda.html
* run `python ../../ts_scripts/install_dependencies.py` to install binary dependencies
* Install all needed packages with `conda env create -f environment.yml`
* Activate conda environment: `conda activate wav2vec2env`

### Prepare model and run server
Next, we need to download our wav2vec2 model and archive it for use by torchserve:
```bash
./download_wav2vec2.py # Downloads model and creates folder `./model` with all necessary files
./archive_model.sh # Creates .mar archive using torch-model-archiver and moves it to folder `./model_store`
```

Now let's start the server and try it out with our example file!
```bash
torchserve --start --model-store model_store --models Wav2Vec2=Wav2Vec2.mar --ncs --disable-token-auth  --enable-model-api
# Once the server is running, let's try it with:
curl -X POST http://127.0.0.1:8080/predictions/Wav2Vec2 --data-binary '@./sample.wav' -H "Content-Type: audio/basic"
# Which will happily return:
I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT%
```
