### ASR (Automated Speech Recognition) Example

In this example we use torchserve to serve a ASR model that convert wav to text.  

#### Steps to run:
- save asr model to jit format.
```bash
./00_save_jit_model.sh 
```
- create model archive
```bash
./01_create_model_archive.sh
```
- configure model server
```bash
./02_configure_server.sh
```

- get prediction results
```
python3 03_predict.py
```
