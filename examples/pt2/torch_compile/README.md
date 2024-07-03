
# TorchServe inference with torch.compile of densenet161 model

This example shows how to take eager model of `densenet161`, configure TorchServe to use `torch.compile` and run inference using `torch.compile`


### Pre-requisites

- `PyTorch >= 2.0`

Change directory to the examples directory
Ex:  `cd  examples/pt2/torch_compile`


### torch.compile config

`torch.compile` supports a variety of config and the performance you get can vary based on the config. You can find the various options [here](https://pytorch.org/docs/stable/generated/torch.compile.html)

In this example , we use the following config

```
echo "pt2:
  compile:
    enable: True" > model-config.yaml
```

### Create model archive

```
wget https://download.pytorch.org/models/densenet161-8d451a50.pth
mkdir model_store
torch-model-archiver --model-name densenet161 --version 1.0 --model-file model.py --serialized-file densenet161-8d451a50.pth --export-path model_store --extra-files ../../image_classifier/index_to_name.json --handler image_classifier --config-file model-config.yaml -f
```

#### Start TorchServe
```
torchserve --start --ncs --model-store model_store --models densenet161.mar --disable-token-auth  --enable-model-api
```

#### Run Inference

```
curl http://127.0.0.1:8080/predictions/densenet161 -T ../../image_classifier/kitten.jpg
```

produces the output

```
{
  "tabby": 0.4664836823940277,
  "tiger_cat": 0.4645617604255676,
  "Egyptian_cat": 0.06619937717914581,
  "lynx": 0.0012969186063855886,
  "plastic_bag": 0.00022856894065625966
}
```

### Performance improvement from using `torch.compile`

To measure the handler `preprocess`, `inference`, `postprocess` times, run the following

#### Measure inference time with PyTorch eager

```
echo "handler:" > model-config.yaml && \
echo "  profile: true" >> model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
After a few iterations of warmup, we see the following

```
2024-02-03T00:54:31,136 [INFO ] W-9000-densenet161_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:6.118656158447266|#ModelName:densenet161,Level:Model|#type:GAUGE|#hostname:ip-172-31-11-40,1706921671,c02b3170-c8fc-4396-857d-6c6266bf94a9, pattern=[METRICS]
2024-02-03T00:54:31,155 [INFO ] W-9000-densenet161_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:18.77564811706543|#ModelName:densenet161,Level:Model|#type:GAUGE|#hostname:ip-172-31-11-40,1706921671,c02b3170-c8fc-4396-857d-6c6266bf94a9, pattern=[METRICS]
2024-02-03T00:54:31,155 [INFO ] W-9000-densenet161_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.16630400717258453|#ModelName:densenet161,Level:Model|#type:GAUGE|#hostname:ip-172-31-11-40,1706921671,c02b3170-c8fc-4396-857d-6c6266bf94a9, pattern=[METRICS]
```

#### Measure inference time with `torch.compile`

```
echo "pt2:
  compile:
    enable: True
    backend: inductor
    mode: reduce-overhead" > model-config.yaml && \
echo "handler:
  profile: true" >> model-config.yaml
```

Once the `yaml` file is updated, create the model-archive, start TorchServe and run inference using the steps shown above.
`torch.compile` needs a few inferences to warmup. Once warmed up, we see the following
```
2024-02-03T00:56:14,808 [INFO ] W-9000-densenet161_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_preprocess.Milliseconds:5.9771199226379395|#ModelName:densenet161,Level:Model|#type:GAUGE|#hostname:ip-172-31-11-40,1706921774,d38601be-6312-46b4-b455-0322150509e5, pattern=[METRICS]
2024-02-03T00:56:14,814 [INFO ] W-9000-densenet161_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_inference.Milliseconds:5.8818559646606445|#ModelName:densenet161,Level:Model|#type:GAUGE|#hostname:ip-172-31-11-40,1706921774,d38601be-6312-46b4-b455-0322150509e5, pattern=[METRICS]
2024-02-03T00:56:14,814 [INFO ] W-9000-densenet161_1.0-stdout org.pytorch.serve.wlm.WorkerLifeCycle - result=[METRICS]ts_handler_postprocess.Milliseconds:0.19392000138759613|#ModelName:densenet161,Level:Model|#type:GAUGE|#hostname:ip-172-31-11-40,1706921774,d38601be-6312-46b4-b455-0322150509e5, pattern=[METRICS]
```

### Conclusion

`torch.compile` reduces the inference time from 18ms to 5ms
