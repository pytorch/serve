# Torchscript Support

[Torchscript](https://pytorch.org/docs/stable/jit.html#creating-torchscript-code) along with Pytorch JIT are designed to provide portability and performance for Pytorch models. Torchscript is a static subset of Python language that capture the structure of Pytorch programs and JIT uses this structure for optimization.

Torchscript exposes two APIs, script and trace, using any of these APIs, on the regular Pytorch model developed in python, compiles it to Torchscript. The resulted Torchscript can be loaded in a process where there is no Python dependency. The important difference between trace and script APIs, is that trace does not capture parts of the model which has data dependency such as control flow, this is where script is a better choice.

To create Torchscript from Huggingface Transformers, torch.jit.trace() will be used that returns an executable or [`ScriptFunction`](https://pytorch.org/docs/stable/jit.html#torch.jit.ScriptFunction) that will be optimized using just-in-time compilation. We need to provide example inputs, torch.jit.trace, will record the operations performed on all the tensors when running the inputs through the transformer models. This option can be chosen through the setup_config.json by setting *save_mode* : "torchscript". We need to keep this in mind, as torch.jit.trace()  record operations on tensors,  the size of inputs should be the same both in tracing and when using it for inference, otherwise it will raise an error. Also, there is torchscript flag that needs to be set when setting the configs to load the pretrained models, you can read more about it in this [Huggingface's doc](https://huggingface.co/docs/transformers/torchscript).

Here is how Huggingface transfomers can be converted to Torchscript using the trace API, this has been shown in download_Transformer_models.py as well:

First of all when setting the configs, the torchscript flag should be set :

`config = AutoConfig.from_pretrained(pretrained_model_name,torchscript=True)`

When the model is loaded, we need a dummy input to pass it through the model and record the operations using the trace API:

```
dummy_input = "This is a dummy input for torch jit trace"
inputs = tokenizer.encode_plus(dummy_input,max_length = int(max_length),pad_to_max_length = True, add_special_tokens = True, return_tensors = 'pt')
input_ids = inputs["input_ids"]
traced_model = torch.jit.trace(model, [input_ids])
torch.jit.save(traced_model,os.path.join(NEW_DIR, "traced_model.pt"))
```

## Torchscript model packaging

In case of using Torchscript the packaging step would look like the following:

```
torch-model-archiver --model-name BERTSeqClassification_Torchscript --version 1.0 --serialized-file Transformer_model/traced_model.pt --handler ./Transformer_handler_generalized.py --extra-files "./setup_config.json,./Seq_classification_artifacts/index_to_name.json"

```

And then follow the remaining instructions in [README.md](README.md)
