# How to load a model in TorchServe

There are multiple ways to load to model in TorchServe. The below flowchart tries to simplify the process and shows the various options


```mermaid
flowchart TD
    id1[[How to load a model in TorchServe?]] --> id13{Handler has an initialize method?}
    id13{Handler has an initialize method?} -- No, using BaseHandler initialize method --> id2{Model Type?} --> id3(PyTorch Eager)  & id4(TorchScripted) & id5(ONNX) & id6(TensorRT)
    id3(PyTorch Eager) --Required--> id7(Model File & weights file)
    id4(TorchScripted) --Required--> id8(TorchScripted weights ending in '.pt')
    id5(ONNX) --Required --> id9(Weights ending in '.onnx')
    id6(TensorRT) --Required--> id10(TensorRT weights ending in '.pt')
    id7(Model File & weights file) & id8(TorchScripted weights ending in '.pt') &  id9(Weights ending in '.onnx') & id10(TensorRT weights ending in '.pt') --> id11(Created a model archive .mar file)
    id13{Handler has an initialize method?} --Yes--> id21{"Does the initialize method inherit from BaseHandler?"}
    id21{"Does the initialize method inherit from BaseHandler?"} -- Yes --> id2{Model Type?}
    id21{Does the initialize method inherit from BaseHandler?} -- No --> id20("Create a custom method to
         load the model in the handler") --> id11(Create a model archive .mar file)
    id15["Pass the weights with --serialized-file option
    - Completely packaged for production/reproducibility
    - Model archiving and model loading can be slow for large models"]
    id16["Pass the path to the weights in model-config.yaml
    - Extremely fast to create model archive
    - You can use defered initialization for large models
    - Model loading can be faster for large models
    - Model management can be harder"]
	id11(Create a model archive .mar file) --> id14{"Is your model large?
	Do you care about model archiving and loading time?"} --No--> id15
	id14{"Is your model large?
	Do you care about model archiving and loading time?"} --yes to either--> id16
	id15 & id16 --> id17[Start TorchServe with mar file]
	id15 & id16 --> id18[Start TorchServe] --> id19[Register Model with mar file]


```
