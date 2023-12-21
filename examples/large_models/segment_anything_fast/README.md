
## Segment Anything Fast

[Segment Anything Fast](https://github.com/pytorch-labs/segment-anything-fast) is the optimized version of [Segment Anything](https://github.com/facebookresearch/segment-anything) with 8x performance improvements compared to the original implementation. The improvements were achieved using native PyTorch.

Improvement in speed in achieved using
- Torch.compile: A compiler for PyTorch models
- GPU quantization: Accelerate models with reduced precision operations
- Scaled Dot Product Attention (SDPA): Memory efficient attention implementations
- Semi-Structured (2:4) Sparsity: A GPU optimized sparse memory format
- Nested Tensor: Batch together non-uniformly sized data into a single Tensor, such as images of different sizes.
- Custom operators with Triton: Write GPU operations using Triton Python DSL and easily integrate it into PyTorch’s various components with custom operator registration.

Details on how this is achieved can be found in this [blog](https://pytorch.org/blog/accelerating-generative-ai/)

#### Pre-requisites

Needs python 3.10

`cd` to the example folder `examples/large_models/segment_anything_fast`

Install `Segment Anything Fast` by running
```
chmod +x install_segment_anything_fast.sh
source install_segment_anything_fast.sh
```
Segment Anything Fast needs the nightly version of PyTorch. Hence the script is uninstalling PyTorch, its domain libraries and installing the nightly version of PyTorch.


### Step 1: Download the weights

```
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

If you are not using A100 for inference, turn off the A100 specific optimization using
```
export SEGMENT_ANYTHING_FAST_USE_FLASH_4=0
```

Depending on the available GPU memory, you need to edit the value of `process_batch_size` in `model-config.yaml`
`process_batch_size` is the batch size for the decoding step. Use a smaller value for lower memory footprint.
Higher value will result in faster inference. The following values were tested.

Example:
  - For `A10G` : `process_batch_size=8`
  - For `A100` : `process_batch_size=16`


### Step 2: Generate mar or tgz file

```
torch-model-archiver --model-name sam-fast --version 1.0 --handler custom_handler.py --config-file model-config.yaml --archive-format tgz
```

### Step 3: Add the tgz file to model store

```
mkdir model_store
mv sam-fast.tar.gz model_store
```

### Step 4: Start torchserve

```
torchserve --start --ncs --model-store model_store --models sam-fast.tar.gz
```

### Step 5: Run inference

```
python inference.py
```

results in

![kitten_mask_sam_fast](./kitten_mask_fast.png)
