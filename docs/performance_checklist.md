# Model Inference Optimization Checklist

This checklist describes some steps that should be completed when diagnosing model inference performance issues.  Some of these suggestions are only applicable to NLP models (e.g., ensuring the input is not over-padded and sequence bucketing), but the general principles are useful for other models too.

## General System Optimizations

- Check the versions of PyTorch, Nvidia driver, and other components and update to the latest compatible releases.  Oftentimes known performance bugs have already been fixed.

- Collect system-level activity logs to understand the overall resource utilizations. It’s useful to know how the model inference pipeline is using the system resources at a high level, as the first step of optimization.  Even simple CLI tools such as nvidia-smi and htop would be helpful.

- Start with a target with the highest impact on performance.  It should be obvious from the system activity logs where the biggest bottleneck is – look beyond model inference, as pre/post processing can be expensive and can affect the end-to-end throughput just as much.

- Quantify and mitigate the influence of slow I/O such as disk and network on end-to-end performance.  While optimizing I/O is out of scope for this checklist, look for techniques that use async, concurrency, pipelining, etc. to effectively “hide” the cost of I/O.

- For model inference on input sequences of dynamic length (e.g., transformers for NLP), make sure the tokenizer is not over-padding the input.  If a transformer was trained with padding to a constant length (e.g., 512) and deployed with the same padding, it would run unnecessarily slow (orders of magnitude) on short sequences.

- Vision models with input in JPEG format often benefit from faster JPEG decoding on CPU such as libjpeg-turbo and Pillow-SIMD, and on GPU such as torchvision.io.decode_jpeg and Nvidia DALI.
As this [example](https://colab.research.google.com/drive/1NMaLS8PG0eYhbd8IxQAajXgXNIZ_AvHo?usp=sharing) shows, Nvidia DALI is about 20% faster than torchvision, even on an old K80 GPU.

## Model Inference Optimizations

Start model inference optimization only after other factors, the “low-hanging fruit”, have been extensively evaluated and addressed.

- Use fp16 for GPU inference.  The speed will most likely more than double on newer GPUs with tensor cores, with negligible accuracy degradation.  Technically fp16 is a type of quantization but since it seldom suffers from loss of accuracy for inference it should always be explored. As shown in this [article](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#abstract), use of fp16 offers speed up in large neural network applications.

- Use model quantization (i.e. int8) for CPU inference.  Explore different quantization options: dynamic quantization, static quantization, and quantization aware training, as well as tools such as Intel Neural Compressor that provide more sophisticated quantization methods. It is worth noting that quantization comes with some loss in accuracy and might not always offer significant speed up on some hardware thus this might not always be the right approach.

- Balance throughput and latency with smart batching.  While meeting the latency SLA try larger batch sizes to increase the throughput.

- Try optimized inference engines such as onnxruntime, tensorRT, lightseq, ctranslate-2, etc.  These engines often provide additional optimizations such as operator fusion, in addition to model quantization.

- Try model distillation.  This is more involved and often requires training data, but the potential gain can be large.  For example, MiniLM achieves 99% the accuracy of the original BERT base model while being 2X faster.

- If working on CPU, you can try core pinning. You can find more information on how to work with this [in this blog post](https://pytorch.org/tutorials/intermediate/torchserve_with_ipex#grokking-pytorch-intel-cpu-performance-from-first-principles).

- For batch processing on sequences with different lengths, sequence bucketing could potentially improve the throughput by 2X.  In this case, a simple implementation of sequence bucketing is to sort all input by sequence length before feeding them to the model, as this reduces unnecessary padding when batching the sequences.

While this checklist is not exhaustive, going through the items will likely help you squeeze more performance out of your model inference pipeline.
