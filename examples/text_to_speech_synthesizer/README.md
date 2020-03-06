# Text to speech synthesis using WaveGlow & Tacotron2 model.

We have used the following Waveglow/Tacotron2 model for this example: 

https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/

We have copied WaveGlow's model file from following github repo:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/waveglow/model.py


# Install pip dependencies using following commands

```bash
pip install numpy scipy librosa unidecode inflect
pip install librosa --user
```

# Serve the WaveGlow speech synthesis model on TorchServe

 * Download the checkpoint for NVIDIA WaveGlow model :
 
    ```bash
   https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth 
   ```

 * Create a torch model archive using the torch-model-archiver utility to archive the above files.
 
    ```bash
    torch-model-archiver --model-name waveglow_synthesizer --version 1.0 --model-file waveglow_model.py --serialized-file nvidia_waveglowpyt_fp32_20190306.pth --handler waveglow_handler.py
    ```
   
 * Register the model on TorchServe using the above model archive file and run digit recognition inference
   
    ```bash
    mkdir model_store
    mv waveglow_synthesizer.mar model_store/
    torchserve --start --model-store model_store --models waveglow_synthesizer.mar
    curl -X POST http://127.0.0.1:8080/predictions/waveglow_synthesizer -T sample.txt
    ```
  * Response :
  ```text
    [Audio file generated successfully at /tmp/audio.wav]
  ```

Note :

 * This example works only on NVIDIA CUDA device
 * Update the post process method to change the output location of the audio file.
