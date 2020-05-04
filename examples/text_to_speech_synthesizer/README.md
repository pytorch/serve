# Text to speech synthesis using WaveGlow & Tacotron2 model.

**This example works only on NVIDIA CUDA device and not on CPU**

We have used the following Waveglow/Tacotron2 model for this example: 

https://pytorch.org/hub/nvidia_deeplearningexamples_waveglow/

We have copied WaveGlow's model file from following github repo:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/waveglow/model.py


# Install pip dependencies using following commands

```bash
pip install numpy scipy unidecode inflect
pip install librosa --user
```

# Serve the WaveGlow speech synthesis model on TorchServe

 * Generate the model archive for waveglow speech synthesis model using following command
 
    ```bash
    ./create_mar.sh
    ```
   
 * Register the model on TorchServe using the above model archive file and run digit recognition inference
   
    ```bash
    mkdir model_store
    mv waveglow_synthesizer.mar model_store/
    torchserve --start --model-store model_store --models waveglow_synthesizer.mar
    curl -X POST http://127.0.0.1:8080/predictions/waveglow_synthesizer -T sample_text.txt
    ```
  * Response :
  ```text
    [Audio file generated successfully at /tmp/audio.wav]
  ```

Note :

 * Update the post process method to change the output location of the audio file.
