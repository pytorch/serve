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
   
 * Register the model on TorchServe using the above model archive file
   
    ```bash
    mkdir model_store
    mv waveglow_synthesizer.mar model_store/
    torchserve --start --model-store model_store --models waveglow_synthesizer.mar
    ```
  * Run inference and download audio output using curl command : 
    ```bash
    curl http://127.0.0.1:8080/predictions/waveglow_synthesizer -T sample_text.txt -o audio.wav
    ```
    
  * Run inference and download audio output using python script :
  
    ```python
    import requests
    
    files = {'data': open('sample_text.txt','rb')}
    response = requests.post('http://localhost:8080/predictions/waveglow_synthesizer', files=files)
    data = response.content
    
    with open('audio.wav', 'wb') as audio_file:
        audio_file.write(data)
    ```
  
  * Change the host and port in above samples as per your server configuration.
  
  * Response :
    An audio.wav file gets downloaded.
    
  **Note :** The above example works only for smaller text size. Refer following NVidia/DeepLearningExamples ticket for more details :
  https://github.com/NVIDIA/DeepLearningExamples/issues/497
