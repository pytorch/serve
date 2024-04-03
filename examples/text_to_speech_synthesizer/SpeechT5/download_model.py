from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")

model.save_pretrained(save_directory="model_artifacts/model")
processor.save_pretrained(save_directory="model_artifacts/processor")
vocoder.save_pretrained(save_directory="model_artifacts/vocoder")
embeddings_dataset.save_to_disk("model_artifacts/speaker_embeddings")
print("Save model artifacts to directory model_artifacts")
