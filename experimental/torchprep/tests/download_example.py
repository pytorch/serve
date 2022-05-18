import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torchvision
import os
  

def download_bert():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
    torch.save(model, "models/bert.pt")
    

def download_resnet():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
    torch.save(model, "models/resnet152.pt")


def main():
    if not os.path.exists("models"):
        os.makedirs("models")

    if len(os.listdir("models")) == 0:
        download_resnet()
        download_bert()
    else:
        print("models directory is not empty")


if __name__ == "__main__":
    main()