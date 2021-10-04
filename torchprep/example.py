import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
  

def download_bert():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
    torch.save(model.state_dict(), "models/")
    

def download_resnet():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    torch.save(model.state_dict(), "models/")


if __name__ == "__main__":
    download_resnet()
    download_bert()

    
