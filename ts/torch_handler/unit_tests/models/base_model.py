import torch

class ArgmaxModel(torch.nn.Module):
    def forward(self, x):
        return torch.argmax(x, 1)

if __name__ == '__main__':
    model = ArgmaxModel()
    torch.save(model.state_dict(), 'base_model.pt')
