# pylint: disable=W0622, W0223
# "input" is a built-in, but it's the name used by torch
"""
Simple feed-forward model used only to test BaseHandler
"""

import torch

class ArgmaxModel(torch.nn.Module):
    def forward(self, *input):
        return torch.argmax(input[0], 1)

if __name__ == '__main__':
    model = ArgmaxModel()
    torch.save(model.state_dict(), 'base_model.pt')
