# pylint: disable=W0622, W0223
# "input" is a built-in, but it's the name used by torch
"""
A model that returns its input
"""

import torch

class IdentityModel(torch.nn.Module):
    def forward(self, *input):
        return input[0]

if __name__ == '__main__':
    model = IdentityModel()
    torch.save(model.state_dict(), 'identity_model.pt')
