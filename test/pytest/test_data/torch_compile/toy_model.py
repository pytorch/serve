import torch


class HalfPlusTwoModel(torch.nn.Module):
    def forward(self, *input_args):
        w = torch.tensor(0.5)
        b = torch.tensor(2.0)
        return torch.add(torch.multiply(w, input_args[0]), b)


if __name__ == "__main__":
    model = HalfPlusTwoModel()
    torch.save(model.state_dict(), "model.pt")
