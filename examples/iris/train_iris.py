import torch
import torchvision
import torch.nn.functional as F
import pandas as pd
import numpy as np

iris = pd.read_csv("iris.csv")

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Linear = torch.nn.Linear(4, 3, bias=False)

    def forward(self, x):
        x = self.Linear(x)
        x = F.log_softmax(x, dim=1)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.iris_frame = pd.read_csv(csv_file)
        self.name_to_id = {"setosa":0, "versicolor":1, "virginica":2}

    def __len__(self):
        return len(self.iris_frame)

    def __getitem__(self, idx):

        x = self.iris_frame.iloc[idx]
        x = np.array(x)
        return torch.Tensor([x[0], x[1], x[2], x[3]]), self.name_to_id[x[-1]]

dataset = Dataset("iris.csv")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

for epoch in range(20):
    ncorrect = 0
    for d, sample in enumerate(dataloader):
        optimizer.zero_grad()
        x = sample[0]
        target = sample[1]
        output = model(x)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        ncorrect += len([i for i in (predicted==target) if i==True])

    scheduler.step()
    print(ncorrect, len(dataset))

print(model.Linear.weight, model.Linear.bias)

torch.save(model.state_dict(), "iris.pt")
