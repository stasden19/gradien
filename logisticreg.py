import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class ModelNN(torch.nn.Module):
    def __init__(self):
        super(ModelNN, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.ReLu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = torch.sigmoid(x)
        return x


x = torch.FloatTensor(np.array([[-100, -2, -1, 0, 1, 2, 3, 3.5, 4, 5, 6, 7, 8, 9]], dtype=np.float32).T)
y = torch.FloatTensor(np.array([[0, 0, 0, 0, 0, 0, 0, 0.5, 1, 1, 1, 1, 1, 1]], dtype=np.float32))
# x = torch.FloatTensor(np.array([[1, 0], [2, 0], [3, 0], [3.5, 0.5], [4, 1], [5, 1], [6, 1], [7, 0], [8, 0], [9, 0], [10, 0]], dtype=np.float32))

model = ModelNN()
crit = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=1)
all_losses = []

for epoch in range(100):
    optimizer.zero_grad()
    print(optimizer.state)
    output = model(x)
    loss = crit(output, y.view(-1, 1))  # Используйте .view(-1, 1) для изменения формы y
    all_losses.append(loss.item())
    loss.backward()
    optimizer.step()

plt.plot(all_losses)
plt.show()
# plot_decision_boundary(model, x, y)
# print(model(torch.FloatTensor([[3]])))  # Передача тензора
# print(model(torch.FloatTensor([[6]])))  # Передача тензора
# print(*model.parameters())
