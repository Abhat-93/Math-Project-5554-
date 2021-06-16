import torch
import torch.nn as nn
import matplotlib.pyplot as plt

X = (torch.rand(10000)*2-1).view(-1, 1)
Y = X*X
i =12
model = nn.Sequential(
    nn.Linear(1, i),
    nn.ReLU(),
    nn.Linear(i, i),
    nn.Tanh(),
    nn.Linear(i, i),
    nn.Tanh(),
    nn.Linear(i, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
loss_func = nn.MSELoss()


for _ in range(50000):
    optimizer.zero_grad()
    pred = model(X)
    loss = loss_func(pred, Y)
    loss.backward()
    optimizer.step()

x = torch.linspace(-2, 2, steps=200).view(-1, 1)
y = model(x)
f = x*x

fig, a = plt.subplots(2, 2)

a[0,0].plot(x.detach().view(-1).numpy(), y.detach().view(-1).numpy(), 'r.', linestyle='None')
a[0,0].plot(x.detach().view(-1).numpy(), f.detach().view(-1).numpy(), 'b')
a[0,0].show()

a[0,1].plot(x.detach().view(-1).numpy(), y.detach().view(-1).numpy(), 'r.', linestyle='None')
a[0,1].plot(x.detach().view(-1).numpy(), f.detach().view(-1).numpy(), 'b')
a[0,1].show()

a[1,0].plot(x.detach().view(-1).numpy(), y.detach().view(-1).numpy(), 'r.', linestyle='None')
a[1,0].plot(x.detach().view(-1).numpy(), f.detach().view(-1).numpy(), 'b')
a[1,0].show()

a[1,1].plot(x.detach().view(-1).numpy(), y.detach().view(-1).numpy(), 'r.', linestyle='None')
a[1,1].plot(x.detach().view(-1).numpy(), f.detach().view(-1).numpy(), 'b')
a[1,1].show()

plt.show()
