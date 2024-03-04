import torch.optim as optim

# momentum=0.9 smoothes out updates and can help training
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

optimizer = optim.Adam(model.parameters(), lr=0.01)
