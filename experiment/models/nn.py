import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


def weight_init(m: nn.Module):
  if isinstance(m, nn.Conv2d):
    torch.nn.init.xavier_uniform_(m.weight)
    if m.bias is not None:
      m.bias.data.fill_(0.01)
  elif isinstance(m, nn.Linear):
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.01)

class MLPSolver:
  def __init__(self):
    self.model = nn.Sequential(
      nn.Linear(64, 64),
      nn.ReLU(inplace=True),
      nn.Linear(64, 32),
      nn.ReLU(inplace=True),
      nn.Linear(32, 10),
      nn.LogSoftmax(dim=1)
    )
    self.model.apply(weight_init)

    torch.set_num_threads(24)

  def train(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    opt = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(500):
      opt.zero_grad()
      y_pred = self.model(X)
      loss = loss_func(y_pred, y)
      loss.backward()
      opt.step()

  def score(self, X: np.ndarray, y: np.ndarray):
    X = X.reshape(X.shape[0], -1)
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    y_pred = self.model(X)
    return (y_pred.argmax(dim=1) == y).float().mean().item()


class CNNSolver:
  def __init__(self):
    self.model = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.ReLU(inplace=True),
      nn.Flatten(),
      nn.Linear(32 * 4, 10),
      nn.LogSoftmax(dim=1)
    )
    self.model.apply(weight_init)

    torch.set_num_threads(24)

  def train(self, X: np.ndarray, y: np.ndarray):
    X = torch.from_numpy(X).float().unsqueeze(1)
    y = torch.from_numpy(y).long()

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    opt = optim.SGD(self.model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-4)
    loss_func = nn.CrossEntropyLoss()

    for epoch in range(50):
      for x, y in dataloader:
        opt.zero_grad()
        y_pred = self.model(x)
        loss = loss_func(y_pred, y)
        loss.backward()
        opt.step()

  def score(self, X: np.ndarray, y: np.ndarray):
    X = torch.from_numpy(X).float().unsqueeze(1)
    y = torch.from_numpy(y).long()

    y_pred = self.model(X)
    return (y_pred.argmax(dim=1) == y).float().mean().item()
