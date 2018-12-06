import torch
import torch.nn as nn
import torch.nn.functional as F


class TsinalisCNN(nn.Module):
	
	def __init__(self):
		super(TsinalisCNN, self).__init__()
		# CNN from Tsinalis paper:
		# 	(1,18000) input
		# 	conv1: 	20 1d filters of length 200
		# 	pool1: 	max pool size 20 stride 10
		# 	stack 	20 1d signals into 2d stack
		# 	conv2: 	400 filters size (20,30)
		# 	pool2: 	max pool size 10 stride 2
		# 	fc1: 	500 units
		# 	fc2: 	500 units
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=200)
		self.pool1 = nn.MaxPool1d(kernel_size=20, stride=10)
		self.conv2 = nn.Conv2d(1, 400, kernel_size=(20,30))
		self.pool2 = nn.MaxPool2d(kernel_size=(1,10), stride=(1,2))
		self.fc1 = nn.Linear(in_features=400*871, out_features=500)
		self.fc2 = nn.Linear(500, 5)

		# Smaller version of above
		# self.conv1 = nn.Conv1d(in_channels=1, out_channels=20, kernel_size=200)
		# self.pool1 = nn.MaxPool1d(kernel_size=20, stride=10)
		# self.conv2 = nn.Conv2d(1, 64, kernel_size=(20,30))
		# self.pool2 = nn.MaxPool2d(kernel_size=(1,20), stride=(1,4))
		# self.fc1 = nn.Linear(in_features=64*433, out_features=64)
		# self.fc2 = nn.Linear(64, 6)


	def forward(self, x):
		x = self.pool1(F.relu(self.conv1(x)))
		x = x.view(-1, 1, 20, 1779)
		x = self.pool2(F.relu(self.conv2(x)))
		x = x.view(-1, 400*871)
		#x = x.view(-1, 64*433)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x


class SimpleCNN(nn.Module):

	def __init__(self):
		super(SimpleCNN, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
		self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
		self.conv2 = nn.Conv1d(6, 16, 5)
		self.fc1 = nn.Linear(in_features=16 * 4497, out_features=64)
		self.fc2 = nn.Linear(64, 5)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 4497)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
		