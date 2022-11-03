from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class simple_fc(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(simple_fc, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size) 
		self.bn1 = nn.BatchNorm1d(hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, hidden_size*2) 
		self.bn2 = nn.BatchNorm1d(hidden_size*2)
		self.relu2 = nn.ReLU() 
		self.fc3 = nn.Linear(hidden_size*2, hidden_size*4)  
		self.bn3 = nn.BatchNorm1d(hidden_size*4)
		self.relu3 = nn.ReLU()
		self.fc4 = nn.Linear(hidden_size*4, num_classes)  


	def forward(self, x):
		out= x.view(x.size(0),-1)
		out = self.fc1(out)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.bn2(out)
		out = self.relu2(out)
		out = self.fc3(out)
		out = self.bn3(out)
		out = self.relu3(out)
		out = self.fc4(out)
		return out


class simple_fc_no_bn(nn.Module):
	def __init__(self, input_size, hidden_size, num_classes):
		super(simple_fc_no_bn, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size) 
		# self.bn1 = nn.BatchNorm1d(hidden_size)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_size, hidden_size*2) 
		# self.bn2 = nn.BatchNorm1d(hidden_size*2)
		self.relu2 = nn.ReLU() 
		self.fc3 = nn.Linear(hidden_size*2, hidden_size*4)  
		# self.bn3 = nn.BatchNorm1d(hidden_size*4)
		self.relu3 = nn.ReLU()
		self.fc4 = nn.Linear(hidden_size*4, num_classes)  


	def forward(self, x):
		out= x.view(x.size(0),-1)
		out = self.fc1(out)
		# out = self.bn1(out)
		out = self.relu(out)
		out = self.fc2(out)
		# out = self.bn2(out)
		out = self.relu2(out)
		out = self.fc3(out)
		# out = self.bn3(out)
		out = self.relu3(out)
		out = self.fc4(out)
		return out


##############################################################################################################################
class simple_conv(nn.Module):
	def __init__(self, num_classes):
		super(simple_conv, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, 3, 1)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 64, 3, 1)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 128, 3, 1)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(128, 256, 3, 1)
		self.bn4 = nn.BatchNorm2d(256)
		# self.fc = nn.Linear(6400, num_classes)
		self.fc = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.conv1(x)	# 26
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv2(x)	# 24
		x = self.bn2(x)
		x = F.relu(x)
		x = self.conv3(x)   # 22
		x = self.bn3(x)
		x = F.relu(x)
		x = self.conv4(x)   # 20
		x = self.bn4(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)
		x = F.avg_pool2d(x, 20)
		x = torch.flatten(x, 1)
		out= self.fc(x)

		return out


class conv_fc(nn.Module):
	def __init__(self, num_classes):
		super(conv_fc, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 32, 3, 1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, 3, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 64, 3, 1)
		self.bn4 = nn.BatchNorm2d(64)
		self.fc = nn.Linear(4096, num_classes)
		# self.fc = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.conv1(x)	# 26
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv2(x)	# 24
		x = self.bn2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)	# 12
		x = self.conv3(x)   # 10
		x = self.bn3(x)
		x = F.relu(x)
		x = self.conv4(x)   # 8
		x = self.bn4(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)
		# x = F.avg_pool2d(x, 20)
		x = torch.flatten(x, 1)
		out= self.fc(x)

		return out

class conv_fc8192(nn.Module):
	def __init__(self, num_classes):
		super(conv_fc8192, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, 3, 1)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 64, 3, 1)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 128, 3, 1)
		self.bn3 = nn.BatchNorm2d(128)
		self.conv4 = nn.Conv2d(128, 128, 3, 1)
		self.bn4 = nn.BatchNorm2d(128)
		self.fc = nn.Linear(8192, num_classes)
		# self.fc = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.conv1(x)	# 26
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv2(x)	# 24
		x = self.bn2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)	# 12
		x = self.conv3(x)   # 10
		x = self.bn3(x)
		x = F.relu(x)
		x = self.conv4(x)   # 8
		x = self.bn4(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)
		# x = F.avg_pool2d(x, 20)
		x = torch.flatten(x, 1)
		out= self.fc(x)

		return out


class Conv_avg_fc512(nn.Module):
	def __init__(self, num_classes):
		super(Conv_avg_fc512, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, 3, 1)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 128, 3, 1)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 256, 3, 1)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 512, 3, 1)
		self.bn4 = nn.BatchNorm2d(512)
		self.fc = nn.Linear(512, num_classes)
		# self.fc = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.conv1(x)	# 26
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv2(x)	# 24
		x = self.bn2(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)	# 12
		x = self.conv3(x)   # 22
		x = self.bn3(x)
		x = F.relu(x)
		x = self.conv4(x)   # 20
		x = self.bn4(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)
		x = F.avg_pool2d(x, 20)
		x = torch.flatten(x, 1)
		out= self.fc(x)

		return out


class Conv_avg_fc64(nn.Module):
	def __init__(self, num_classes):
		super(Conv_avg_fc64, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 32, 3, 1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, 3, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 64, 3, 1)
		self.bn4 = nn.BatchNorm2d(64)
		self.fc = nn.Linear(64, num_classes)
		# self.fc = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.conv1(x)	# 26
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv2(x)	# 24
		x = self.bn2(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)	# 12
		x = self.conv3(x)   # 22
		x = self.bn3(x)
		x = F.relu(x)
		x = self.conv4(x)   # 20
		x = self.bn4(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)
		x = F.avg_pool2d(x, 20)
		x = torch.flatten(x, 1)
		out= self.fc(x)

		return out


class Conv_max_fc800(nn.Module):
	def __init__(self, num_classes):
		super(Conv_max_fc800, self).__init__()
		self.conv1 = nn.Conv2d(1, 8, 3, 1)
		self.bn1 = nn.BatchNorm2d(8)
		self.conv2 = nn.Conv2d(8, 8, 3, 1)
		self.bn2 = nn.BatchNorm2d(8)
		self.conv3 = nn.Conv2d(8, 8, 3, 1)
		self.bn3 = nn.BatchNorm2d(8)
		self.conv4 = nn.Conv2d(8, 8, 3, 1)
		self.bn4 = nn.BatchNorm2d(8)
		self.fc = nn.Linear(800, num_classes)
		# self.fc = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.conv1(x)	# 26
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv2(x)	# 24
		x = self.bn2(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)	# 12
		x = self.conv3(x)   # 22
		x = self.bn3(x)
		x = F.relu(x)
		x = self.conv4(x)   # 20
		x = self.bn4(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		# x = F.avg_pool2d(x, 20)
		x = torch.flatten(x, 1)
		out= self.fc(x)

		return out


class Conv_max_fc6400(nn.Module):
	def __init__(self, num_classes):
		super(Conv_max_fc6400, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 64, 3, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 64, 3, 1)
		self.bn4 = nn.BatchNorm2d(64)
		self.fc = nn.Linear(6400, num_classes)
		# self.fc = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.conv1(x)	# 26
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv2(x)	# 24
		x = self.bn2(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)	# 12
		x = self.conv3(x)   # 22
		x = self.bn3(x)
		x = F.relu(x)
		x = self.conv4(x)   # 20
		x = self.bn4(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		# x = F.avg_pool2d(x, 20)
		x = torch.flatten(x, 1)
		out= self.fc(x)

		return out


class Conv_max_fc4096(nn.Module):
	def __init__(self, num_classes):
		super(Conv_max_fc4096, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 32, 3, 1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 64, 3, 1)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64, 64, 3, 1)
		self.bn4 = nn.BatchNorm2d(64)
		self.fc = nn.Linear(4096, num_classes)
		# self.fc = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.conv1(x)	# 26
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv2(x)	# 24
		x = self.bn2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)	# 12
		x = self.conv3(x)   # 10
		x = self.bn3(x)
		x = F.relu(x)
		x = self.conv4(x)   # 8
		x = self.bn4(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)
		# x = F.avg_pool2d(x, 20)
		x = torch.flatten(x, 1)
		out= self.fc(x)

		return out


class Conv_max_fc16384(nn.Module):
	def __init__(self, num_classes):
		super(Conv_max_fc16384, self).__init__()
		self.conv1 = nn.Conv2d(1, 128, 3, 1)
		self.bn1 = nn.BatchNorm2d(128)
		self.conv2 = nn.Conv2d(128, 128, 3, 1)
		self.bn2 = nn.BatchNorm2d(128)
		self.conv3 = nn.Conv2d(128, 256, 3, 1)
		self.bn3 = nn.BatchNorm2d(256)
		self.conv4 = nn.Conv2d(256, 256, 3, 1)
		self.bn4 = nn.BatchNorm2d(256)
		self.fc = nn.Linear(16384, num_classes)
		# self.fc = nn.Linear(256, num_classes)

	def forward(self, x):
		x = self.conv1(x)	# 26
		x = self.bn1(x)
		x = F.relu(x)
		x = self.conv2(x)	# 24
		x = self.bn2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)	# 12
		x = self.conv3(x)   # 10
		x = self.bn3(x)
		x = F.relu(x)
		x = self.conv4(x)   # 8
		x = self.bn4(x)
		x = F.relu(x)
		# x = F.max_pool2d(x, 2)
		# x = F.avg_pool2d(x, 20)
		x = torch.flatten(x, 1)
		out= self.fc(x)

		return out


class simple_conv_no_bn(nn.Module):
	def __init__(self, num_classes):
		super(simple_conv_no_bn, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.conv3 = nn.Conv2d(64, 64, 3, 1)
		self.conv4 = nn.Conv2d(64, 64, 3, 1)
		self.fc = nn.Linear(6400, num_classes)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = self.conv3(x)
		x = F.relu(x)
		x = self.conv4(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = torch.flatten(x, 1)
		out= self.fc(x)

		return out
##############################################################################################################################

def fc_bn(hidden_size=512, num_classes=10):
	return simple_fc(784, hidden_size=hidden_size, num_classes=num_classes)

def fc_no_bn(hidden_size=512, num_classes=10):
	return simple_fc_no_bn(784, hidden_size=hidden_size, num_classes=num_classes)


def conv_bn_avg(num_classes=10):
	return simple_conv(num_classes=num_classes)


def conv_bn_fc(num_classes=10):
	return conv_fc(num_classes=num_classes)
	

def conv_no_bn(num_classes=10):
	return simple_conv_no_bn(num_classes=num_classes)



# 
def conv_bn_fc8192(num_classes=10):
	return conv_fc8192(num_classes=num_classes)

def conv_avg_fc512(num_classes=10):
	return Conv_avg_fc512(num_classes=num_classes)

def conv_avg_fc64(num_classes=10):
	return Conv_avg_fc64(num_classes=num_classes)



def conv_max_fc800(num_classes=10):
	return Conv_max_fc800(num_classes=num_classes)


def conv_max_fc6400(num_classes=10):
	return Conv_max_fc6400(num_classes=num_classes)


def conv_max_fc4096(num_classes=10):
	return Conv_max_fc4096(num_classes=num_classes)


def conv_max_fc16384(num_classes=10):
	return Conv_max_fc16384(num_classes=num_classes)	