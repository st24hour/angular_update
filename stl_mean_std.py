import numpy as np
# load data
from torchvision import datasets
import torch
from torchvision import transforms

# load the training data
# train_data = datasets.STL10('/home/user/jusung/data/stl10-data', split='train', download=True, transform=transforms.ToTensor())
# imgs = [item[0] for item in train_data] # item[0] and item[1] are image and its label
# imgs = torch.stack(imgs, dim=0).numpy()
# print(imgs.shape)	# 5000, 3, 96, 96

# # calculate mean over each channel (r,g,b)
# mean_r = imgs[:,0,:,:].mean()
# mean_g = imgs[:,1,:,:].mean()
# mean_b = imgs[:,2,:,:].mean()
# print(mean_r,mean_g,mean_b) # 0.44671047 0.43981034 0.40664658

# # calculate std over each channel (r,g,b)
# std_r = imgs[:,0,:,:].std()
# std_g = imgs[:,1,:,:].std()
# std_b = imgs[:,2,:,:].std()
# print(std_r,std_g,std_b) # 0.26034108 0.25657734 0.27126735


train_data = datasets.ImageFolder('/home/user/jusung/data/tiny-imagenet-200/train/', transform=transforms.ToTensor())
imgs = [item[0] for item in train_data] # item[0] and item[1] are image and its label
imgs = torch.stack(imgs, dim=0).numpy()
print(imgs.shape)	# 100000, 3, 64, 64

# calculate mean over each channel (r,g,b)
mean_r = imgs[:,0,:,:].mean()
mean_g = imgs[:,1,:,:].mean()
mean_b = imgs[:,2,:,:].mean()
print(mean_r,mean_g,mean_b) # 0.48024544 0.44806936 0.39754808

# calculate std over each channel (r,g,b)
std_r = imgs[:,0,:,:].std()
std_g = imgs[:,1,:,:].std()
std_b = imgs[:,2,:,:].std()
print(std_r,std_g,std_b) # 0.27698606 0.26906446 0.28208157


# train_data = datasets.CIFAR10('/home/user/jusung/data/cifar10-data', train=True, download=True, transform=transforms.ToTensor())

# imgs = [item[0] for item in train_data] # item[0] and item[1] are image and its label
# imgs = torch.stack(imgs, dim=0).numpy()

# # calculate mean over each channel (r,g,b)
# mean_r = imgs[:,0,:,:].mean()
# mean_g = imgs[:,1,:,:].mean()
# mean_b = imgs[:,2,:,:].mean()
# print(mean_r,mean_g,mean_b)

# # calculate std over each channel (r,g,b)
# std_r = imgs[:,0,:,:].std()
# std_g = imgs[:,1,:,:].std()
# std_b = imgs[:,2,:,:].std()
# print(std_r,std_g,std_b)


