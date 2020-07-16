#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[15]:


from __future__ import print_function, division
import torch.nn as nn
import os
import cv2
import torch
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
from skimage import io, transform, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models , utils , datasets
import time
import os
import copy
import random
import sklearn.metrics
from PIL import ImageFile , Image
import matplotlib.pyplot as plt
from resnet_classifier import Resnet_Classifier
import utils
from model import Attention
# In[16]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[17]:



def imshow(inp, title=None):
	"""Imshow for Tensor."""
	inp = inp.numpy().transpose((1, 2, 0))
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(10)


# In[18]:


data_transform = transforms.Compose([
		transforms.RandomResizedCrop(450),
		transforms.RandomHorizontalFlip(),
		transforms.RandomRotation((-90 , 90)),
		transforms.ToTensor(),  
	])

data_transform_test = transforms.Compose([
		transforms.RandomResizedCrop(450),
		transforms.ToTensor(),  
	])


# In[19]:


def acceptor(data):
	image = data[0]
# 	print(image.shape)
	image = torch.Tensor(np.array(image))
# 	new_image = image[1: , : ,:]
	black_new_image = torch.sum(image, 0) == 0
	black_score = sum(sum(black_new_image))
# 	print("black score = " , float(black_score)/(450*450))
# 	score = sum(sum(torch.sum(new_image > 200.0/255.0 , 0) > 1)).double()
	# print(score.item() , black_score.item())
	if black_score.item() > 450*450*0.20:
		return False
	return True


# In[20]:


def my_collate(batch): # batch size 4 [{tensor image, tensor label},{},{},{}] could return something like G = [None, {},{},{}]
	batch = list(filter (lambda x:acceptor(x), batch)) # this gets rid of nones in batch. For example above it would result to G = [{},{},{}]
	transform_ = transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.1),
                                     transforms.ToTensor()]
                                   )
# 	print(batch)
# 	for x in batch:
# 		print(Image.fromarray(x[0].numpy().transpose((1, 2, 0))))
# 		print(x[1])
# 	batch = [transform_(x) for x in batch]
# 	print("Length Of the batch = " , len(batch))
	if len(batch) == 0:
		return [None , None]
	else:
		batch = [(transform_(Image.fromarray(img_as_ubyte(x[0].numpy().transpose((1, 2, 0))))),x[1]) for x in batch]
	# print(batch)
       
	return torch.utils.data.dataloader.default_collate(batch)


# In[21]:


tumor_train_dataset = datasets.ImageFolder(root='train',
										   transform=data_transform)


# In[22]:
tumor_val_dataset = datasets.ImageFolder(root='test',
										   transform=data_transform_test)

train_dataset_loader = torch.utils.data.DataLoader(tumor_train_dataset,
											 batch_size=10, shuffle=True,
											 num_workers=1 , collate_fn = my_collate)

val_dataset_loader = torch.utils.data.DataLoader(tumor_val_dataset,
											 batch_size=10, shuffle=True,
											 num_workers=1 , collate_fn = my_collate)


# In[ ]:


# inputs, classes = next(iter(train_dataset_loader))
# # Make a grid from batch
# if inputs is not None:
#     out = utils.make_grid(inputs)
# # # print(out.size())
# # # print(inputs.size())
# # # print(classes)
#     imshow(out)
# # if inputs is not None:
# # 	for input in inputs:
# #     if input != None:
# 		imshow(input)
# # 		print(np.amax(input.numpy()))


# In[10]:


# [1] *4


# In[11]:


# for image,label in train_dataset_loader:
#     out = utils.make_grid(image)
#     imshow(out)


# In[23]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()
	# best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	loss_array = []
	accuracy_array = []
	best_acc = 0
	for epoch in range(num_epochs):
		# total = 0
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		model.train()  # Set model to training mode

		running_loss = 0.0
		running_corrects = 0
			# Iterate over data.
		total = 0
		for inputs, labels in train_dataset_loader:
			if inputs is None:
				continue
			inputs = inputs.to(device)
			labels = labels.to(device)
			# print(labels)
			total += len(inputs)
			# zero the parameter gradients
			optimizer.zero_grad()

			# forward
			# track history if only in train
			with torch.set_grad_enabled(True):
				outputs  = model(inputs)
				# print()
				_, preds = torch.max(outputs, 1)
				# labels = labels.float().unsqueeze(0)
				# labels = torch.transpose(labels, 0, 1)	
				# print(labels)
				# print(outputs)
				# print(labels)
				loss = criterion(outputs , labels)
				# loss =  -1. * (1.0/58) * (labels * torch.log(outputs) + (1. - labels) * (1.0/18) * torch.log(1. - outputs))
				# loss = torch.sum(loss , 0).squeeze(0)
				# print(loss)
				# print(labels , preds)
				# backward + optimize only if in training phas
				loss.backward()
				optimizer.step()


			# statistics
			running_loss += loss.item() * inputs.size(0)
			running_corrects += torch.sum(preds == labels.data)


		
				
# 		time.sleep(10)
		# torch.save(model.state_dict() , "model34")
		scheduler.step()
		# print('total = {}'.format(total))
		epoch_loss = running_loss / total
		epoch_acc = running_corrects.double() / total
		# loss_array.append(epoch_loss)
		# accuracy_array.append(epoch_acc.item())
# 		np.save("loss_array" , loss_array)
# 		np.save("accuracy_array" , accuracy_array)
		print('Loss: {:.4f} Acc: {:.4f}'.format(
			epoch_loss, epoch_acc))
		plotter.plot('loss', 'train', 'Loss for third cross', epoch, epoch_loss)
		plotter.plot('accuracy', 'train', 'Accuracy for third cross', epoch, epoch_acc)


		model.eval()
		with torch.no_grad():
			running_loss = 0.0
			running_corrects = 0
				# Iterate over data.
			total = 0
			for inputs, labels in val_dataset_loader:
				if inputs is None:
					continue
				inputs = inputs.to(device)
				labels = labels.to(device)
				# print(labels)
				total += len(inputs)
				# zero the parameter gradients
				# optimizer.zero_grad()

				# forward
				# track history if only in train
				# with torch.set_grad_enabled(True):
				outputs  = model(inputs)
				# print()
				_, preds = torch.max(outputs, 1)
				# labels = labels.float().unsqueeze(0)
				# labels = torch.transpose(labels, 0, 1)	
				# print(labels)
				# print(outputs)
				# print(labels)
				loss = criterion(outputs , labels)
				# loss =  -1. * (1.0/58) * (labels * torch.log(outputs) + (1. - labels) * (1.0/18) * torch.log(1. - outputs))
				# loss = torch.sum(loss , 0).squeeze(0)
				# print(loss)
				# print(labels , preds)
				# backward + optimize only if in training phas
				# loss.backward()
				# optimizer.step()


				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
# 		print()
		val_loss = running_loss / total
		val_acc = running_corrects.double() / total
		if val_acc > best_acc:
			best_acc = val_acc
			torch.save(model.state_dict() , "model34")
		torch.save(model.state_dict() , "model34_" + str(epoch))
		plotter.plot('loss', 'val', 'Val Loss', epoch, val_loss)
		plotter.plot('accuracy', 'val', 'Class Accuracy', epoch, val_acc)
	vis.save(['Tutorial Plots'])
	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	return model


# In[24]:

if __name__ == "__main__":
	model_ft = Resnet_Classifier()
	# model_ft  = Attention()
	model_ft = model_ft.to(device)
	criterion = nn.CrossEntropyLoss(weight = torch.Tensor([1.0/163.0 , 1.0/123.0]).to(device))

	optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.00001)
	scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=200)
	global plotter
	plotter = utils.VisdomLinePlotter(env_name='Tutorial Plots_Resnet')

# In[ ]:


	model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=200)

