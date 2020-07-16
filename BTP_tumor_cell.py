#!/usr/bin/env python
# coding: utf-8

# In[28]:


from __future__ import print_function, division
import torch.nn as nn
import os
import cv2
import torch
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models , utils , datasets
import time
import os
import copy
import random
import sklearn.metrics
from PIL import ImageFile
import matplotlib.pyplot as plt
from resnet_classifier import Resnet_Classifier
ImageFile.LOAD_TRUNCATED_IMAGES = True
# random.seed(2)
# In[12]:


data_transform = transforms.Compose([
		transforms.RandomResizedCrop(224),
		# transforms.RandomHorizontalFlip(),
		transforms.RandomRotation((-90 , 90)),
		transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
		transforms.ToTensor(),
	])


total = 0 

# In[120]:


# print(tumor_dataset.class_to_idx)

# In[133]:

def acceptor(data):
#     print(image.shape)
	image = data[0]
	image = torch.Tensor(np.array(image))
	new_image = image[1: , : ,:]
	black_new_image = torch.sum(image == 0 , 0)
	black_score = sum(sum(black_new_image))
#     print("black score = " , black_score)
	score = sum(sum(torch.sum(new_image > 200.0/255.0 , 0) > 1)).double()
	# print(score.item() , black_score.item())
	if data[1] == 0:
		if score.item() > 224*224*0.5 or black_score.item() > 224*224*0.25:
			return False
	return True
	


# In[136]:


def my_collate(batch): # batch size 4 [{tensor image, tensor label},{},{},{}] could return something like G = [None, {},{},{}]
	batch = list(filter (lambda x:acceptor(x), batch)) # this gets rid of nones in batch. For example above it would result to G = [{},{},{}]
	if len(batch) == 0:
		return [None , None]
	return torch.utils.data.dataloader.default_collate(batch) 


# In[139]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[117]:


def imshow(inp, title=None):
	"""Imshow for Tensor."""
	inp = inp.numpy().transpose((1, 2, 0))
	plt.imshow(inp)
	if title is not None:
		plt.title(title)
	plt.pause(0.001)


# In[140]:


# inputs, classes = next(iter(dataset_loader))

# # Make a grid from batch
# out = utils.make_grid(inputs)
# print(out.size())
# print(inputs.size())
# print(classes)
# imshow(out)


# In[107]:


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
	since = time.time()
	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
	loss_array = []
	accuracy_array = []
	for epoch in range(num_epochs):
		# total = 0
		torch.cuda.manual_seed_all(np.random.randint(100))
		data_transform = transforms.Compose([
				transforms.RandomResizedCrop(224),
				# transforms.RandomHorizontalFlip(),
				transforms.RandomRotation((-90 , 90)),
				transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
				transforms.ToTensor(),
					])
		tumor_dataset = datasets.ImageFolder(root='train',
										   transform=data_transform)
		dataset_loader = torch.utils.data.DataLoader(tumor_dataset,
											 batch_size=8, shuffle=True,
											 num_workers=8 , collate_fn = my_collate)
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)

		model.train()  # Set model to training mode

		running_loss = 0.0
		running_corrects = 0
			# Iterate over data.
		total = 0
		for inputs, labels in dataset_loader:
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
			
		time.sleep(10)
		torch.save(model.state_dict() , "model34")
		scheduler.step()
		print('total = {}'.format(total))
		epoch_loss = running_loss / total
		epoch_acc = running_corrects.double() / total
		loss_array.append(epoch_loss)
		accuracy_array.append(epoch_acc.item())
		np.save("loss_array" , loss_array)
		np.save("accuracy_array" , accuracy_array)
		print('Loss: {:.4f} Acc: {:.4f}'.format(
			epoch_loss, epoch_acc))

		print()

	time_elapsed = time.time() - since
	print('Training complete in {:.0f}m {:.0f}s'.format(
		time_elapsed // 60, time_elapsed % 60))
	return model


# In[109]:


model_ft = Resnet_Classifier()
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss(weight = torch.Tensor([1.0/61 , 1.0/17]).to(device))

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

# model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=200)
model_ft.load_state_dict(torch.load("model34_2"))

# In[83]:


tumor_test_dataset = datasets.ImageFolder(root='test',
										   transform=data_transform)


# In[84]:


test_dataset_loader = torch.utils.data.DataLoader(tumor_test_dataset,
											 batch_size=4, shuffle=True,
											 num_workers=1 , collate_fn = my_collate)


# In[177]:


model_ft.eval()
running_loss = 0
running_corrects = 0
y_true = []
y_pred = []
y_prob = []
total = 0
for _ in range(20):
	torch.cuda.manual_seed_all(np.random.randint(100))
	data_transform = transforms.Compose([
				transforms.RandomResizedCrop(224),
				# transforms.RandomHorizontalFlip(),
				# transforms.RandomRotation((-90 , 90)),
				# transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
				transforms.ToTensor(),
					])
	tumor_test_dataset = datasets.ImageFolder(root='test',
										   transform=data_transform)



	test_dataset_loader = torch.utils.data.DataLoader(tumor_test_dataset,
											 batch_size=4, shuffle=True,
											 num_workers=1 , collate_fn = my_collate)
	for inputs, labels in test_dataset_loader:
		inputs = inputs.to(device)
		labels = labels.to(device)
		total += inputs.size(0)
		# zero the parameter gradients
	#         optimizer.zero_grad()

		# forward
		# track history if only in train
	#         with torch.set_grad_enabled(True):
		with torch.no_grad():
			outputs = model_ft(inputs)
			_, preds = torch.max(outputs, 1)
			print(labels , preds)
			loss = criterion(outputs, labels)
		# exp_output = np.exp(outputs.detach())
		# sum_output = torch.sum(exp_output , 1).view(inputs.size(0),1).numpy()
		# prob = exp_output.numpy() / sum_output
		# print(prob)
		# backward + optimize only if in training phas
	#         loss.backward()
	#         optimizer.step()

		# statistics
		y_true = np.append(y_true , labels.cpu().numpy())
		y_pred = np.append(y_pred , preds.cpu().numpy())
		y_prob = np.append(y_prob , outputs.detach().cpu().numpy()[: , 1])
		running_loss += loss.item() * inputs.size(0)
		running_corrects += torch.sum(preds == labels.data)
np.save("y_true" , y_true)
np.save("y_pred" , y_pred)
print('loss = {}'.format(running_loss / total))
print('accuracy = {}'.format(running_corrects.double() / total))
# fpr, tpr, threshold = sklearn.metrics.roc_curve(y_true, y_pred)
# roc_auc = sklearn.metrics.auc(fpr, tpr)
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])	
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.save("Roc_Curve.png")
print(sklearn.metrics.confusion_matrix(y_true, y_pred))
print(sklearn.metrics.roc_auc_score(y_true , y_prob))

print('Patient Level AUC = {:.4f}'.format(metrics.roc_auc_score(np.array(labels)[: len(test_loader)] , np.array(patient_probs))))


# fpr, tpr, threshold = sklearn.metrics.roc_curve(1 - y_true, 1 - y_pred)
# roc_auc = sklearn.metrics.auc(fpr, tpr)
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.save("Roc_Curve_2.png")
print(sklearn.metrics.confusion_matrix(1 - y_true, 1 - y_pred))
print(sklearn.metrics.roc_auc_score(1 - y_true , 1 - y_prob))


	  
