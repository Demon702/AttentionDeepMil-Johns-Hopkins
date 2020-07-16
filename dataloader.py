"""Pytorch dataset object that loads MNIST dataset as bags."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
import random

class MnistBags(data_utils.Dataset):
    def __init__(self, train=True):
        self.train = train
        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def acceptor(self , data):
      # image = data[0]
      label = data[1]
      # image = torch.Tensor(np.array(image))
      # new_image = image[1: , : ,:]
      # black_new_image = torch.sum(image == 0 , 0)
      # black_score = sum(sum(black_new_image))
      # score = sum(sum(torch.sum(new_image > 200/255 , 0) > 1))
      # # print(score.item() , black_score.item())
      # if score > float(224*224*0.5) or black_score > float(224*224*0.25):
      #     return False
      if label == 0:
        if random.uniform(0,1) > 0.3:
          return False
        else :
          return True
    


    def my_collate(self , batch):
        batch = list(filter (lambda x:self.acceptor(x), batch))
        print(len(batch))
        if len(batch) == 0:
          return []
        return torch.utils.data.dataloader.default_collate(batch) 

    def _create_bags(self):
        if self.train:
            data_transform = transforms.Compose([
                    # transforms.RandomRotation(90),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    transforms.ToTensor(),
                ])

            train_dataset = datasets.ImageFolder(root='train',
                                           transform=data_transform)
            loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=1, shuffle=True,
                                             num_workers=8)
        else:
            data_transform = transforms.Compose([
                    # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                    # transforms.RandomRotation(90),
                    transforms.ToTensor(),
                ])
            # color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1)
            test_dataset = datasets.ImageFolder(root='test',
                                           transform=data_transform)
            loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=1, shuffle=False,
                                             num_workers=8)

        bags_list = []
        labels_list = []
        # counter = 0
        for (image, label) in loader:
          # print(counter)
          # counter+= 1
          images_in_bag = torch.Tensor([])
      #     print(image.shape)
      #     print(label)
      #     image  = image.squeeze()
      #     print(image.shape)
          [_ ,  _ , xlimit , ylimit] = image.shape
          xlimit = xlimit // 224
          ylimit = ylimit // 224
          for i in range(xlimit):
              for j in range(ylimit):
                  new_image = image[: , : , i*224 : i*224 + 224 , j*224 : j*224 + 224]
                  # new_image = color_jitter(new_image)
                  # new_image = torch.FloatTensor([new_image])

                  # print(squeezed_image) 
                  # if self.acceptor(squeezed_image):
                  images_in_bag = torch.cat((images_in_bag , new_image) , 0)
          # if self.train:
          #   if label.data[0] == 0:
          #     if np.random.uniform(0,1) < 17.0/ 61.0:
          #       labels_list.append(label.data[0])
          #       bags_list.append(images_in_bag)
          #   else:
          #     labels_list.append(label.data[0])
          #     bags_list.append(images_in_bag)
          # else:
          labels_list.append(label.data[0])
          bags_list.append(images_in_bag)



        # for i in range(self.num_bag):
        #     bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
        #     if bag_length < 1:
        #         bag_length = 1

        #     if self.train:
        #         indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
        #     else:
        #         indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

        #     labels_in_bag = all_labels[indices]
        #     labels_in_bag = labels_in_bag == self.target_number

        #     bags_list.append(all_imgs[indices])
        #     labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = self.train_labels_list[index]
        else:
            bag = self.test_bags_list[index]
            label = self.test_labels_list[index]

        return bag, label


if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistBags(train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistBags(train=False),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
      print(bag.shape)
      print(label)
        # len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        # mnist_bags_train += label[0].numpy()[0]
    # print('Number positive train bags: {}/{}\n'
    #       'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
    #     mnist_bags_train, len(train_loader),
    #     np.mean(len_bag_list_train), np.min(len_bag_list_train), np.max(len_bag_list_train)))

    # len_bag_list_test = []
    # mnist_bags_test = 0
    # for batch_idx, (bag, label) in enumerate(test_loader):
    #     len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
    #     mnist_bags_test += label[0].numpy()[0]
    # print('Number positive test bags: {}/{}\n'
    #       'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
    #     mnist_bags_test, len(test_loader),
    #     np.mean(len_bag_list_test), np.min(len_bag_list_test), np.max(len_bag_list_test)))
