
### IMPORTANT ###

#File Structure

#
# - Space_Net_Data (any folder)
#        |-------- building_mask
#        |-------- SN1_buildings_train_AOI_1_Rio_3band
#        |                       |--------- 3band
#        |-------- space_net_data_loader.py (THIS FILE)
#

import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
import os
import cv2


class SpaceNetDataset(Dataset):
    def __init__(self, usage, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.usage = usage
        self.img_list = []

        img_path = Path(os.path.dirname(__file__) + img_dir)
        for name in os.listdir(img_path):
            if os.path.isfile(os.path.join(img_path, name)):
                self.img_list.append(name)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])
        img_path = os.getcwd() + img_path
        #for some reason half the images are 439,407 and half are 438,406
        image = np.array(Image.open(img_path))[:406,:438]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (572, 572))
        image = np.asarray([image])

        if self.usage == 'train':
            lbl_path = os.path.join(self.label_dir, self.img_list[idx])
            lbl_path = os.getcwd() + lbl_path
            #for some reason half the images are 439,407 and half are 438,406
            label_building = cv2.imread(lbl_path)
            label_building = cv2.cvtColor(label_building, cv2.COLOR_BGR2GRAY)

            label_building = label_building[:406,:438]
            label_building = cv2.resize(label_building, (388, 388))
            label_no_building = 1 - label_building
            label = np.asarray([label_building, label_no_building])

            sample = {'image': image, 'label': label_building, 'image_name': self.img_list[idx]}
        elif self.usage == 'test':
            sample = {'image': image, 'image_name': self.img_list[idx]}
        return sample



if __name__ == '__main__':

    img_dir = os.path.join('SN1_buildings_train_AOI_1_Rio_3band','3band')
    label_dir = 'building_mask'

    # As of right now usage, just 'train' and 'test'
    # train returns image, label, and name
    # test has no label, so just image and name
    dataset = SpaceNetDataset('train', img_dir, label_dir)

    batch_size = 64
    validation_split = .2
    shuffle_dataset = True
    random_seed = 10

    #https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    ### TRAINING HERE ###
    for epoch in range(1):
        for samples in train_loader:
            print(samples)


        #validation_loader
