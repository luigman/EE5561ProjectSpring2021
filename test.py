import matplotlib.pyplot as plt
from space_net_data_loader import *
from unet import *
from space_net_data_loader import SpaceNetDataset
import torch
from torchvision import transforms
import torch.nn as nn
import os
import argparse

def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='CSCI 5563 HW3')
    parser.add_argument('--d', type=str,
                        default='0')
    parser.add_argument('--batch_size', type=int,
                        default='1')
    parser.add_argument('--load', type=str,
                        default='trained_models/trained_model-00015.ckpt')

    return parser.parse_args()

if __name__ == '__main__':
    args = ParseCmdLineArguments()
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.d

    img_dir = '/SN1_buildings_train_AOI_1_Rio_3band/3band/'
    label_dir = '/building_mask/'

    # As of right now usage, just 'train' and 'test'
    # train returns image, label, and name
    # test has no label, so just image and name
    dataset = SpaceNetDataset('train', img_dir, label_dir, scale=0.5)
    #dataset = BasicDataset(os.getcwd() +img_dir,os.getcwd() + label_dir, 0.5)

    batch_size = args.batch_size
    shuffle_dataset = True
    random_seed = 10

    #https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = dataset_size
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1, pin_memory=False, drop_last=True)

    UNet = UNet().cuda()
    UNet.load_state_dict(torch.load(args.load, map_location=torch.device('cuda')))

    UNet.eval()
    with torch.no_grad():
        for val_idx, val_data in enumerate(validation_loader):
            image = val_data['image'].cuda(non_blocking=True)
            mask = val_data['label'].cuda(non_blocking=True)

            mask_pred = UNet(image)
            mask_pred = torch.sigmoid(mask_pred)

            image_vis = np.squeeze(image.detach().cpu().numpy()[0]).transpose((1,2,0))
            mask_vis = np.squeeze(mask_pred.detach().cpu().numpy()[0])
            label_vis = np.squeeze(mask.detach().cpu().numpy()[0])

            fig, axs = plt.subplots(1,3)
            axs[0].imshow(image_vis,cmap='gray')
            axs[1].imshow(mask_vis,cmap='gray')
            axs[2].imshow(label_vis,cmap='gray')
        
            plt.show()