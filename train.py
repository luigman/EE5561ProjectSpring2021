import matplotlib.pyplot as plt
from space_net_data_loader import *
from unet import *
from space_net_data_loader import SpaceNetDataset
import torch
from torch import optim
from torchvision import transforms
import torch.nn as nn
import os
import argparse

def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='CSCI 5563 HW3')
    parser.add_argument('--d', type=str,
                        default='0')
    parser.add_argument('--batch_size', type=int,
                        default='64')
    parser.add_argument('--save', type=str,
                        default='trained_models')
    parser.add_argument('--load', type=str,
                        default=None)

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
    validation_split = .2
    shuffle_dataset = True
    random_seed = 10

    #https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=1, pin_memory=False, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1, pin_memory=False, drop_last=True)

    UNet = UNet().cuda()
    if args.load is not None:
        UNet.load_state_dict(torch.load(args.load, map_location=torch.device('cuda')))
    #optimizer = torch.optim.Adam(UNet.parameters(), lr=1e-4)
    #loss = nn.CrossEntropyLoss() #used for multiclass segmentation

    optimizer = torch.optim.RMSprop(UNet.parameters(), lr=1e-4, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    loss = nn.BCEWithLogitsLoss()

    loss_train = []
    loss_val = []
    for epoch in range(50):
        UNet.train()
        for train_idx, train_data in enumerate(train_loader):
            image = train_data['image'].cuda(non_blocking=True)
            mask = train_data['label'].cuda(non_blocking=True)

            mask_pred = UNet(image)
            #mask_pred = mask_pred + torch.randn(mask_pred.shape).cuda()*1e-2

            loss_i = loss(mask_pred, mask)
            loss_train.append(loss_i.item())
            optimizer.zero_grad()
            loss_i.backward()
            nn.utils.clip_grad_value_(UNet.parameters(), 0.1)
            optimizer.step()
            
            print("Epoch:",epoch,"Iteration:",train_idx,"Loss",loss_train[-1])

        # Saving network's weights
        path = os.path.join(args.save, 'trained_model-%05d.ckpt' % epoch)
        torch.save(UNet.state_dict(), path)

        UNet.eval()
        with torch.no_grad():
            loss_val_i = []
            for val_idx, val_data in enumerate(validation_loader):
                image = val_data['image'].cuda(non_blocking=True)
                mask = val_data['label'].cuda(non_blocking=True)

                mask_pred = UNet(image)
                loss_i = loss(mask_pred, mask)
                loss_val_i.append(loss_i.item())

            loss_val.append(np.mean(loss_val_i))
            print("Epoch:",epoch,"Validation Loss",loss_val[-1])
            scheduler.step(loss_val[-1])

            image_vis = np.squeeze(image.cpu().numpy()[0]).transpose((1,2,0))
            mask_vis = np.squeeze(torch.sigmoid(mask_pred).cpu().numpy()[0])
            label_vis = np.squeeze(mask.cpu().numpy()[0])
            fig, axs = plt.subplots(1,3)
            axs[0].imshow(image_vis,cmap='gray')
            axs[0].title.set_text("Input Image")
            axs[1].imshow(mask_vis,cmap='gray')
            axs[1].title.set_text("Predicted Mask")
            axs[2].imshow(label_vis,cmap='gray')
            axs[2].title.set_text("Ground Truth")
        
            plt.show()
            fig, axs = plt.subplots(1,2)
            axs[0].plot(loss_train)
            axs[0].title.set_text("Training Loss")
            axs[0].set(xlabel='Iterations', ylabel='Loss')
            axs[1].plot(loss_val)
            axs[1].title.set_text("Validation Loss")
            axs[1].set(xlabel='Epochs', ylabel='Loss')
            plt.show()