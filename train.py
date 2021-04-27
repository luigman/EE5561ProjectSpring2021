import matplotlib.pyplot as plt
from space_net_data_loader import *
from unet import *
import os
import argparse

def ParseCmdLineArguments():
    parser = argparse.ArgumentParser(description='CSCI 5563 HW3')
    parser.add_argument('--d', type=str,
                        default='0')
    parser.add_argument('--batch_size', type=int,
                        default='64')

    return parser.parse_args()

if __name__ == '__main__':
    args = ParseCmdLineArguments()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.d

    img_dir = '/SN1_buildings_train_AOI_1_Rio_3band/3band/'
    label_dir = '/building_mask/'

    # As of right now usage, just 'train' and 'test'
    # train returns image, label, and name
    # test has no label, so just image and name
    dataset = SpaceNetDataset('train', img_dir, label_dir)

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

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler)

    UNet = UNet().cuda()
    optimizer = torch.optim.Adam(UNet.parameters(), lr=1e-4)
    loss = nn.CrossEntropyLoss()

    for epoch in range(50):
        UNet.train()
        for train_idx, train_data in enumerate(train_loader):
            image = train_data['image'].cuda(non_blocking=True)
            mask = train_data['label'].cuda(non_blocking=True)

            mask_pred = UNet.forward(image)
            mask_pred = mask_pred + torch.randn(mask_pred.shape).cuda()*1e-2
            #mask_pred = mask_pred.squeeze(1)
            print(mask.shape)
            print(mask_pred.shape)

            loss_i = loss(mask_pred, mask.long())
            loss_i.backward()

        if epoch % 1 == 0:
            
            image_vis = np.squeeze(image.cpu()[0])
            mask_vis = np.squeeze(mask_pred.cpu()[0])
            label_vis = np.squeeze(mask.cpu()[0])

            fig, axs = plt.subplots(1,3)
            axs[0,0].imshow(image_vis)
            axs[0,1].imshow(mask_vis)
            axs[0,2].imshow(label_vis)
            fig.show()