import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import segmentation_models_pytorch as smp
import pandas as pd
from Dataset_Isic import MyDataset
from HmsUnet_nn import HmsUnet
import random
import numpy as np
import albumentations as al
from torch.nn.modules.loss import CrossEntropyLoss

imgsize = 256


def dice_coef_metric(output, target, smooth=1, eps = 1e-12):
    dice = (2.0 * torch.sum(output * target) + smooth) / (
            torch.sum(output) + torch.sum(target) + smooth + eps
    )
    return dice


def trans(flag):
    if flag == 'train':
        transforms = al.Compose([
            al.OneOf([al.HorizontalFlip(p=0.5),
                      al.VerticalFlip(p=0.5),
                      al.Transpose(p=0.5)]),
        ])
    else:
        transforms = None
    return transforms


def tr_main(fold):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    net = HmsUnet(num_classes=1, img_size=256,  in_chans=3)
    net.to(device)
    df_path = '/remote-home/hhhhh/Seg/ISIC/train_label.csv'
    df = pd.read_csv(df_path)
    batchsize = 24
    train_dataset = MyDataset(df[df.fold != fold], trans('train'), img_size=imgsize)
    validate_dataset = MyDataset(df[df.fold == fold], trans('test'), img_size=imgsize)
    train_loader = DataLoader(
        train_dataset, batch_size=batchsize, num_workers=8, shuffle=True, pin_memory=True
    )
    validate_loader = DataLoader(
        validate_dataset, batch_size=batchsize, num_workers=8, pin_memory=True
    )

    loss_function = smp.losses.DiceLoss(mode='binary')
    loss_function1 = CrossEntropyLoss()
    optimizer = optim.AdamW(params=net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20, eta_min=4e-8)

    best_acc = 0.0
    save_path = './weight/Isic_test' + str(fold) + '.pth'
    for epoch in range(100):
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, masks = data
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs= net(images)
            loss = 0.6*loss_function(outputs, masks)+0.4*loss_function1(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        net.eval()
        dice_list = []
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                pred_mask = (outputs > 0.5).float()
                acc = dice_coef_metric(pred_mask, val_labels.to(device))
                dice_list.append(acc)

            val_accurate = sum(dice_list) * 1.0 / len(dice_list)
            if best_acc < val_accurate:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            print('best Dice is : %.4f' % (best_acc))
            print('[epoch %d] train_loss: %.3f test_accuracy: %.3f' % (epoch + 1, running_loss / step, val_accurate))
        scheduler.step()
    print('Finish')

if __name__ == '__main__':
    tr_main(0) #0 means 0 fold. It needs to be modified according to your own settings.


