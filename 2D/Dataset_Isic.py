import cv2
import os
from torch.utils.data import Dataset
from torchvision import transforms as t
import albumentations as al

image_path = './ISIC/img'
mask_path = './ISIC/label'

class MyDataset(Dataset):
    def __init__(self, df, transform=None, img_size=256):
        # It needs to be modified according to your own settings.
        self.df = df
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, item):
        row = self.df.iloc[item]
        fn = row.image_name
        img_size = self.img_size

        img = cv2.imread(os.path.join(image_path, fn[:12]+'.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        masks = cv2.imread(os.path.join(mask_path, fn), cv2.IMREAD_GRAYSCALE)/255
        masks = cv2.resize(masks, (img_size, img_size), interpolation=cv2.INTER_NEAREST)

        astentor = t.ToTensor()
        if self.transform :
            augmented = self.transform(image = img, mask = masks)
            img, masks = augmented['image'], augmented['mask']

        aug = al.Normalize(mean=(-0.2060, 0.1087, -1.1557), std=(0.7035, 0.5382, 1.0019), p=1)
        img = aug(image = img)['image']
        img = astentor(img)
        mask = astentor(masks)

        return img, mask

    def __len__(self):
        return len(self.df)

