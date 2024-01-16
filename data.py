import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import shutil


class IDDDataset(Dataset):

    def __init__(self,image_dir,mask_dir,transform = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.masks = os.listdir(mask_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"),dtype = np.float32)
        mask[mask == 255.0] = 0.0

        if self.transform is not None:
            augmentations = self.transform(image=image,mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        # print(mask.shape)
        return image,mask
    
def folder(image_dir,mask_dir,save_image_dir,save_mask_dir):
    for folder in os.listdir(image_dir):
        fold = os.path.join(image_dir,folder)
        fold_mask = os.path.join(mask_dir,folder)
        for file in os.listdir(fold):
            shutil.copy(fold+'/'+file,save_image_dir)
            filename,_ = file.split('_')
            mask_file = filename+'_label.png'
            shutil.copy(fold_mask+'/'+mask_file,save_mask_dir)

if __name__ == "__main__":
    image_dir = '/home/astonshingwolf/soham-devel/semseg/data/image/val'
    mask_dir = '/home/astonshingwolf/soham-devel/semseg/data/mask/val'
    save_image_dir = '/home/astonshingwolf/soham-devel/semseg/data_processed/val/image'
    save_mask_dir = '/home/astonshingwolf/soham-devel/semseg/data_processed/val/mask'
    folder(image_dir,mask_dir,save_image_dir,save_mask_dir)


