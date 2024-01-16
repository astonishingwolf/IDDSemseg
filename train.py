import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    )

lr = 1e-4
Device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 4
num_epoch = 10
num_worker = 2
image_size = [160,240]
pin_memory = True
load_model = False
train_image_dir = '/home/astonshingwolf/soham-devel/semseg/data_processed/train/image'
train_mask_dir = '/home/astonshingwolf/soham-devel/semseg/data_processed/train/mask'
val_image_dir = '/home/astonshingwolf/soham-devel/semseg/data_processed/val/image'
val_mask_dir = '/home/astonshingwolf/soham-devel/semseg/data_processed/val/mask'

def train(loader,model,optimizer,loass_fn,scaler):
    loop = tqdm(loader)

    for batch_idx,(data,targets) in enumerate(loop):
        data = data.to(device = Device)
        # targets = targets.long().unsqueeze(1).to(device=Device)
        targets = targets.long().to(device=Device)
        # print(f"Shape of target{targets.shape}")
        # print(targets)
        # targets = torch.argmax(targets, dim=1)

        predictions = model(data)
        loss = loass_fn(predictions,targets)       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward
        # scaler.step(optimizer)
        # scaler.update()

def main():
    train_transform = A.Compose(
        [
            A.Resize(height = image_size[0], width = image_size[1]),
            A.Rotate(limit= 35,p = 1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0
            ),
            ToTensorV2(),
        ],
    )
    val_transform = A.Compose(
        [
            A.Resize(height = image_size[0], width = image_size[1]),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels = 3, out_channels = 7).to(Device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr = lr)
    train_loader, val_loader = get_loaders(
        train_image_dir,
        train_mask_dir,
        val_image_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        val_transform,
        num_worker,
        pin_memory
    )

    if load_model:
        pass

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(num_epoch):
        train(train_loader,model,optimizer,loss_fn,scaler)
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        check_accuracy(val_loader,model,device = Device)

    

if __name__ == "__main__":
    main()