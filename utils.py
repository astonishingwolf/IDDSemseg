import torch 
import torchvision
from data import IDDDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename = "my_checkpoint.pth.tar"):
    print("Saving Checkpoint")
    torch.save(state,filename)

def load_checkpoint(checkpoint,model):
    print("loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = IDDDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = IDDDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):

    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    class_iou = torch.zeros(7).to(device)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            softmax = torch.nn.Softmax(dim=1)
            preds = torch.argmax(softmax(model(x)),axis=1)
            preds = torch.argmax(model(x),axis=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            for class_idx in range(7):
                class_mask = (y == class_idx)
                pred_mask = (preds == class_idx)
                intersection = (class_mask & pred_mask).sum().item()
                union = (class_mask | pred_mask).sum().item()
                class_iou[class_idx] += intersection / (union + 1e-8)
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    # print(f"Dice score: {dice_score/len(loader)}")
    # mean_iou = torch.mean(class_iou) / len(loader)
    # print(f"Mean IoU: {mean_iou:.4f}")
    # model.eval()

# def check_accuracy(loader, model, device="cuda"):

#     num_correct = 0
#     num_pixels = 0
#     dice_score = 0
#     model.eval()

#     with torch.no_grad():
#         for x, y in loader:
#             x = x.to(device)
#             y = y.to(device)
#             softmax = torch.nn.Softmax(dim=1)
#             preds = torch.argmax(softmax(model(x)),axis=1)
#             preds = torch.argmax(model(x),axis=1)
#             num_correct += (preds == y).sum()
#             num_pixels += torch.numel(preds)
#             dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
#     print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
#     print(f"Dice score: {dice_score/len(loader)}")
#     model.eval()

def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()