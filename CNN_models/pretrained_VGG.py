import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import pytorch_lightning as pl
from PIL import Image
from pytorch_lightning.loggers import WandbLogger
from torchvision import models
import wandb


class VGG16Model(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        num_features = self.vgg16.classifier[6].in_features
        self.vgg16.classifier[6] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.vgg16(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        _, predicted = torch.max(y_hat, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0)
        self.log('train_accuracy', accuracy)
        return loss


class PolygonDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        # Define your classes
        self.class_names = ['1', '2', '3', '4', '5']

        # Get the list of image file names
        all_files = os.listdir(data_dir)
        self.image_files = [f for f in all_files if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.data_dir, img_file)
        img = Image.open(img_path).convert('RGB')  # Convert grayscale images to RGB

        # Load corresponding label
        label_file = img_file.replace('.jpg', '.txt')
        label_path = os.path.join(self.data_dir, label_file)
        with open(label_path, 'r') as f:
            label = int(f.readline().strip())

        # Transform image
        if self.transform:
            img = self.transform(img)

        return img, label


if __name__ == '__main__':
    # Data transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Parameters
    data_dir = './data/training'  # Update with your directory
    batch_size = 64
    num_workers = 4
    epochs = 250
    num_classes = 5

    # Dataset and DataLoader
    train_dataset = PolygonDataset(data_dir, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Model, Trainer, and Training
    model = VGG16Model(num_classes=num_classes)
    wandb.login()
    wandb.init(project="CNN-like-VGG shape detection project")
    wandb_logger = WandbLogger(project='CNN-like-VGG shape detection project', log_model=True)
   
