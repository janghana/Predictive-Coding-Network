import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
import numpy as np
from PIL import Image
import wandb
from pytorch_lightning.loggers import WandbLogger

def accuracy(preds, labels):
    """
    Computes the accuracy of predictions.
    Args:
        preds (torch.Tensor): Tensor of predicted labels
        labels (torch.Tensor): Tensor of true labels
    Returns:
        float: Accuracy of predictions
    """
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    acc = correct / total
    return acc

class SimpleCNN(pl.LightningModule):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer7 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc1 = nn.Linear(1024 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1) 
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

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

class PolygonDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for i in range(60000):
            img_path = os.path.join(self.data_dir, f"{i}.jpg")
            label_path = os.path.join(self.data_dir, f"{i}.txt")
            img = Image.open(img_path)
            img = np.array(img)
            with open(label_path, 'r') as f:
                label = int(f.readline().strip()) - 1

            self.images.append(img)
            self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

if __name__ == '__main__':
    # Data transforms
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    wandb.login()

    wandb.init(project="CNN-like-VGG shape detection project")
    wandb_logger = WandbLogger(project='CNN-like-VGG shape detection project', log_model=True)

    # Parameters
    data_dir = './data/training'
    batch_size = 64
    num_workers = 4
    epochs = 250

    # Dataset and DataLoader
    train_dataset = PolygonDataset(data_dir, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    # Model, Trainer, and Training
    model = SimpleCNN()
    trainer = pl.Trainer(max_epochs=epochs, accelerator='gpu', logger=wandb_logger)
    trainer.fit(model, train_loader)

    # Save the model
    model_save_path = './model/model.pth'
    torch.save(model.state_dict(), model_save_path)

    wandb.finish()  
