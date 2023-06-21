import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as transforms
import pytorch_lightning as pl
import cv2
import numpy as np
import pandas as pd
import os
import wandb
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from multiprocessing import freeze_support
from sklearn.preprocessing import LabelEncoder
from PIL import Image


def predict_image(image_path, model, transform):
    img = Image.open(image_path)
    img = np.array(img)  # Convert to numpy array
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
    return predicted.item()
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

class MyModel(pl.LightningModule):
    def __init__(self, num_layers, num_filters, learning_rate=0.001):
        super().__init__()
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.lr = learning_rate
        self.conv_layers = nn.ModuleList()
        in_channels = 3
        for i in range(self.num_layers):
            out_channels = self.num_filters * (2**int(i/2))
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.conv_layers.append(nn.ReLU(inplace=True))
            if i % 2 == 1:
                self.conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(in_channels, 5),
            nn.Softmax(dim=1)
        )
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler ,"monitor": "val_loss"}


    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y = y - 1
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        _, predicted = torch.max(y_hat, 1)
        correct = (predicted == y).sum().item()
        accuracy = correct / y.size(0)
        self.log('train_accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _ = self.compute_loss_and_accuracy(batch)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self.compute_loss_and_accuracy(batch)
        self.log('test_loss', loss)
        return {"test_loss": loss, "test_accuracy": acc}
    
    def compute_loss_and_accuracy(self, batch):
        x, y = batch
        y_hat = self.forward(x)
        # In your compute_loss_and_accuracy method, before calling nn.functional.cross_entropy
        y = y - 1
        # In your training/validation loop or dataset class
        loss = nn.functional.cross_entropy(y_hat, y)
        preds = y_hat.argmax(dim=1)
        acc = accuracy(preds, y)
        return loss, acc


class MyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, class_names, train_labels, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.class_names = class_names
        self.train_labels = train_labels
        self.num_workers = num_workers
    def setup(self, stage=None):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.data = []
        self.targets = []
        for i in range(60000):  # 데이터셋 크기 수정
            img = cv2.imread(os.path.join(self.data_dir, f"{i}.jpg"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = self.transform(img)
            self.data.append(img)
            with open(os.path.join(self.data_dir, f"{i}.txt")) as f:
                label = int(f.readline().strip())
            self.targets.append(label)
        self.data = torch.stack(self.data)
        self.targets = torch.tensor(self.targets)

    def train_dataloader(self):
        dataset = torch.utils.data.TensorDataset(self.data[:48000], self.targets[:48000])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        dataset = torch.utils.data.TensorDataset(self.data[48000:54000], self.targets[48000:54000])
        return DataLoader(dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        dataset = torch.utils.data.TensorDataset(self.data[54000:], self.targets[54000:])
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def get_progress_bar_dict(self):
        # Override this method to show class names instead of numerical labels in progress bar
        items = super().get_progress_bar_dict()
        if self.class_names:
            items["class_names"] = self.class_names
        return items

if __name__ == '__main__':
    freeze_support()
    batch_size = 32
    data_dir = "data/training"
    num_layers = 8
    num_filters = 64
    class_names = ['triangle', 'pentagon', 'hexagon', 'heptagon', 'octagon']
    train_labels = ['triangle', 'pentagon', 'hexagon', 'heptagon', 'octagon']
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    train_labels = torch.tensor(train_labels)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    wandb.login()
    
    wandb.init(project="CNN-like-VGG shape detection project")
    wandb_logger = WandbLogger(project='CNN-like-VGG shape detection project', log_model=True)

    model = MyModel(num_layers, num_filters)
    data_module = MyDataModule(data_dir, batch_size, class_names, train_labels, num_workers=56)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # callbacks = EarlyStoppingCallback(
    #     monitor="val_loss",
    #     min_delta=0.00,
    #     patience=3,
    #     verbose=False,
    #     mode="min"
    # )
    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator='gpu',
        max_epochs=250,
    )

    trainer.fit(model, data_module)
    trainer.test(datamodule=data_module)
    
    model_save_path = "./model/model.pth"
    torch.save(model.state_dict(), model_save_path)

    test_results = trainer.test(datamodule=data_module)

    # Load the model
    loaded_model = MyModel(num_layers, num_filters)
    loaded_model.load_state_dict(torch.load(model_save_path))
    loaded_model.eval()

    wandb.finish()

