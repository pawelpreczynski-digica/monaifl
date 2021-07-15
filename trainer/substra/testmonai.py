import os
import torch
from io import BytesIO
from monai.networks.nets import DenseNet121
from monai.transforms import (Activations, AddChannel, AsDiscrete, Compose, LoadImage, RandFlip, RandRotate, RandZoom,
    ScaleIntensity, ToTensor,)
from monaiopener import MonaiOpener, MedNISTDataset
from monaialgo import MonaiAlgo
from substraclient import Client
from utils import Mapping

from pathlib import Path
home = str(Path.home())
print(home)


datapath= os.path.join(home, "monaifl", "trainer", "MONAI","data")
datasetName = 'MedNIST'
data_dir = os.path.join(datapath, datasetName)
folders = os.listdir(data_dir)
#modelpath = os.path.join(home, "monaifl", "save","models","client")
#model_dir = "./model/"

mo = MonaiOpener(data_dir)
print(mo.data_summary(folders))
train_x, train_y, val_x, val_y, test_x, test_y = mo.get_x_y(folders, 0.1, 0.1)
print(f"Training count: {len(train_x)}, Validation count: {len(val_x)}, Test count: {len(test_x)}")

##transforms
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        AddChannel(),
        ScaleIntensity(),
        RandRotate(range_x=15, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        ToTensor(),
    ]
)

val_transforms = Compose([LoadImage(image_only=True), AddChannel(), ScaleIntensity(), ToTensor()])

# monai algorithm object
ma = MonaiAlgo()

ma.act = Activations(softmax=True)
ma.to_onehot = AsDiscrete(to_onehot=True, n_classes=mo.num_class)

train_ds = MedNISTDataset(train_x, train_y, train_transforms)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)

val_ds = MedNISTDataset(val_x, val_y, val_transforms)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=64, num_workers=2)

test_ds = MedNISTDataset(test_x, test_y, val_transforms)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, num_workers=2)

# model initiliatization
ma.model = DenseNet121(spatial_dims=2, in_channels=1, out_channels=mo.num_class)#.to(device)

# model loss function
ma.loss_function = torch.nn.CrossEntropyLoss()

# model optimizer
ma.optimizer = torch.optim.Adam(ma.model.parameters(), 1e-5)

# number of epochs
ma.epochs = 1

# training/validation/testing datasets
ma.train_ds = train_ds
ma.val_ds = val_ds
ma.test_ds = test_ds

# training/validation/testing data loaders
ma.train_loader = train_loader
ma.val_loader = val_loader
ma.test_loader = test_loader

# training and checkpoints
checkpoint = Mapping()
checkpoint = ma.train()
print(checkpoint)

#creating client
client  = Client("localhost:50051")

#aggregation request
client.request(ma.model, ma.optimizer, checkpoint)