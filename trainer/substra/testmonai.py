import os
import sys
sys.path.append('.')

import torch
from io import BytesIO
from monai.networks.nets import DenseNet121
from monai.transforms import (Activations, AddChannel, AsDiscrete, Compose, LoadImage, RandFlip, RandRotate, RandZoom,
    ScaleIntensity, ToTensor,)
from monaiopener import MonaiOpener, MedNISTDataset
from monaialgo import MonaiAlgo
from monai.networks.layers import Norm
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.networks.nets import UNet
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from substraclient import Client
from common.utils import Mapping
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)

from pathlib import Path

if __name__ == '__main__':
    cwd = Path.cwd()
    print(cwd)

    datasetName = 'Task09_Spleen'
    data_path = os.path.join(cwd, 'datasets')
    data_dir = os.path.join(data_path, datasetName)
    folders = os.listdir(data_dir)
    #modelpath = os.path.join(home, "monaifl", "save","models","client")
    #model_dir = "./model/"

    mo = MonaiOpener(data_dir)
    # print(mo.data_summary(folders))
    # train, val, test = mo.get_x_y()
    train, val = mo.get_x_y()
    # print(f"Training count: {len(train)}, Validation count: {len(val)}, Test count: {len(test)}")

    ##transforms
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(96, 96, 96),
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(96, 96, 96),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Spacingd(keys=["image", "label"], pixdim=(
                1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            ScaleIntensityRanged(
                keys=["image"], a_min=-57, a_max=164,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    # monai algorithm object
    ma = MonaiAlgo()

    ma.act = Activations(softmax=True)
    ma.to_onehot = AsDiscrete(to_onehot=True, n_classes=mo.num_class)    # ma.test_loader = test_loader_workers=2)

    train_ds = Dataset(train, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=2)

    val_ds = Dataset(val, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2)

    # test_ds = Dataset(test, val_transforms)
    # test_loader = DataLoader(test_ds, batch_size=1, num_workers=2)


    # model initiliatization
    ma.model = UNet(
                    dimensions=3,
                    in_channels=1,
                    out_channels=2,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                    norm=Norm.BATCH)

    ma.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    ma.optimizer = torch.optim.Adam(ma.model.parameters(), 1e-4)
    ma.dice_metric = DiceMetric(include_background=False, reduction="mean")

    # number of epochs
    ma.epochs = 2

    # training/validation/testing datasets
    ma.train_ds = train_ds
    ma.val_ds = val_ds
    # ma.test_ds = test_ds

    # training/validation/testing data loaders
    ma.train_loader = train_loader
    ma.val_loader = val_loader
    # ma.test_loader = test_loader

    client = Client("localhost:50051")

    client.bootstrap(ma.model, ma.optimizer)

    # training and checkpoints
    checkpoint = Mapping()
    checkpoint = ma.train()
    # print(checkpoint)

    #aggregation request
    client.aggregate(ma.model, ma.optimizer, checkpoint)