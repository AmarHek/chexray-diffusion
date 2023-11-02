import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from cheff.machex import MaCheXDataset, MimicT2IDataset


class DataModuleFromConfig(pl.LightningDataModule):

    def __init__(self,
                 batch_size, machex_path, test_size, num_workers, mimic=False,
                 *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transforms = Compose([
            Resize(256),
            ToTensor(),
            Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

        if not mimic:
            self.machex = MaCheXDataset(machex_path, self.transforms)
        else:
            self.machex = MimicT2IDataset(machex_path, self.transforms)

        train_size = len(self.machex) - test_size
        self.train_dataset, self.test_dataset = random_split(
            self.machex,
            (train_size, test_size),
            generator=torch.Generator().manual_seed(1337)
        )

    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False
        )
        return loader
