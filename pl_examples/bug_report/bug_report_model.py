import os

import torch
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning import LightningModule, Trainer, LightningDataModule


class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class BoringDataModule(LightningDataModule):
    def __init__(self, batch_size=2):
        super().__init__()
        self.batch_size = batch_size

    def dataloader(self):
        print(f"loading dataloader with batch_size: {self.batch_size}")
        return DataLoader(RandomDataset(32, 64), batch_size=self.batch_size)

    train_dataloader = dataloader
    test_dataloader = dataloader
    val_dataloader = dataloader


class BoringModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(32, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("valid_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self(batch).sum()
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.1)


def run():

    datamodule = BoringDataModule()

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=os.getcwd(),
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        num_sanity_val_steps=0,
        max_epochs=1,
        enable_model_summary=False,
    )
    results = trainer.tune(model, datamodule)
    print(results)
    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)


if __name__ == "__main__":
    run()
