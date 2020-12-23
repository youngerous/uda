import logging
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataloader
from utils import EarlyStopping


class Trainer:
    def __init__(self, model: nn.Module, config: dict):
        super().__init__()
        # model hparam
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = config["transform"]
        self.path = config["path"]
        self.model = model.to(self.device)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)

        # training hparam
        self.epochs = config["epoch"]
        self.criterion = config["criterion"]
        self.batch_size = config["batch_size"]
        self.lr = config["lr"]
        self.optimizer = self.get_optimizer(config["optimizer"])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=20
        )
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloader()

        # model saving hparam
        self.save_path = config["ckpt_path"]
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.writer = SummaryWriter(self.save_path)
        self.global_step = 0
        self.eval_step = config["eval_step"]
        self.earlystopping = EarlyStopping(
            verbose=True, path=os.path.join(self.save_path, "best_model.ckpt")
        )

    def get_dataloader(self) -> Tuple[DataLoader]:
        return get_dataloader(self.path, batch_size=self.batch_size)

    def get_optimizer(self, optim) -> optim:
        assert optim in ["sgd", "adam"]
        if optim == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
            )
        elif config["optimizer"] == "adam":
            return optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self) -> None:
        for epoch in tqdm(range(self.epochs), desc="epoch"):
            val_loss = self._train_epoch(epoch)

            self.earlystopping(val_loss, self.model)
            if self.earlystopping.early_stop:
                print("Early Stopped.")
                break

        self.writer.close()

    def _train_epoch(self, epoch: int) -> float:
        train_loss = 0.0
        start_time = time.time()

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader), desc="steps", total=len(self.train_loader)
        ):
            img, label = map(lambda x: x.to(self.device), batch)

            output = self.model(img)

            self.optimizer.zero_grad()
            loss = self.criterion(output.squeeze(), label)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.global_step += 1
            if self.global_step % self.eval_step == 0:
                tqdm.write(
                    "global step: {}, train loss: {:.3f}".format(self.global_step, loss.item())
                )

        train_loss /= step + 1
        val_loss = self._valid_epoch(epoch)

        self.writer.add_scalars("loss", {"val": val_loss, "train": train_loss}, self.global_step)
        tqdm.write("** global step: {}, val loss: {:.3f}".format(self.global_step, val_loss))
        self.lr_scheduler.step(val_loss)

        elapsed = time.time() - start_time
        m, s = divmod(elapsed, 60)
        h, m = divmod(m, 60)
        tqdm.write("*** Epoch {} ends, it takes {}-hour {}-minute".format(epoch, int(h), int(m)))

        return val_loss

    def _valid_epoch(self, epoch: int) -> float:
        val_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader), desc="val steps", total=len(self.val_loader)
            ):
                img, label = map(lambda x: x.to(self.device), batch)

                output = self.model(img)
                loss = self.criterion(output.squeeze(), label)

                val_loss += loss.item()

        val_loss /= step + 1

        return val_loss

    def test(self) -> Tuple[float]:
        test_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.test_loader), desc="test steps", total=len(self.test_loader),
            ):
                img, label = map(lambda x: x.to(self.device), batch)

                output = self.model(img)
                loss = self.criterion(output.squeeze(), label)

                test_loss += loss.item()

        test_loss /= step + 1

        return test_loss

