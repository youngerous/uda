import logging
import os
import sys
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_dataloader
from utils import EarlyStopping


class Trainer:
    def __init__(self, config: dict, model: nn.Module):
        super().__init__()
        self.config = config
        # model hparam
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.path = config["path"]
        self.model = model.to(self.device)
        if torch.cuda.is_available():
            self.model = nn.DataParallel(self.model)
        self.num_classes = config["num_classes"]
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.kld = nn.KLDivLoss(reduction="none")

        # training hparam
        self.epochs = config["epoch"]
        self.lr = config["lr"]
        self.optimizer = self.get_optimizer(config["optimizer"])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.1, patience=5
        )
        self.do_uda = config["do_uda"]
        self.batch_size_l = config["batch_size_l"]
        self.batch_size_v = config["batch_size_val"]
        self.num_labeled = config["num_labeled"]
        if self.do_uda:
            self.train_loader_l = get_dataloader(
                path=self.path,
                mode="train",
                batch_size=self.batch_size_l,
                num_labeled=self.num_labeled,
                labeled=True,
                uda=True,
            )
            self.batch_size_u = config["batch_size_u"]
            self.train_loader_u = get_dataloader(
                path=self.path,
                mode="train",
                batch_size=self.batch_size_u,
                num_labeled=self.num_labeled,
                labeled=False,
                uda=True,
            )
        else:
            self.train_loader_l = get_dataloader(
                path=self.path,
                mode="train",
                batch_size=self.batch_size_l,
                num_labeled=self.num_labeled,
                labeled=True,
                uda=False,
            )
        self.val_loader = get_dataloader(path=self.path, mode="val", batch_size=self.batch_size_v)
        self.test_loader = get_dataloader(path=self.path, mode="test", batch_size=self.batch_size_v)

        self.temperature = config["temperature"]
        self.confidence = config["confidence"]
        self.lmda = config["lambda"]
        self.tsa = config["tsa"]
        self.tsa_max_step = config["tsa_max_step"]

        assert self.tsa in [
            "linear",
            "exp",
            "log",
        ], "TSA scheduling method is only available among linear/exp/log."

        # model saving hparam
        self.save_path = config["ckpt_path"]
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.writer = SummaryWriter(self.save_path)
        self.global_step = 0
        self.eval_step = config["eval_step"]
        self.earlystopping = EarlyStopping(verbose=True, path=os.path.join(self.save_path))

    def get_tsa_threshold(self):
        K = self.num_classes
        T = self.tsa_max_step

        alpha = None
        if self.tsa == "exp":
            alpha = np.exp((self.global_step / T - 1) * 5)
        elif self.tsa == "linear":
            alpha = self.global_step / T
        else:
            alpha = 1 - np.exp(-(self.global_step / T) * 5)
        threshold = alpha * (1 - 1 / K) + 1 / K
        return threshold

    def get_optimizer(self, opt) -> optim:
        assert opt in [
            "sgd",
            "adam",
            "adamw",
        ], "For now you can only shoose 'sgd', 'adam', and 'adamw'."
        if opt == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"],
            )
        elif opt == "adam":
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif opt == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.lr)

    def train(self) -> None:
        self.udaiter = iter(self.train_loader_u) if self.do_uda else None

        for epoch in tqdm(range(self.epochs), desc="epoch"):
            val_loss = self._train_epoch(epoch)

            self.earlystopping(epoch, val_loss, self.model)
            if self.earlystopping.early_stop:
                print("Early Stopped.")
                break

        self.writer.close()

    def _train_epoch(self, epoch: int) -> float:
        train_loss = 0.0

        self.model.train()
        for step, batch in tqdm(
            enumerate(self.train_loader_l), desc="steps", total=len(self.train_loader_l)
        ):
            img, label = batch["labeled"], batch["label"]
            img, label = img.to(self.device), label.to(self.device)
            output = self.model(img)

            tsa_threshold, tsa_mask = None, True
            consistency_loss = 0.0
            if self.do_uda:
                try:
                    batch_uda = self.udaiter.next()
                except StopIteration:
                    self.udaiter = iter(self.train_loader_u)
                    batch_uda = self.udaiter.next()
                img_unlabeled, img_augmented = batch_uda["unlabeled"], batch_uda["augmented"]
                img_unlabeled, img_augmented = img_unlabeled.to(self.device), img_augmented.to(
                    self.device
                )
                self.model.eval()
                with torch.no_grad():
                    output_unlabeled = self.model(img_unlabeled)
                self.model.train()
                output_augmented = self.model(img_augmented)

                pred_unlabeled = F.softmax(output_unlabeled / self.temperature, dim=1)
                pred_augmented = F.softmax(output_augmented, dim=1)
                unlabeled_prob = pred_unlabeled.max(dim=1)[0]
                unlabeled_mask = unlabeled_prob.ge(self.confidence)

                consistency_loss = self.kld(pred_augmented, pred_unlabeled).mean()
                consistency_loss = consistency_loss * unlabeled_mask
                consistency_loss = consistency_loss.mean()

                # tsa
                tsa_threshold = self.get_tsa_threshold()
                pred_labeled = F.softmax(output, dim=1)
                onehot_label = torch.zeros(self.batch_size_l, self.num_classes).to(self.device)
                onehot_label[range(onehot_label.shape[0]), label.reshape(1, -1)] = 1
                prob_of_label = pred_labeled * onehot_label

                labeled_prob, labeled_idx = prob_of_label.max(dim=1)
                tsa_mask = labeled_prob.le(tsa_threshold)

            self.optimizer.zero_grad()
            sup_loss = self.criterion(output, label) * tsa_mask
            loss = sup_loss.mean() + self.lmda * consistency_loss
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            self.global_step += 1
            if self.global_step % self.eval_step == 0:
                tqdm.write(
                    "global step: {}, train loss: {:.3f}".format(self.global_step, loss.item())
                )
                if tsa_threshold:
                    tqdm.write("tsa_threshold: {:.5f}".format(tsa_threshold))

        train_loss /= step + 1
        val_loss = self._valid_epoch(epoch)

        self.writer.add_scalars("loss", {"val": val_loss, "train": train_loss}, self.global_step)
        tqdm.write("** global step: {}, val loss: {:.3f}".format(self.global_step, val_loss))
        self.lr_scheduler.step(val_loss)

        return val_loss

    def _valid_epoch(self, epoch: int) -> float:
        val_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.val_loader), desc="val steps", total=len(self.val_loader)
            ):
                img, label = batch["labeled"], batch["label"]
                img, label = img.to(self.device), label.to(self.device)

                output = self.model(img)
                loss = self.criterion(output.squeeze(), label).mean()

                val_loss += loss.item()

        val_loss /= step + 1

        return val_loss

    def test(self, best_model) -> Tuple[float]:
        test_loss = 0.0
        test_acc = 0.0
        correct = 0
        total = 0

        best_model = best_model.to(self.device)
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(self.test_loader),
                desc="test steps",
                total=len(self.test_loader),
            ):
                img, label = batch["labeled"], batch["label"]
                img, label = img.to(self.device), label.to(self.device)

                output = best_model(img)
                _, predicted = torch.max(output.data, 1)

                total += label.size(0)
                correct += (predicted == label).sum().item()
                loss = self.criterion(output.squeeze(), label).mean()

                test_loss += loss.item()

        test_loss /= step + 1
        print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))
        print("Test loss of the network on the 10000 test images: %.3f" % (test_loss))
        return test_loss
