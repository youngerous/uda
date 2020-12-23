import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from model.net import Model
from trainer import Trainer
from utils import str2bool, fix_seed


def main(args):
    config = {
        "path": args.path,
        "ckpt_path": args.ckpt_path,
        "batch_size": args.batch_size,
        "epoch": args.epoch,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "optimizer": args.optimizer,
        "criterion": nn.CrossEntropyLoss(),
        "eval_step": args.eval_step,
    }

    fix_seed(args.seed)
    model = Model()
    trainer = Trainer(model=model, config=config)

    t = time.time()
    trainer.train()
    train_time = time.time() - t
    m, s = divmod(train_time, 60)
    h, m = divmod(m, 60)

    print()
    print("Training Finished.")
    print("** Total Time: {}-hour {}-minute".format(int(h), int(m)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=711)
    parser.add_argument("--path", type=str, default="{your_path}")  ##
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--eval_step", type=int, default=50)

    args = parser.parse_args()
    main(args)
