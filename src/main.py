import argparse
import glob
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from config import load_config
from resnet import resnet50
from trainer import Trainer
from utils import fix_seed, ModeChoiceError


def main(args):
    fix_seed(args.seed)
    model = resnet50()
    trainer = Trainer(config=vars(args), model=model)

    if args.train:
        t = time.time()
        trainer.train()
        train_time = time.time() - t
        m, s = divmod(train_time, 60)
        h, m = divmod(m, 60)
        print()
        print("Training Finished.")
        print("** Total Time: {}-hour {}-minute".format(int(h), int(m)))
    elif args.test:
        test_model = resnet50()
        state_dict = torch.load(glob.glob(os.path.join(args.ckpt_path, "*.pt"))[0])

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():  # dataparallel processing
            if "module" in k:
                k = k.replace("module.", "")
            new_state_dict[k] = v

        test_model.load_state_dict(new_state_dict)
        trainer.test(test_model)
    else:
        raise ModeChoiceError()


if __name__ == "__main__":
    args = load_config()
    main(args)
