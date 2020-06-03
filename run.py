from pathlib import Path
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

#
# Loading arguments
#
if len(sys.argv) <= 1:
    print("""
    =======
    JSON config helper
    =======
    TBA
    """)
    raise Exception('Please give json settings file path!')
args_p = Path(sys.argv[1])
if args_p.exists() is False:
    raise Exception('Path not found. Please check an argument again!')

with args_p.open(mode='r') as f:
    true = True
    false = False
    null = None
    args = json.load(f)


#
# Logger
#
import datetime
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

import logging
logfile = str('log/log-{}.txt'.format(run_start_time))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(logfile),
                        logging.StreamHandler(sys.stdout)
                    ])
logger = logging.getLogger(__name__)


#
# Settings
#
torch.manual_seed(args['seed'])
if args["gpu"] is True:
    torch.cuda.manual_seed_all(args['seed'])
if args["save_model"] is True:
    save_path = Path(f'./weights/{run_start_time}.pth')


def train():
    logger.info("***** Setup *****")
    logger.info(f"Configs: {args}")
    pass


def test():
    logger.info("***** Setup *****")
    logger.info(f"Configs: {args}")
    pass


def train_run():
    pass


def val_run():
    pass


def test_run():
    pass


def random():
    logger.info("***** Setup *****")
    logger.info(f"Configs: {args}")
    pass


def random_run():
    pass


def save_model(model):
    torch.save(model.state_dict(), save_path)


def main():
    if args['mode'] == 'train':
        train()
    elif args['mode'] == 'test':
        test()
    elif args['mode'] == 'random':
        random()
    else:
        raise NotImplementedError('Check the mode argument!')


if __name__ == '__main__':
    main()