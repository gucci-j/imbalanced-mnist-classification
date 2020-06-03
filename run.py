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

from src import DataProcessor, Model, EarlyStopping
from src import binary_accuracy, draw_det, draw_prc, draw_roc, draw_cm

#
# Loading arguments
#
if len(sys.argv) <= 1:
    print("""
    =======
    JSON config helper
    =======
    `mode` (str): Choose ``train'', ``test'' or ``random''.
    `seed` (int): Random seed.
    `batch_size` (int): Specify the size of a batch.
    `epochs` (int): Maximum number of epochs to train.
    `hidden_dim` (int): The number of units for the first hidden layer.
    `drop_rate` (float)  [0.0, 1.0): Specify a dropout rate.
    `patience` (int): Number of epochs to terminate the training process
                    if validation loss does not improve.
    `loss_correction` (bool): Whether to apply class weight correction.
    `gpu` (bool): Whether you want to use a GPU.
    `gpu_number` (int): When gpu is `true`, which gpu you want to use?.
    `save_model` (bool): Whether you want to save weights (train mode only).
    `weight_name` (str): Specify a path where weights are saved (test mode only).
    `split_rate` (float) [0.0, 1.0): The proportion of validation set.
    `data_ratio` (float) (0.0, 1.0]: The proportion of training data with an even label being preserved.
    """)
    raise Exception('Please give a json file path!')
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
    
    # make iterators
    data_proceessor = DataProcessor()
    train_data, val_data = data_proceessor.get_data(args['split_rate'], args['seed'])
    train_iterator = DataLoader(train_data, batch_size=args["batch_size"], shuffle=True)
    val_iterator = DataLoader(val_data, batch_size=args["batch_size"], shuffle=True)

    # build a model
    model = Model(input_dim=28 * 28, hidden_dim=args['hidden_dim'], drop_rate=args['drop_rate'])

    # define an optimizer
    optimizer = optim.Adam(model.parameters())
    if args['loss_correction'] is True:
        pos_weight = torch.tensor(pos_weight)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    # additional settings (e.g., early stopping)
    early_stopping = EarlyStopping(logger, patience=args['patience'], verbose=True)

    # for gpu environment
    if args['gpu'] is True and args['gpu_number'] is not None:
        torch.cuda.set_device(args['gpu_number'])
        device = torch.device('cuda')
        model = model.to(device)
        criterion = criterion.to(device)
    else:
        device = torch.device('cpu')
        model = model.to(device)
        criterion = criterion.to(device)

    logger.info(f"Number of training samples: {len(train_iterator.dataset)}")
    logger.info(f"Number of validation samples: {len(val_iterator.dataset)}")

    logger.info("***** Training *****")
    _history = []
    for epoch in range(args['epochs']):
        train_loss, train_acc = train_run(model, train_iterator, optimizer, criterion, device)
        logger.info(f'| Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.3f}% |')
        val_loss, val_acc = eval_run(model, val_iterator, criterion, device)
        logger.info(f'| Val. Loss: {val_loss:.3f} | Val. Acc: {val_acc*100:.3f}% |')
        _history.append([train_loss, train_acc, val_loss, val_acc])

        # early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info(f'  Early stopping at {epoch+1:02}')
            if args['save_model'] is True:
                save_model(model)
                break

    else: # end of the for loop
        if args['save_model'] is True:
            save_model(model)


    logger.info("***** Evaluation *****")
    # plot loss
    _history = np.array(_history)
    plt.figure()
    plt.plot(np.arange(len(_history)), _history[:, 0], label="train")
    plt.plot(np.arange(len(_history)), _history[:, 2], label='validation')
    plt.grid(True)
    plt.legend()
    plt.title("Training Monitoring")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('./fig/loss_{}.png'.format(run_start_time), bbox_inches="tight", pad_inches=0.1)

    # draw figures for evaluation
    _, _, val_auc, val_ap, val_eer = test_run(model, val_iterator, criterion, device)
    logger.info(f'| Val. AUC: {val_auc:.3f} | Val. AP: {val_ap:.3f} | Val. EER: {val_eer:.3f} |')


def test():
    logger.info("***** Setup *****")
    logger.info(f"Configs: {args}")
    pass


def train_run(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in iterator:
        batch = tuple(t.to(device) for t in batch)
        optimizer.zero_grad()
        output = model(batch[0])
        loss = criterion(output, batch[1])
        acc = binary_accuracy(output, batch[1])
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def eval_run(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in iterator:
            batch = tuple(t.to(device) for t in batch)
            predictions = model(batch[0])
            loss = criterion(predictions, batch[1])
            acc = binary_accuracy(predictions, batch[1])
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def test_run(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    y_pred_list = np.array([])
    y_true_list = np.array([])

    with torch.no_grad():
        for batch in iterator:
            batch = tuple(t.to(device) for t in batch)
            predictions = model(batch[0])
            loss = criterion(predictions, batch[1])
            acc = binary_accuracy(predictions, batch[1])
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()

            y_pred_list = np.append(y_pred_list, predictions.sigmoid().cpu().numpy())
            y_true_list = np.append(y_true_list, batch[1].float().cpu().numpy())
    
    draw_cm(y_pred_list, y_true_list, run_start_time)
    auc = draw_roc(y_pred_list, y_true_list, run_start_time)
    ap = draw_prc(y_pred_list, y_true_list, run_start_time)
    eer = draw_det(y_pred_list, y_true_list, run_start_time)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), auc, ap, eer


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