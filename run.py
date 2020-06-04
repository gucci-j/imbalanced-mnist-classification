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
from src import binary_accuracy, draw_det, draw_prc, draw_roc, draw_cm, compute_f1_prec_rec

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
fig_path = Path('fig') / run_start_time
if fig_path.exists():
    raise Exception('Path already exists!')
else:
    fig_path.mkdir()


def train():
    logger.info("***** Setup *****")
    logger.info(f"Configs: {args}")
    
    # make iterators
    data_proceessor = DataProcessor()
    train_data, val_data, pos_weight = data_proceessor.get_data(args['split_rate'], args['data_ratio'], args['seed'])
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
            logger.info(f'\tEarly stopping at {epoch+1:02}')
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
    plt.savefig('./fig/{}/loss.png'.format(run_start_time), bbox_inches="tight", pad_inches=0.1)

    # draw figures for evaluation
    _, _, val_auc, val_ap, val_eer, val_prec, val_rec, val_f1 = test_run(model, val_iterator, criterion, device)
    logger.info(f'| Val. AUC: {val_auc:.3f} | Val. AP: {val_ap:.3f} | Val. EER: {val_eer:.3f} | Val. Precision: {val_prec:.3f} |  Val. Recall: {val_rec:.3f} | Val. F1: {val_f1:.3f} |')


def test():
    logger.info("***** Setup *****")
    logger.info(f"Configs: {args}")

    # make iterators
    data_proceessor = DataProcessor()
    test_data = data_proceessor.get_test_data(args['data_ratio'])
    test_iterator = DataLoader(test_data, batch_size=args["batch_size"], shuffle=True)

    # build a model
    model = Model(input_dim=28 * 28, hidden_dim=args['hidden_dim'], drop_rate=args['drop_rate'])

    # load weights
    model_dict = model.state_dict()
    weights_dict = torch.load(args["weight_name"])
    model.load_state_dict(weights_dict)

    # define an optimizer
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()

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
    
    logger.info(f"Number of testing samples: {len(test_iterator.dataset)}")

    logger.info("***** Testing *****")
    _, test_acc, test_auc, test_ap, test_eer, test_prec, test_rec, test_f1 = test_run(model, test_iterator, criterion, device)
    logger.info(f'| Test Accuracy: {test_acc:.3f} | Test AUC: {test_auc:.3f} | Test AP: {test_ap:.3f} | Test EER: {test_eer:.3f} | Test Precision: {test_prec:.3f} |  Test Recall: {test_rec:.3f} | Test F1: {test_f1:.3f} |')



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
    f1, prec, rec = compute_f1_prec_rec(y_pred_list, y_true_list)

    return epoch_loss / len(iterator), epoch_acc / len(iterator), auc, ap, eer, prec, rec, f1


def random():
    logger.info("***** Setup *****")
    logger.info(f"Configs: {args}")

    # make iterators
    data_proceessor = DataProcessor()
    train_data, val_data, pos_weight = data_proceessor.get_data(args['split_rate'], args['data_ratio'], args['seed'])
    test_data = data_proceessor.get_test_data(args['data_ratio'])
    train_iterator = DataLoader(train_data, batch_size=args["batch_size"], shuffle=True)
    val_iterator = DataLoader(val_data, batch_size=args["batch_size"], shuffle=True)
    test_iterator = DataLoader(test_data, batch_size=args["batch_size"], shuffle=True)
    
    logger.info(f"Number of training samples: {len(train_iterator.dataset)}")
    logger.info(f"Number of validation samples: {len(val_iterator.dataset)}")
    logger.info(f"Number of testing samples: {len(test_iterator.dataset)}")

    # create a naive classifier
    # https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    y_true_list = np.array([])
    with torch.no_grad():
        for batch in train_iterator:
            y_true_list = np.append(y_true_list, batch[1].float().cpu().numpy())
    model = DummyClassifier(strategy="stratified") # "most_frequent" or 'stratified'
    X = np.zeros_like(y_true_list).reshape(-1, 1)
    model.fit(X, y_true_list)

    logger.info("***** Testing *****")
    val_acc, val_auc, val_ap, val_eer, val_prec, val_rec, val_f1 = random_run(model, val_iterator)
    test_acc, test_auc, test_ap, test_eer, test_prec, test_rec, test_f1 = random_run(model, test_iterator)
    logger.info(f'| Val. Accuracy: {test_acc:.3f} | Val. AUC: {test_auc:.3f} | Val. AP: {test_ap:.3f} | Val. EER: {test_eer:.3f} | Val. Precision: {test_prec:.3f} |  Val. Recall: {test_rec:.3f} | Val. F1: {test_f1:.3f} |')
    logger.info(f'| Test Accuracy: {test_acc:.3f} | Test AUC: {test_auc:.3f} | Test AP: {test_ap:.3f} | Test EER: {test_eer:.3f} | Test Precision: {test_prec:.3f} |  Test Recall: {test_rec:.3f} | Test F1: {test_f1:.3f} |')


def random_run(model, iterator):
    y_true_list = []

    # combine data
    with torch.no_grad():
        for batch in iterator:
            y_true_list = np.append(y_true_list, batch[1].float().numpy())

    # predict
    X = np.zeros_like(y_true_list).reshape(-1, 1)
    preds = model.predict(X)

    # compute metrics
    acc = accuracy_score(y_true_list, (preds >= 0.5).astype(int))
    draw_cm(preds, y_true_list, run_start_time)
    auc = draw_roc(preds, y_true_list, run_start_time)
    ap = draw_prc(preds, y_true_list, run_start_time)
    eer = draw_det(preds, y_true_list, run_start_time)
    f1, prec, rec = compute_f1_prec_rec(preds, y_true_list)

    return acc, auc, ap, eer, prec, rec, f1


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