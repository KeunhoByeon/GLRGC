import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, cohen_kappa_score
from termcolor import colored

from dataset import NoisyDataset
from logger import Logger
from modules.GLRGC import GLRGC


def scores(pred, true, num_classes, prefix, color):
    acc = np.mean(np.array(pred) == np.array(true))
    precision = precision_score(true, pred, average='macro')
    recall = recall_score(true, pred, average='macro')
    f1 = f1_score(true, pred, average='macro')
    kappa = cohen_kappa_score(true, pred, weights='quadratic')
    conf_mat = confusion_matrix(true, pred, labels=np.arange(num_classes))

    output = dict(acc=acc,
                  precision=precision,
                  recall=recall,
                  f1_score=f1,
                  kappa=kappa,
                  conf_mat=conf_mat,
                  )

    max_length = len(max(output.keys(), key=len))

    for metric in output:
        key = colored(prefix + '-' + metric.ljust(max_length) + ':', color)
        if metric in ['acc', 'f1_score', 'recall', 'precision', 'kappa']:
            print("-----%s" % key, end=' ')
            print("%0.7f" % output[metric])
        elif metric in ['conf_mat']:
            print("-----%s" % key, end=' ')
            conf_mat = output['conf_mat']
            conf_mat_df = pd.DataFrame(conf_mat)
            conf_mat_df.index.name = 'True'
            conf_mat_df.columns.name = 'Pred'
            output['conf_mat'] = conf_mat_df
            print('\n', conf_mat_df)
        else:
            continue

    return


def run(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    network_A = GLRGC(args, args.model, num_classes=args.num_classes, pretrained=args.pretrained, tag="A")
    network_B = GLRGC(args, args.model, num_classes=args.num_classes, pretrained=args.pretrained, tag="B")

    state_dict = torch.load(args.resume)
    network_A.load_state_dict(state_dict["network_A"])
    network_B.load_state_dict(state_dict["network_B"])

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            network_A.network = torch.nn.DataParallel(network_A.network).cuda()
            network_B.network = torch.nn.DataParallel(network_B.network).cuda()
        else:
            network_A.network = network_A.network.cuda()
            network_B.network = network_B.network.cuda()

    test_dataset = NoisyDataset(args, args.data, input_size=args.input_size, stage="test", tag="Test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    logger = Logger(os.path.join(args.result, 'log_test.txt'), epochs=args.epochs, dataset_size=len(test_loader.dataset), float_round=5)
    logger.set_sort(['loss', 'loss_CE', 'loss_local', 'loss_global', 'loss_const', 'accuracy', 'accuracy_ema', 'loss_lambda', 'lr', 'time'])
    logger(str(args))

    save_dir = os.path.join(args.result, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    test_pred, test_pred_ema, test_gt = network_A.val(test_loader, epoch="Test", logger=logger)
    scores(test_pred, test_gt, 4, "A student", "blue")
    scores(test_pred_ema, test_gt, 4, "A teacher", "red")
    test_pred, test_pred_ema, test_gt = network_B.val(test_loader, epoch="Test", logger=logger)
    scores(test_pred, test_gt, 4, "B student", "blue")
    scores(test_pred_ema, test_gt, 4, "B teacher", "red")


if __name__ == '__main__':
    # Arguments 설정
    parser = argparse.ArgumentParser(description='PyTorch Training')
    # Model Arguments
    parser.add_argument('--model', default='resnet34')
    parser.add_argument('--num_classes', default=4, type=int, help='number of classes')
    parser.add_argument('--pretrained', default=True, action='store_true', help='Load pretrained model.')
    parser.add_argument('--resume', default=None, type=str, help='path to latest checkpoint')
    # Data Arguments
    parser.add_argument('--data', default='colon')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--input_size', default=384, type=int, help='image input size')
    parser.add_argument('--random_crop', default=(0.8, 1.0))
    # Training Arguments
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.001, type=float, help='initial learning rate', dest='lr')
    parser.add_argument('--local_contrastive_weight', default=1., type=float)
    parser.add_argument('--loss_lambda', default=1., type=float)
    parser.add_argument('--loss_lambda_warmup_duration', default=10, type=int)
    parser.add_argument('--warmup_ema', default=10, type=int)
    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=1, type=int, help='validation frequency')
    parser.add_argument('--print_freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--tag', default="", type=str)
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.dirname(args.resume).replace("checkpoints", args.tag)
    os.makedirs(args.result, exist_ok=True)

    run(args)
