import argparse
import os
import random
import time

import torch

from dataset import NoisyDataset
from logger import Logger
from modules.GLRGC import GLRGC
from modules.NLF import NLF


def run(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    network_A = GLRGC(args, args.model, num_classes=args.num_classes, pretrained=args.pretrained, tag="A")
    network_B = GLRGC(args, args.model, num_classes=args.num_classes, pretrained=args.pretrained, tag="B")
    nlf = NLF()

    if args.resume is not None:  # resume
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

    train_dataset_A = NoisyDataset(args, args.data, input_size=args.input_size, stage="train", tag="A")
    train_loader_A = torch.utils.data.DataLoader(train_dataset_A, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    train_dataset_B = NoisyDataset(args, args.data, input_size=args.input_size, stage="train", tag="B")
    train_loader_B = torch.utils.data.DataLoader(train_dataset_B, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=True)
    val_dataset = NoisyDataset(args, args.data, input_size=args.input_size, stage="valid", tag="Valid")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    nlf_dataset = NoisyDataset(args, args.data, input_size=args.input_size, stage="train", tag="NLF")
    nlf_loader = torch.utils.data.DataLoader(nlf_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, shuffle=False)

    logger = Logger(os.path.join(args.result, 'log.txt'), epochs=args.epochs, dataset_size=len(train_loader_A.dataset), float_round=5)
    logger.set_sort(['loss', 'loss_CE', 'loss_local', 'loss_global', 'loss_const', 'accuracy', 'accuracy_ema', 'loss_lambda', 'lr', 'time'])
    logger(str(args))

    save_dir = os.path.join(args.result, 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    for epoch in range(args.start_epoch, args.epochs):
        network_A.train_one_epoch(train_loader_A, epoch=epoch, logger=logger)
        network_B.train_one_epoch(train_loader_B, epoch=epoch, logger=logger)

        noisy_data_A = nlf(network_A, nlf_loader)
        noisy_data_B = nlf(network_B, nlf_loader)
        nlf.step_threshold()

        train_dataset_A.update_noise_labels(noisy_data_B)
        train_dataset_B.update_noise_labels(noisy_data_A)

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            network_A.val(val_loader, epoch=epoch, logger=logger)
            network_B.val(val_loader, epoch=epoch, logger=logger)
            torch.save({
                "network_A": network_A.module.state_dict() if torch.cuda.device_count() > 1 else network_A.state_dict(),
                "network_B": network_B.module.state_dict() if torch.cuda.device_count() > 1 else network_B.state_dict(),
            }, os.path.join(save_dir, '{}.pth'.format(epoch)))


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
    parser.add_argument('--loss_lambda', default=10., type=float)
    parser.add_argument('--loss_lambda_warmup_duration', default=10, type=int)
    parser.add_argument('--warmup_ema', default=10, type=int)
    parser.add_argument('--seed', default=103, type=int, help='seed for initializing training.')
    # Validation and Debugging Arguments
    parser.add_argument('--val_freq', default=1, type=int, help='validation frequency')
    parser.add_argument('--print_freq', default=1000, type=int, help='print frequency')
    parser.add_argument('--result', default='results_classifier', type=str, help='path to results')
    parser.add_argument('--tag', default=None, type=str)
    args = parser.parse_args()

    # Paths setting
    args.data = os.path.expanduser(args.data)
    args.result = os.path.expanduser(args.result)
    args.result = os.path.join(args.result, time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
    if args.tag is not None:
        args.result = '{}_{}'.format(args.result, args.tag)
    os.makedirs(args.result, exist_ok=True)

    run(args)
