import torch
import torch.nn as nn

from .loss import LocalContrastiveLoss, GlobalRelationLoss
from .network import Network


class GLRGC:
    def __init__(self, args, base_model, num_classes=2, **kwargs):
        self.args = args
        self.network = Network(base_model, num_classes=num_classes, **kwargs)
        self.optimizer = torch.optim.Adam(self.network.student.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.criterion = {
            "CrossEntropy": nn.CrossEntropyLoss(),
            "LocalContrastive": LocalContrastiveLoss(),
            "GlobalRelation": GlobalRelationLoss(),
            "Consistency": nn.MSELoss(),
        }

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.local_contrastive_loss = LocalContrastiveLoss()
        self.global_relation_loss = GlobalRelationLoss()
        self.consistency_loss = nn.MSELoss()

        self.loss_lambda = 0.1

    def train_one_epoch(self, train_loader):
        self.network.train()

        for i, (inputs, targets, is_noisy) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            self.optimizer.zero_grad()

            features = self.network.extract_feature(inputs)
            features_ema = self.network.extract_feature(inputs, ema=True)

            loss_global_relation = self.global_relation_loss(features, features_ema)

            output = self.network.feed_classifier(features)
            output_ema = self.network.feed_classifier(features_ema, ema=True)

            loss_consistency = self.consistency_loss(output, output_ema)
            loss_cross_entropy = self.cross_entropy_loss(output[~is_noisy], targets[~is_noisy])
            loss_local_contrastive = self.local_contrastive_loss(output[is_noisy], targets[is_noisy])

            loss = loss_cross_entropy + self.loss_lambda * (loss_global_relation + loss_local_contrastive + loss_consistency)
            loss.backward()

            self.optimizer.step()
            self.network.update_ema()

            preds = torch.argmax(output, dim=1)
            acc = torch.sum(preds == targets).item() / len(inputs) * 100.

            # # Log

            # num_progress, next_print = 0, args.print_freq
            # confusion_mat = [[0 for _ in range(args.num_classes)] for _ in range(args.num_classes)]

            # num_progress += len(inputs)
            # logger.add_history('total', {'loss': loss.item(), 'accuracy': acc})
            # logger.add_history('batch', {'loss': loss.item(), 'accuracy': acc})
            #
            # for t, p in zip(targets, preds):
            #     confusion_mat[int(t.item())][p.item()] += 1
            #
            # if num_progress >= next_print:
            #     if logger is not None:
            #         logger(history_key='batch', epoch=epoch, batch=num_progress, time=time.strftime('%Y.%m.%d.%H:%M:%S'))
            #     next_print += args.print_freq
            #
            # self.optimizer.param_groups[0]['lr']
