import time

import torch
import torch.nn as nn

from .loss import LocalContrastiveLoss, GlobalRelationLoss
from .network import Network


class GLRGC(nn.Module):
    def __init__(self, args, base_model, num_classes=2, tag="", **kwargs):
        super(GLRGC, self).__init__()
        self.args = args
        self.tag = tag

        self.network = Network(base_model, num_classes=num_classes, include_ema=True, **kwargs)
        self.optimizer = torch.optim.Adam(self.network.student.parameters(), lr=args.lr, weight_decay=0.0001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)
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
        self.max_loss_lambda = self.args.loss_lambda
        self.loss_lambda_warmup_duration = self.args.loss_lambda_warmup_duration

    def forward(self, x, ema=False):
        return self.network(x, ema=ema)

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def val(self, val_loader, epoch=0, logger=None):
        self.network.eval()
        if logger is not None:
            logger.print_and_write_log(curr_line="Network {} validating".format(self.tag))

        mat, mat_ema = [0, 0], [0, 0]
        for i, (input_paths, inputs, targets, is_noisy) in enumerate(val_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()

            features = self.network.extract_feature(inputs)
            features_ema = self.network.extract_feature(inputs, ema=True)

            output = self.network.feed_classifier(features)
            output_ema = self.network.feed_classifier(features_ema, ema=True)

            loss_cross_entropy = self.cross_entropy_loss(output[~is_noisy], targets[~is_noisy])
            loss_local_contrastive = self.local_contrastive_loss(output[is_noisy], targets[is_noisy])
            loss_global_relation = self.global_relation_loss(features, features_ema)
            loss_consistency = self.consistency_loss(output, output_ema)
            loss = loss_cross_entropy + self.loss_lambda * (loss_global_relation + loss_local_contrastive + loss_consistency)

            preds = torch.argmax(output, dim=1)
            preds_ema = torch.argmax(output_ema, dim=1)

            mat[0] += torch.sum(preds == targets).item()
            mat[1] += len(targets)

            mat_ema[0] += torch.sum(preds_ema == targets).item()
            mat_ema[1] += len(targets)

            if logger is not None:
                logger.add_history('total', {'loss': loss.item(),
                                             'loss_CE': loss_cross_entropy.item(), 'loss_local': loss_local_contrastive.item(),
                                             'loss_global': loss_global_relation.item(), 'loss_const': loss_consistency.item()})

        if logger is not None:
            logger('*Validation {}'.format(epoch), history_key='total',
                   accuracy=round(mat[0] / mat[1] * 100, 4),
                   accuracy_ema=round(mat_ema[0] / mat_ema[1] * 100, 4),
                   time=time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))

        return mat[0] / mat[1] * 100, mat_ema[0] / mat_ema[1] * 100

    def train_one_epoch(self, train_loader, epoch=0, logger=None):
        self.network.train()
        if logger is not None:
            logger.print_and_write_log(curr_line="Network {} training".format(self.tag))

        mat, mat_ema = [0, 0], [0, 0]
        num_progress, next_print = 0, self.args.print_freq
        for i, (input_paths, inputs_1, inputs_2, targets, is_noisy) in enumerate(train_loader):
            if torch.cuda.is_available():
                inputs_1 = inputs_1.cuda()
                inputs_2 = inputs_2.cuda()
                targets = targets.cuda()

            self.optimizer.zero_grad()

            features = self.network.extract_feature(inputs_1)
            output = self.network.feed_classifier(features)
            with torch.no_grad():
                features_ema = self.network.extract_feature(inputs_2, ema=True)
                output_ema = self.network.feed_classifier(features_ema, ema=True)

            loss_cross_entropy = self.cross_entropy_loss(output[~is_noisy], targets[~is_noisy])
            loss_local_contrastive = self.local_contrastive_loss(output[is_noisy], targets[is_noisy])
            loss_global_relation = self.global_relation_loss(features, features_ema)
            loss_consistency = self.consistency_loss(output, output_ema)
            loss = loss_cross_entropy + self.loss_lambda * (loss_global_relation + loss_local_contrastive + loss_consistency)

            debug = False
            if torch.isnan(loss_cross_entropy).any() or torch.isinf(loss_cross_entropy).any():
                print("NaNs or Infs in loss_cross_entropy")
                debug=True
            if torch.isnan(loss_local_contrastive).any() or torch.isinf(loss_local_contrastive).any():
                print("NaNs or Infs in loss_local_contrastive")
                debug = True
            if torch.isnan(loss_global_relation).any() or torch.isinf(loss_global_relation).any():
                print("NaNs or Infs in loss_global_relation")
                debug = True
            if torch.isnan(loss_consistency).any() or torch.isinf(loss_consistency).any():
                print("NaNs or Infs in loss_consistency")
                debug=True
            if debug:
                import pdb; pdb.set_trace()

            if logger is not None:
                logger.add_history('total', {'loss': loss.item(), 'loss_CE': loss_cross_entropy.item(), 'loss_local': loss_local_contrastive.item(),
                                             'loss_global': loss_global_relation.item(), 'loss_const': loss_consistency.item()})
                logger.add_history('batch', {'loss': loss.item(), 'loss_CE': loss_cross_entropy.item(), 'loss_local': loss_local_contrastive.item(),
                                             'loss_global': loss_global_relation.item(), 'loss_const': loss_consistency.item()})

            loss.backward()

            self.optimizer.step()
            self.network.update_ema()

            # Log
            preds = torch.argmax(output, dim=1)
            mat[0] += torch.sum(preds == targets).item()
            mat[1] += len(targets)

            preds_ema = torch.argmax(output_ema, dim=1)
            mat_ema[0] += torch.sum(preds_ema == targets).item()
            mat_ema[1] += len(targets)

            num_progress += len(targets)
            if num_progress >= next_print:
                print(output.shape, targets.shape, output[~is_noisy].shape, targets[~is_noisy].shape, output[is_noisy].shape, targets[is_noisy].shape)
                if logger is not None:
                    logger(history_key='batch', epoch=epoch, batch=num_progress,
                           accuracy=round(mat[0] / mat[1] * 100, 4),
                           accuracy_ema=round(mat_ema[0] / mat_ema[1] * 100, 4),
                           time=time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))
                    next_print += self.args.print_freq

        if logger is not None:
            logger(history_key='total', epoch=epoch, batch=num_progress,
                   accuracy=round(mat[0] / mat[1] * 100, 4),
                   accuracy_ema=round(mat_ema[0] / mat_ema[1] * 100, 4),
                   lr=round(self.optimizer.param_groups[0]['lr'], 12),
                   loss_lambda=self.loss_lambda,
                   time=time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))

        if self.loss_lambda < self.max_loss_lambda:
            self.loss_lambda += self.max_loss_lambda / self.loss_lambda_warmup_duration
            self.loss_lambda = min(self.loss_lambda, self.max_loss_lambda)
        self.scheduler.step()

        return mat[0] / mat[1] * 100, mat_ema[0] / mat_ema[1] * 100
