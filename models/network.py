import copy
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision.models import *


def get_model(model_name, **kwargs):
    if model_name == 'resnet34':
        model = resnet34(**kwargs)
    else:
        print('Model name {} is not implemented yet!'.format(model_name))
        raise NotImplementedError

    return model


class Network(nn.Module):
    def __init__(self, base_model, num_classes=2, include_ema=False, **kwargs):
        super(Network, self).__init__()

        self.student = get_model(base_model, num_classes=num_classes)
        if include_ema:
            self.teacher = copy.deepcopy(self.model)
            for param in self.teacher.parameters():
                param.detach_()

    def update_ema(self):
        if not self.training:
            print("EMA update should only be called during training")
            raise AssertionError

        model_params = OrderedDict(self.student.named_parameters())
        shadow_params = OrderedDict(self.teacher.named_parameters())
        assert model_params.keys() == shadow_params.keys()

        for name, param in model_params.items():
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.student.named_buffers())
        shadow_buffers = OrderedDict(self.teacher.named_buffers())
        assert model_buffers.keys() == shadow_buffers.keys()

        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def extract_feature(self, x, ema=False):
        if ema:
            model = self.teacher
        else:
            model = self.student

        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)

        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)

        x = model.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def feed_classifier(self, x, ema=False):
        if ema:
            return self.teacher.fc(x)
        else:
            return self.student.fc(x)

    def forward(self, x, ema=False):
        if ema:
            return self.teacher(x)
        else:
            return self.student(x)
