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


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class Network(nn.Module):
    def __init__(self, base_model, num_classes=2, include_ema=False, **kwargs):
        super(Network, self).__init__()

        self.student = get_model(base_model, num_classes=num_classes)
        self.ema_temperature = kwargs.get("ema_temperature", 0.99)
        if include_ema:
            self.teacher = copy.deepcopy(self.student)
            for param in self.teacher.parameters():
                param.detach_()
        proj_in_channel = self.student.fc.in_features
        self.projection_head = ProjectionHead(input_dim=proj_in_channel, hidden_dim=proj_in_channel, output_dim=128)

    def update_ema(self):
        if not self.training:
            print("EMA update should only be called during training")
            raise AssertionError

        student_params = OrderedDict(self.student.named_parameters())
        teacher_params = OrderedDict(self.teacher.named_parameters())
        assert student_params.keys() == teacher_params.keys(), "Student and Teacher models are not aligned"

        # Update teacher model parameters
        with torch.no_grad():
            for (name, student_param), (_, teacher_param) in zip(student_params.items(), teacher_params.items()):
                teacher_param.data = self.ema_temperature * teacher_param.data + (1 - self.ema_temperature) * student_param.data

        student_buffers = OrderedDict(self.student.named_buffers())
        teacher_buffers = OrderedDict(self.teacher.named_buffers())
        assert student_buffers.keys() == teacher_buffers.keys(), "Buffers are not aligned"

        # Update buffers directly
        for (name, student_buffer), (_, teacher_buffer) in zip(student_buffers.items(), teacher_buffers.items()):
            teacher_buffer.data.copy_(student_buffer.data)

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

    def feed_projection_head(self, x):
        return self.projection_head(x)

    def forward(self, x, ema=False):
        if ema:
            return self.teacher(x)
        else:
            return self.student(x)
