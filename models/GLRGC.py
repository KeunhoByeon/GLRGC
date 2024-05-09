import torch.nn as nn
from torchvision.models import *


class GLRGC(nn.Module):
    def __init__(self, base_model, num_classes=2, **kwargs):
        super(GLRGC, self).__init__()

        backbone_model = resnet34

        self.student_A = backbone_model(**kwargs)
        self.student_B = backbone_model(**kwargs)
        self.teacher_A = backbone_model(**kwargs)
        self.teacher_B = backbone_model(**kwargs)

        if base_model == 'resnet18':
            self.base_model = resnet18(**kwargs)
        elif base_model == 'resnet101':
            self.base_model = resnet101(**kwargs)
        elif base_model == 'efficientnet_b0':
            self.base_model = efficientnet_b0(**kwargs)
        elif base_model == 'efficientnet_b4':
            self.base_model = efficientnet_b4(**kwargs)
        elif base_model == 'efficientnet_b7':
            self.base_model = efficientnet_b7(**kwargs)
        else:
            print('Model name {} is not implemented yet!'.format(base_model))
            raise TypeError

        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        output = self.base_model(x)
        output = self.fc(output)
        return output
