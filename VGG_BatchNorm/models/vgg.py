"""
VGG
"""
import numpy as np
from torch import nn
from .. import utils
# import sys
# sys.path.append(r"D:\python object\neural network\project2\task2\VGG_BatchNorm")
# import utils


# ## Models implementation
def get_number_of_parameters(model):
    parameters_n = 0
    for parameter in model.parameters():
        if parameter.requires_grad:  # 只计算可训练参数
            parameters_n += np.prod(parameter.shape).item()
    return parameters_n


# 完整VGG模型
class VGG_A(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10, init_weights_flag=True):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True), nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512), nn.ReLU(True),
            nn.Linear(512, 512), nn.ReLU(True),
            nn.Linear(512, num_classes)
        )
        if init_weights_flag:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            utils.nn.init_weights_(m)


# 更轻量的VGG模型
class VGG_A_Light(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        '''
        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        '''
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        # x = self.stage3(x)
        # x = self.stage4(x)
        # x = self.stage5(x)
        x = self.classifier(x.view(-1, 32 * 8 * 8))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            utils.nn.init_weights_(m)


# 全连接层启用drop out的完整的VGG模型
class VGG_A_Dropout(nn.Module):
    def __init__(self, inp_ch=3, num_classes=10):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.stage5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.classifier(x.view(-1, 512 * 1 * 1))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            utils.nn.init_weights_(m)


# 启用BatchNorm的更完整的VGG模型
class VGG_A_BatchNorm(nn.Module):
    """VGG_A model with Batch Normalization
    BN is typically added after Conv and before ReLU.
    """
    def __init__(self, inp_ch=3, num_classes=10, init_weights_flag=True,
                 batch_norm_2d=True, batch_norm_1d=True):  # Renamed init_weights
        super().__init__()
        self.batch_norm_2d = batch_norm_2d
        self.batch_norm_1d = batch_norm_1d

        self.features = nn.Sequential(
            # stage 1
            # Bias False common with BN
            nn.Conv2d(in_channels=inp_ch, out_channels=64, kernel_size=3, padding=1, bias=not self.batch_norm_2d),
            nn.BatchNorm2d(64) if self.batch_norm_2d else nn.Identity(),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=not self.batch_norm_2d),
            nn.BatchNorm2d(128) if self.batch_norm_2d else nn.Identity(),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=not self.batch_norm_2d),
            nn.BatchNorm2d(256) if self.batch_norm_2d else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1, bias=not self.batch_norm_2d),
            nn.BatchNorm2d(256) if self.batch_norm_2d else nn.Identity(),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage 4
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=not self.batch_norm_2d),
            nn.BatchNorm2d(512) if self.batch_norm_2d else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=not self.batch_norm_2d),
            nn.BatchNorm2d(512) if self.batch_norm_2d else nn.Identity(),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # stage5
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=not self.batch_norm_2d),
            nn.BatchNorm2d(512) if self.batch_norm_2d else nn.Identity(),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=not self.batch_norm_2d),
            nn.BatchNorm2d(512) if self.batch_norm_2d else nn.Identity(),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 在于VGG-A比较时，为了控制变量，全连接层不启用drop out
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512) if self.batch_norm_1d else nn.Identity(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512) if self.batch_norm_1d else nn.Identity(),
            nn.Linear(512, num_classes)
        )

        if init_weights_flag:
            self._init_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            utils.nn.init_weights_(m)


if __name__ == '__main__':
    print(f"VGG_A 参数量: {get_number_of_parameters(VGG_A())}")
    print(f"VGG_A_BatchNorm 参数量: {get_number_of_parameters(VGG_A_BatchNorm())}")
    print(f"VGG_A_Light 参数量: {get_number_of_parameters(VGG_A_Light())}")
    print(f"VGG_A_Dropout 参数量: {get_number_of_parameters(VGG_A_Dropout())}")
