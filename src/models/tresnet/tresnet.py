from functools import partial

import torch
import torch.nn as nn
from collections import OrderedDict
from src.models.tresnet.layers.anti_aliasing import AntiAliasDownsampleLayer
from .layers.avg_pool import FastGlobalAvgPool2d
from .layers.squeeze_and_excite import SEModule
from src.models.tresnet.layers.space_to_depth import SpaceToDepthModule
import pytorch_lightning as ptl
from src.models.utils.score_utils import compute_scores, Statistics
import torch.nn.functional as F
import wandb

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        # self.tresnet = Tresnet_lightning(layers=[3, 4, 11, 3])
        if stride == 1:
            self.conv1 = self.conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = self.conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential( self.conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv2 = self.conv2d_ABN(planes, planes, stride=1, activation="identity")
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, reduce_layer_planes) if use_se else None

    def conv2d_ABN(self, ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
        return nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size=(kernel_size, kernel_size),  stride=stride, padding=kernel_size // 2, groups=groups,
                      bias=False),
            nn.BatchNorm2d(num_features=nf),
            nn.LeakyReLU(negative_slope=activation_param)
        )

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None: out = self.se(out)

        out += residual

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        # self.tresnet = Tresnet_lightning(layers=[3, 4, 11, 3])
        self.conv1 = self.conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation="leaky_relu",
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = self.conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation="leaky_relu",
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = self.conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation="leaky_relu",
                                        activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(self.conv2d_ABN(planes, planes, kernel_size=3, stride=1,
                                                      activation="leaky_relu", activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))

        self.conv3 = self.conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1,
                                activation="identity")

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def conv2d_ABN(self, ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=(kernel_size, kernel_size),  stride=stride, padding=kernel_size // 2, groups=groups,
                      bias=False),
            nn.BatchNorm2d(num_features=nf),
            nn.LeakyReLU(negative_slope=activation_param)
        )

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        if x is None:
            print('1: x is None!')
        out = self.conv1(x)
        if out is None:
            print('2: out is None!')
        out = self.conv2(out)
        if out is None:
            print('3: out is None!')
        if self.se is not None: out = self.se(out)

        out = self.conv3(out)
        if out is None:
            print('4: out is None!')
        out = out + residual  # no inplace
        out = self.relu(out)
        return out


class Tresnet_lightning(ptl.LightningModule):
    def __init__(self, layers, in_chans=3, num_classes=80, width_factor=1.0, remove_aa_jit=True):
        super(Tresnet_lightning, self).__init__()

        # JIT layers
        space_to_depth = SpaceToDepthModule()
        anti_alias_layer = partial(AntiAliasDownsampleLayer, remove_aa_jit=remove_aa_jit)
        global_pool_layer = FastGlobalAvgPool2d(flatten=True)

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        conv1 = self.conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 56x56
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)  # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer)  # 7x7

        # body
        self.body = nn.Sequential(OrderedDict([
            ('SpaceToDepth', space_to_depth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4)]))

        # head
        self.embeddings = []
        self.global_pool = nn.Sequential(OrderedDict([('global_pool_layer', global_pool_layer)]))
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        fc = nn.Linear(self.num_features, num_classes)
        self.head = nn.Sequential(OrderedDict([('fc', fc)]))

        # model initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                m.conv2[1].weight = nn.Parameter(torch.zeros_like(m.conv2[1].weight))  # BN to zero
            if isinstance(m, Bottleneck):
                m.conv3[1].weight = nn.Parameter(torch.zeros_like(m.conv3[1].weight))  # BN to zero
            if isinstance(m, nn.Linear): m.weight.data.normal_(0, 0.01)

    def conv2d_ABN(self, ni, nf, stride, activation="leaky_relu", kernel_size=3, activation_param=1e-2, groups=1):
        return nn.Sequential(
            nn.Conv2d(ni, nf, kernel_size=(kernel_size, kernel_size),  stride=stride, padding=kernel_size // 2, groups=groups,
                      bias=False),
            nn.BatchNorm2d(num_features=nf),
            nn.LeakyReLU(negative_slope=activation_param)
        )

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [self.conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                                  activation="identity")]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks): layers.append(
            block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body(x)
        self.embeddings = self.global_pool(x)
        logits = self.head(self.embeddings)
        return logits

    def bcewithlogits_loss(self, logits, labels):
        return F.binary_cross_entropy_with_logits(logits, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-4)
        return optimizer

    def lr_schedulers(self):
        optimizer = self.configure_optimizers()
        step_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        return step_lr_scheduler

    def training_step(self, train_batch, batch_idx):
        self.train_stats = Statistics()
        x, y = train_batch
        logits = self.forward(x)
        loss = self.bcewithlogits_loss(logits, y.float())
        preds = (logits.detach() >= 0.45)
        current_loss = loss.item() * x.size(0)
        scores = compute_scores(preds.cpu(), y.cpu())
        self.train_stats.update(float(current_loss), *scores)
        self.log('learning rate:', self.lr_schedulers().get_lr()[0])
        self.log('train acc', self.train_stats.precision())
        self.log('train loss', self.train_stats.loss())
        return loss

    def training_epoch_end(self, outputs):
        self.log('train acc on epoch', self.train_stats.precision())
        self.log('train loss on epoch', self.train_stats.loss())

    def validation_step(self, val_batch, batch_idx):
        self.val_stats = Statistics()
        x, y = val_batch
        logits = self.forward(x)
        loss = self.bcewithlogits_loss(logits, y.float())
        preds = (logits.detach() >= 0.45)
        current_loss = loss.item() * x.size(0)
        scores = compute_scores(preds.cpu(), y.cpu())
        self.val_stats.update(float(current_loss), *scores)
        # self.log('val acc', cum_stats.precision(), on_step=False, on_epoch=True)
        # self.log('val loss', cum_stats.loss(), on_step=False, on_epoch=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('val acc on epoch', self.val_stats.precision())
        self.log('val loss on epoch', self.val_stats.loss())


def TResnetM(model_params):
    """ Constructs a medium TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    remove_aa_jit = model_params['remove_aa_jit']
    model = Tresnet_lightning(layers=[3, 4, 11, 3], num_classes=num_classes, in_chans=in_chans,
                    remove_aa_jit=remove_aa_jit)
    return model


def TResnetL(model_params):
    """ Constructs a large TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    remove_aa_jit = model_params['remove_aa_jit']
    model = Tresnet_lightning(layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.2,
                    remove_aa_jit=remove_aa_jit)
    return model


def TResnetXL(model_params):
    """ Constructs an extra-large TResnet model.
    """
    in_chans = 3
    num_classes = model_params['num_classes']
    remove_aa_jit = model_params['remove_aa_jit']
    model = Tresnet_lightning(layers=[4, 5, 24, 3], num_classes=num_classes, in_chans=in_chans, width_factor=1.3,
                    remove_aa_jit=remove_aa_jit)

    return model


