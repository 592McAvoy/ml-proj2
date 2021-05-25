'''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
Without BN, the start learning rate should be 0.01
(c) YANG, Wei 
'''

'''
All standard network architectures for CIFAR-10 images (also
32x32 pixels) can be applied to this project.

You may read papers and do a survey on the network architecture, but
you do NOT need to try all architectures. 
Just learn >=2 DNNs
'''


# __all__ = ['alexnet']

"""
Alex block
"""




import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import math
class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride=1, padding=0, down_sample=False, add_bn=False):
        super(BasicBlock, self).__init__()
        components = [
            nn.Conv2d(in_c, out_c, kernel_size=k_size,
                      stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        ]
        if add_bn:
            components.append(nn.BatchNorm2d(out_c))
        # components.append(nn.ReLU(inplace=True))
        if down_sample:
            components.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.components = nn.Sequential(*components)

    def forward(self, x):
        # print(x.size())
        return self.components(x)


"""
Resnet block
"""


def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)


def conv1x1(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, 1, stride, 0, bias=False)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, k_size, stride=1, padding=0, down_sample=False, add_bn=False):
        super(ResBlock, self).__init__()
        if down_sample:
            self.conv0 = nn.Sequential(
                conv1x1(in_c, out_c, stride=2),
                nn.BatchNorm2d(out_c)
            )

            self.conv1 = nn.Sequential(
                conv3x3(in_c, out_c, stride=2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv0 = Identity()
            self.conv1 = nn.Sequential(
                conv3x3(in_c, out_c, stride=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.conv2 = nn.Sequential(
            conv3x3(out_c, out_c, stride=1),
            nn.BatchNorm2d(out_c)
        )

        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x += identity
        x = self.relu(x)

        return x


class DeepNeuralNetwork(BaseModel):
    def __init__(self, num_classes=10, block_type='basic', fea_base=64, n_layer=4):
        super(DeepNeuralNetwork, self).__init__()
        self.blk_dic = {
            "basic": BasicBlock,
            "basic_bn": BasicBlock,
            'res': ResBlock
        }
        self.n_layer = n_layer
        if block_type not in self.blk_dic.keys():
            raise NotImplementedError(block_type)

        block = self.blk_dic[block_type]

        use_bn = block_type != 'basic'

        # shrink the input size from (32, 32) to (8, 8)
        self.input_blk = BasicBlock(
            in_c=3, out_c=fea_base, k_size=7, stride=2, padding=3, down_sample=True, add_bn=use_bn)

        # extract features
        feature_ext = [
            block(in_c=fea_base, out_c=fea_base*2, k_size=3, padding=1,
                  down_sample=True, add_bn=use_bn),  # (8,8) -> (4,4)
            block(in_c=fea_base*2, out_c=fea_base*4, k_size=3, padding=1,
                  down_sample=True, add_bn=use_bn),  # (4,4) -> (2,2)
        ]
        for _ in range(n_layer-2):
            feature_ext.append(
                block(in_c=fea_base*4, out_c=fea_base*4, k_size=3, padding=1,
                      down_sample=False, add_bn=use_bn),
            )

        # self.feature_ext = nn.Sequential(
        #     block(in_c=fea_base, out_c=fea_base*2, k_size=3, padding=1,
        #           down_sample=True, add_bn=use_bn),  # (8,8) -> (4,4)
        #     block(in_c=fea_base*2, out_c=fea_base*4, k_size=3, padding=1,
        #           down_sample=True, add_bn=use_bn),  # (4,4) -> (2,2)
        #     block(in_c=fea_base*4, out_c=fea_base*4, k_size=3, padding=1,
        #           down_sample=False, add_bn=use_bn),  # (2,2) -> (2,2)
        #     block(in_c=fea_base*4, out_c=fea_base*4, k_size=3, padding=1,
        #           down_sample=True, add_bn=use_bn),  # (2,2) -> (1,1)
        # )
        self.feature_ext = nn.Sequential(*feature_ext)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(fea_base*4, num_classes)

        self._weight_init()

    def features(self, x):
        x = self.input_blk(x)
        x = self.feature_ext(x)

        return x
    
    def embedding(self, x, all_ret=False):
        if all_ret:
            ret = {}
            ret['input']=x.view(x.size(0), -1)
            x = self.input_blk(x)
            ret['input_blk']=x.view(x.size(0), -1)
            for i in range(self.n_layer):
                x = self.feature_ext[i](x)
                ret['fea_blk_{}'.format(i)] = x.view(x.size(0), -1)
            x = self.pooling(x)
            ret['pooling'] = x.view(x.size(0), -1)
            return ret
        else:
            x = self.features(x)
            x = self.pooling(x)
            x = x.view(x.size(0), -1)

            return x

    def classifier(self, x):
        # x=self.feature_ext[-1].components[3:](x)
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)


if __name__ == '__main__':
    for bt in ['basic', 'basic_bn', 'res']:
        net = DeepNeuralNetwork(block_type=bt)
        print(net)
        dummy = torch.rand((10, 3, 32, 32))
        net(dummy)
