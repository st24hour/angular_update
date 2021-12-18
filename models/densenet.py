import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# by JS
from torch.cuda.amp import autocast
from .gbn import GBN
from .gbn import GBN_invariant

# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, dropRate=0.0):
#         super(BasicBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.droprate = dropRate
#     def forward(self, x):
#         out = self.conv1(self.relu(self.bn1(x)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, training=self.training)
#         return torch.cat([x, out], 1)

# class BottleneckBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, dropRate=0.0):
#         super(BottleneckBlock, self).__init__()
#         inter_planes = out_planes * 4
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
#                                padding=0, bias=False)
#         self.bn2 = nn.BatchNorm2d(inter_planes)
#         self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.droprate = dropRate
#     def forward(self, x):
#         out = self.conv1(self.relu(self.bn1(x)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
#         out = self.conv2(self.relu(self.bn2(out)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
#         return torch.cat([x, out], 1)

# class TransitionBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, dropRate=0.0):
#         super(TransitionBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
#                                padding=0, bias=False)
#         self.droprate = dropRate
#     def forward(self, x):
#         out = self.conv1(self.relu(self.bn1(x)))
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
#         return F.avg_pool2d(out, 2)

# class DenseBlock(nn.Module):
#     def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
#         super(DenseBlock, self).__init__()
#         self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
#     def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
#         layers = []
#         for i in range(int(nb_layers)):
#             layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
#         return nn.Sequential(*layers)
#     def forward(self, x):
#         return self.layer(x)

# class DenseNet3(nn.Module):
#     def __init__(self, depth, num_classes, growth_rate=12,
#                  reduction=0.5, bottleneck=True, dropRate=0.0):
#         super(DenseNet3, self).__init__()
#         in_planes = 2 * growth_rate
#         n = (depth - 4) / 3
#         if bottleneck == True:
#             n = n/2
#             block = BottleneckBlock
#         else:
#             block = BasicBlock
#         # 1st conv before any dense block
#         self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         # 1st block
#         self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
#         in_planes = int(in_planes+n*growth_rate)
#         self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
#         in_planes = int(math.floor(in_planes*reduction))
#         # 2nd block
#         self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
#         in_planes = int(in_planes+n*growth_rate)
#         self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
#         in_planes = int(math.floor(in_planes*reduction))
#         # 3rd block
#         self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
#         in_planes = int(in_planes+n*growth_rate)
#         # global average pooling and classifier
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.fc = nn.Linear(in_planes, num_classes)
#         self.in_planes = in_planes

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 m.bias.data.zero_()
                    
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.trans1(self.block1(out))
#         out = self.trans2(self.block2(out))
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out = F.avg_pool2d(out, 8)
#         out = out.view(-1, self.in_planes)
#         return self.fc(out)
    
#     # function to extact the multiple features
#     def feature_list(self, x):
#         out_list = []
#         out = self.conv1(x)
#         out_list.append(out)
#         out = self.trans1(self.block1(out))
#         out_list.append(out)
#         out = self.trans2(self.block2(out))
#         out_list.append(out)
#         out = self.block3(out)
#         out = self.relu(self.bn1(out))
#         out_list.append(out)
#         out = F.avg_pool2d(out, 8)
#         out = out.view(-1, self.in_planes)
        
#         return self.fc(out), out_list
    
#     def intermediate_forward(self, x, layer_index):
#         out = self.conv1(x)
#         if layer_index == 1:
#             out = self.trans1(self.block1(out)) 
#         elif layer_index == 2:
#             out = self.trans1(self.block1(out))
#             out = self.trans2(self.block2(out))                
#         elif layer_index == 3:
#             out = self.trans1(self.block1(out))
#             out = self.trans2(self.block2(out))
#             out = self.block3(out)
#             out = self.relu(self.bn1(out))
#         return out
    
#     # function to extact the penultimate features
#     def penultimate_forward(self, x):
#         out = self.conv1(x)
#         out = self.trans1(self.block1(out))
#         out = self.trans2(self.block2(out))
#         out = self.block3(out)
#         penultimate = self.relu(self.bn1(out))
#         out = F.avg_pool2d(penultimate, 8)
#         out = out.view(-1, self.in_planes)
#         return self.fc(out), penultimate



'''
Refer:
https://github.com/kuangliu/pytorch-cifar/blob/ab908327d44bf9b1d22cd333a4466e85083d3f21/models/densenet.py#L68
'''
class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4*growth_rate)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, zero_init_residual=False):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)


    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def feature_list(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        feature_1 = out
        out = self.trans1(out)
        out = self.dense2(out)
        feature_2 = out
        out = self.trans2(out)
        out = self.dense3(out)
        feature_3 = out
        out = self.trans3(out)
        out = self.dense4(out)
        feature_4 = out
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)        
        return y, [feature_1, feature_2, feature_3, feature_4] # 96, 192, 384, 384






############################################################################################################
# Ghost batch norm DenseNet
class Bottleneck_GBN(nn.Module):
    def __init__(self, in_planes, growth_rate, eps=1e-05):
        super(Bottleneck_GBN, self).__init__()
        self.bn1 = GBN(in_planes, eps=eps)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = GBN(4*growth_rate, eps=eps)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out

class Bottleneck_GBN_bn_fix(nn.Module):
    def __init__(self, in_planes, growth_rate, eps=1e-05):
        super(Bottleneck_GBN_bn_fix, self).__init__()
        self.bn1 = GBN_invariant(in_planes, eps=eps)
        self.conv1 = nn.Conv2d(in_planes, 4*growth_rate, kernel_size=1, bias=False)
        self.bn2 = GBN_invariant(4*growth_rate, eps=eps)
        self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out

class Transition_GBN(nn.Module):
    def __init__(self, in_planes, out_planes, eps=1e-05):
        super(Transition_GBN, self).__init__()
        self.bn = GBN(in_planes, eps=eps)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class Transition_GBN_bn_fix(nn.Module):
    def __init__(self, in_planes, out_planes, eps=1e-05):
        super(Transition_GBN_bn_fix, self).__init__()
        self.bn = GBN(in_planes, eps=eps)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet_GBN(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, zero_init_residual=False):
        super(DenseNet_GBN, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition_GBN(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition_GBN(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition_GBN(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        self.bn = GBN(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class DenseNet_GBN_invariant(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, zero_init_residual=False, amp=True, eps=1e-05):
        super(DenseNet_GBN_invariant, self).__init__()
        self.amp = amp
        self.eps = eps
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition_GBN(num_planes, out_planes, eps=self.eps)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition_GBN(num_planes, out_planes, eps=self.eps)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition_GBN(num_planes, out_planes, eps=self.eps)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        # self.bn = GBN(num_planes)
        self.bn = GBN_invariant(num_planes, eps=self.eps)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, eps=self.eps))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.amp:
            with autocast():
                out = self.conv1(x)
                out = self.trans1(self.dense1(out))
                out = self.trans2(self.dense2(out))
                out = self.trans3(self.dense3(out))
                out = self.dense4(out)
                out = F.avg_pool2d(F.relu(self.bn(out)), 4)
                out = out.view(out.size(0), -1)
                out = self.linear(out)
                return out
        else:
            out = self.conv1(x)
            out = self.trans1(self.dense1(out))
            out = self.trans2(self.dense2(out))
            out = self.trans3(self.dense3(out))
            out = self.dense4(out)
            out = F.avg_pool2d(F.relu(self.bn(out)), 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out                

class DenseNet_GBN_bn_fix(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10, zero_init_residual=False, amp=True, eps=1e-05):
        super(DenseNet_GBN_bn_fix, self).__init__()
        self.amp = amp
        self.eps = eps
        self.growth_rate = growth_rate

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition_GBN_bn_fix(num_planes, out_planes, eps=self.eps)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition_GBN_bn_fix(num_planes, out_planes, eps=self.eps)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans3 = Transition_GBN_bn_fix(num_planes, out_planes, eps=self.eps)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        num_planes += nblocks[3]*growth_rate

        # self.bn = GBN(num_planes)
        self.bn = GBN_invariant(num_planes, eps=self.eps)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate, eps=self.eps))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.amp:
            with autocast():
                out = self.conv1(x)
                out = self.trans1(self.dense1(out))
                out = self.trans2(self.dense2(out))
                out = self.trans3(self.dense3(out))
                out = self.dense4(out)
                out = F.avg_pool2d(F.relu(self.bn(out)), 4)
                out = out.view(out.size(0), -1)
                out = self.linear(out)
                return out
        else:
            out = self.conv1(x)
            out = self.trans1(self.dense1(out))
            out = self.trans2(self.dense2(out))
            out = self.trans3(self.dense3(out))
            out = self.dense4(out)
            out = F.avg_pool2d(F.relu(self.bn(out)), 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out     


def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

# By js
def densenetBC100(num_classes):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12, num_classes=num_classes, zero_init_residual=False)
# By js
def densenetBC100_GBN(num_classes, zero_init_residual):
    return DenseNet_GBN(Bottleneck_GBN, [6,12,24,16], growth_rate=12, num_classes=num_classes, zero_init_residual=False)

def densenetBC100_GBN_invariant(num_classes, zero_init_residual, amp=True, eps=1e-05):
    return DenseNet_GBN_invariant(Bottleneck_GBN, [6,12,24,16], growth_rate=12, num_classes=num_classes, zero_init_residual=False, amp=amp, eps=eps)

def densenetBC100_GBN_bn_fix(num_classes, zero_init_residual, amp=True, eps=1e-05):
    return DenseNet_GBN_bn_fix(Bottleneck_GBN_bn_fix, [6,12,24,16], growth_rate=12, num_classes=num_classes, zero_init_residual=False, amp=amp, eps=eps)
    
def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()