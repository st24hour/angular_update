'''ResNet in PyTorch.
BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
Original code is from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.cuda.amp import autocast

import init_js
from .gbn import GBN
from .gbn import GBN_invariant


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv3x3_bias(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, eps=1e-05, momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=momentum, eps=eps)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=momentum, eps=eps)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, momentum=momentum, eps=eps)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.add(out, self.shortcut(x))
        out = F.relu(out)

        # print(self.bn1.running_mean)
        # print(self.bn1.running_var)
        # print(self.bn2.running_mean)
        # print(self.bn2.running_var)
        return out

class BasicBlock_GBN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, eps=1e-05):
        super(BasicBlock_GBN, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = GBN(planes, eps=eps)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = GBN(planes, eps=eps)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                GBN(self.expansion*planes, eps=eps)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.add(out, self.shortcut(x))
        out = F.relu(out)
        return out

# This module is used in resnet18_invariant -> not used in ECCV 2022 submission and 2021 neurips submission
class BasicBlock_GBN_invariant(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_GBN_invariant, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = GBN(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = GBN_invariant(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                GBN_invariant(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = torch.add(out, self.shortcut(x))
        out = F.relu(out)
        return out

class BasicBlock_LN(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, eps=1e-05):
        super(BasicBlock_LN, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.ln1 = nn.LayerNorm(planes, eps=eps)
        self.conv2 = conv3x3(planes, planes)
        self.ln2 = nn.LayerNorm(planes, eps=eps)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            #     nn.LayerNorm(self.expansion*planes, eps=eps)
            # )
            self.sc_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.sc_ln = nn.LayerNorm(self.expansion*planes, eps=eps)

    # def forward(self, x):
    #     out = F.relu(self.ln1(self.conv1(x)))
    #     out = self.ln2(self.conv2(out))
    #     out = torch.add(out, self.shortcut(x))
    #     out = F.relu(out)
    #     return out
    def forward(self, x):
        out = self.conv1(x)
        out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        out = F.relu(self.ln1(out))
        out = out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        out = self.conv2(out)
        out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        out = self.ln2(out)
        out = out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            shortcut = self.sc_conv(x)
            shortcut = shortcut.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            shortcut = self.sc_ln(shortcut)
            shortcut = shortcut.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        else:
            shortcut = self.shortcut(x)
        
        out = torch.add(out, shortcut)
        out = F.relu(out)
        return out

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, eps=1e-05):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, eps=eps)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=eps)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes, eps=eps)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck_GBN(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, eps=1e-05):
        super(Bottleneck_GBN, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = GBN(planes, eps=eps)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = GBN(planes, eps=eps)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = GBN(self.expansion*planes, eps=eps)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                GBN(self.expansion*planes, eps=eps)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        out = torch.add(out, self.shortcut(x))
        out = F.relu(out)
        return out

class Bottleneck_GBN_invariant(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_GBN_invariant, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = GBN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = GBN(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = GBN_invariant(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                GBN_invariant(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        out = torch.add(out, self.shortcut(x))
        out = F.relu(out)
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=10, zero_init_residual=False, amp=True, eps=1e-05, momentum=0.1):
        super(ResNet, self).__init__()
        self.amp = amp
        self.eps = eps        
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64, eps=self.eps)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init_js.constant_(m.weight, 1)
                init_js.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    @autocast()
    def forward(self, x):
        if self.amp:
            with autocast():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                y = self.linear(out)
                # print(self.bn1.running_var)
                return y
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            # print(self.bn1.running_var)
            return y            

class ResNet_GBN(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=10, zero_init_residual=False):
        super(ResNet_GBN, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = GBN(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init_js.constant_(m.weight, 1)
                init_js.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     init_js.normal_(m.weight, 0, 0.01)
            #     init_js.normal_(m.bias, 0, 0.01)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_GBN):
                    nn.init.constant_(m.bn3.bn.weight, 0)
                elif isinstance(m, BasicBlock_GBN):
                    nn.init.constant_(m.bn2.bn.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    @autocast()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        # print(self.bn1.running_var)
        return y

class ResNet_GBN_invariant(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=10, zero_init_residual=False):
        super(ResNet_GBN_invariant, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = GBN(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_invariant(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    init_js.constant_(m.weight, 1)
                    init_js.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     init_js.normal_(m.weight, 0, 0.01)
            #     init_js.normal_(m.bias, 0, 0.01)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_GBN):
                    nn.init.constant_(m.bn3.bn.weight, 0)
                elif isinstance(m, BasicBlock_GBN):
                    nn.init.constant_(m.bn2.bn.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_invariant(self, block, planes, num_blocks, stride):
        # first and last block should be scale invariant  
        if block == BasicBlock_GBN:
            block_invariant = BasicBlock_GBN_invariant
        else:  
            block_invariant = Bottleneck_GBN_invariant
            
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block_invariant(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    @autocast()
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        # print(self.bn1.running_var)
        return y

class ResNet_GBN_invariant2(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=10, zero_init_residual=False, amp=True, eps=1e-05, width=1.0, **kwargs):
        super(ResNet_GBN_invariant2, self).__init__()
        self.amp = amp
        self.eps = eps
        self.in_planes = int(64*width)

        self.conv1 = conv3x3(3,int(64*width))
        self.bn1 = GBN(int(64*width), eps=self.eps)
        self.layer1 = self._make_layer(block, int(64*width), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*width), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*width), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*width), num_blocks[3], stride=2)
        self.bn2 = GBN_invariant(int(512*width), eps=self.eps)
        self.linear = nn.Linear(int(512*width)*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    init_js.constant_(m.weight, 1)
                    init_js.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     init_js.normal_(m.weight, 0, 0.01)
            #     init_js.normal_(m.bias, 0, 0.01)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_GBN):
                    nn.init.constant_(m.bn3.bn.weight, 0)
                elif isinstance(m, BasicBlock_GBN):
                    nn.init.constant_(m.bn2.bn.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, eps=self.eps))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.amp:
            with autocast():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.bn2(out)
                out = F.avg_pool2d(out, 4)
                # out = F.adaptive_avg_pool2d(out, (1,1))
                out = out.view(out.size(0), -1)
                y = self.linear(out)
                return y
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.bn2(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            return y

class ResNet_LN_invariant(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=10, zero_init_residual=False, amp=True, eps=1e-05, width=1.0, **kwargs):
        super(ResNet_LN_invariant, self).__init__()
        self.amp = amp
        self.eps = eps
        self.in_planes = int(64*width)

        self.conv1 = conv3x3(3,int(64*width))
        self.ln1 = nn.LayerNorm(int(64*width), eps=self.eps)
        self.layer1 = self._make_layer(block, int(64*width), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*width), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*width), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*width), num_blocks[3], stride=2)
        self.ln2 = nn.LayerNorm(int(512*width), eps=self.eps, elementwise_affine=False)
        self.linear = nn.Linear(int(512*width)*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            # elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            #     if m.weight is not None:
            #         init_js.constant_(m.weight, 1)
            #         init_js.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_GBN):
                    nn.init.constant_(m.bn3.bn.weight, 0)
                elif isinstance(m, BasicBlock_GBN):
                    nn.init.constant_(m.bn2.bn.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, eps=self.eps))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.amp:
            with autocast():
                out = self.conv1(x)
                out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
                out = F.relu(self.ln1(out))
                out = out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
                out = self.ln2(out)
                out = out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
                out = F.avg_pool2d(out, 4)
                # out = F.adaptive_avg_pool2d(out, (1,1))
                out = out.view(out.size(0), -1)
                y = self.linear(out)
                return y
        else:
            out = self.conv1(x)
            out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            out = F.relu(self.ln1(out))
            out = out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            out = self.ln2(out)
            out = out.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
            out = F.avg_pool2d(out, 4)
            # out = F.adaptive_avg_pool2d(out, (1,1))
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            return y   

class ResNet_GBN_invariant2_avg(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=10, zero_init_residual=False, amp=True, eps=1e-05):
        super(ResNet_GBN_invariant2_avg, self).__init__()
        self.amp = amp
        self.eps = eps
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = GBN(64, eps=self.eps)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn2 = GBN_invariant(512, eps=self.eps)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    init_js.constant_(m.weight, 1)
                    init_js.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     init_js.normal_(m.weight, 0, 0.01)
            #     init_js.normal_(m.bias, 0, 0.01)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_GBN):
                    nn.init.constant_(m.bn3.bn.weight, 0)
                elif isinstance(m, BasicBlock_GBN):
                    nn.init.constant_(m.bn2.bn.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, eps=self.eps))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.amp:
            with autocast():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                out = self.bn2(out)
                out = out.view(out.size(0), -1)
                y = self.linear(out)
                return y
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = self.bn2(out)
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            return y


class ResNet_invariant(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=10, zero_init_residual=False, amp=True, eps=1e-05, momentum=0.1):
        super(ResNet_invariant, self).__init__()
        self.amp = amp
        self.eps = eps
        self.momentum = momentum
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.momentum, eps=self.eps)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512*block.expansion, momentum=self.momentum, affine=False, eps=self.eps)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    init_js.constant_(m.weight, 1)
                    init_js.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, eps=self.eps, momentum=self.momentum))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.amp:
            with autocast():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.bn2(out)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                y = self.linear(out)
                return y
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.bn2(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            return y


class ResNet_invariant_avg(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=10, zero_init_residual=False, amp=True, eps=1e-05, momentum=0.1):
        super(ResNet_invariant_avg, self).__init__()
        self.amp = amp
        self.eps = eps
        self.momentum = momentum
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64, momentum=self.momentum, eps=self.eps)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512*block.expansion, momentum=self.momentum, affine=False, eps=self.eps)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    init_js.constant_(m.weight, 1)
                    init_js.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, eps=self.eps, momentum=self.momentum))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.amp:
            with autocast():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.bn2(out)
                out = F.avg_pool2d(out, 4)
                out = out.view(out.size(0), -1)
                y = self.linear(out)
                return y
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = self.bn2(out)
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            return y
        
class ResNet_imagenet(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=1000, zero_init_residual=False):
        super(ResNet_imagenet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.conv1 = conv3x3(3,64)
        # self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init_js.constant_(m.weight, 1)
                init_js.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    @autocast()
    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        y = self.linear(out)
        # print(self.bn1.running_var)
        return y


class ResNet_GBN_imagenet(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=1000, zero_init_residual=True, amp=True):
        super(ResNet_GBN_imagenet, self).__init__()
        self.amp = amp
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = GBN(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.conv1 = conv3x3(3,64)
        # self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.normal_(m.bias, 0, 0.01)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_GBN):
                    nn.init.constant_(m.bn3.bn.weight, 0)
                elif isinstance(m, BasicBlock_GBN):
                    nn.init.constant_(m.bn2.bn.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    

    def forward(self, x):
        if self.amp:
            with autocast():
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.maxpool(out)
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.avgpool(out)
                # out = torch.flatten(out, 1)
                out = out.view(out.size(0), -1)
                y = self.linear(out)
                # print(self.bn1.running_var)
                return y
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            # out = torch.flatten(out, 1)
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            # print(self.bn1.running_var)
            return y

class ResNet_GBN_imagenet_invariant(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=1000, zero_init_residual=True, amp=True):
        super(ResNet_GBN_imagenet_invariant, self).__init__()
        self.amp = amp
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = GBN(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.conv1 = conv3x3(3,64)
        # self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer_invariant(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.normal_(m.bias, 0, 0.01)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_GBN):
                    nn.init.constant_(m.bn3.bn.weight, 0)
                elif isinstance(m, BasicBlock_GBN):
                    nn.init.constant_(m.bn2.bn.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _make_layer_invariant(self, block, planes, num_blocks, stride):
        # first and last block should be scale invariant  
        if block == BasicBlock_GBN:
            block_invariant = BasicBlock_GBN_invariant
        else:  
            block_invariant = Bottleneck_GBN_invariant
            
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block_invariant(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
            
    def forward(self, x):
        with autocast():
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.maxpool(out)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            return y


class ResNet_GBN_imagenet_invariant2(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=1000, zero_init_residual=True, amp=True, eps=1e-05):
        super(ResNet_GBN_imagenet_invariant2, self).__init__()
        self.amp = amp
        self.eps = eps
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = GBN(self.in_planes, eps=self.eps)
        self.bn2 = GBN_invariant(512*block.expansion, eps=self.eps)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.conv1 = conv3x3(3,64)
        # self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.normal_(m.bias, 0, 0.01)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_GBN):
                    nn.init.constant_(m.bn3.bn.weight, 0)
                elif isinstance(m, BasicBlock_GBN):
                    nn.init.constant_(m.bn2.bn.weight, 0)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, eps=self.eps))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
           
    @autocast()
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        y = self.linear(out)
        return y


class ResNet_GBN_inv_stl(nn.Module):
    def __init__(self, block, num_blocks, std_weight=1., num_classes=10, zero_init_residual=False, amp=True, eps=1e-05):
        super(ResNet_GBN_inv_stl, self).__init__()
        self.amp = amp
        self.eps = eps
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = GBN(64, eps=self.eps)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.bn2 = GBN_invariant(512, eps=self.eps)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_js.kaiming_normal_(m.weight, std_weight=std_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    init_js.constant_(m.weight, 1)
                    init_js.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     init_js.normal_(m.weight, 0, 0.01)
            #     init_js.normal_(m.bias, 0, 0.01)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_GBN):
                    nn.init.constant_(m.bn3.bn.weight, 0)
                elif isinstance(m, BasicBlock_GBN):
                    nn.init.constant_(m.bn2.bn.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, eps=self.eps))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.amp:
            with autocast():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = self.bn2(out)
                # out = F.avg_pool2d(out, 4)
                out = F.adaptive_avg_pool2d(out, (1,1))
                out = out.view(out.size(0), -1)
                y = self.linear(out)
                return y
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.bn2(out)
            # out = F.avg_pool2d(out, 4)
            out = F.adaptive_avg_pool2d(out, (1,1))
            out = out.view(out.size(0), -1)
            y = self.linear(out)
            return y

        
def resnet18(num_classes, std_weight=1., zero_init_residual=False, amp=True, eps=1e-05, momentum=0.1, **kwargs):
    return ResNet(BasicBlock, [2,2,2,2], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual, \
                                amp=amp, eps=eps, momentum=momentum)    # momentum is not implemented

def resnet34(num_classes):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def resnet50(num_classes, std_weight=1., zero_init_residual=False):
    return ResNet_imagenet(Bottleneck, [3,4,6,3], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual)

def resnet101(num_classes):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def resnet152(num_classes):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)


def resnet18_GBN(num_classes, std_weight=1., zero_init_residual=False):
    return ResNet_GBN(BasicBlock_GBN, [2,2,2,2], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual)

def resnet18_GBN_invariant(num_classes, std_weight=1., zero_init_residual=False):
    return ResNet_GBN_invariant(BasicBlock_GBN, [2,2,2,2], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual)

def resnet18_GBN_invariant2(num_classes, std_weight=1., zero_init_residual=False, amp=True, eps=1e-05, width=1.0, **kwargs):
    return ResNet_GBN_invariant2(BasicBlock_GBN, [2,2,2,2], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual, \
                                amp=amp, eps=eps, width=width, **kwargs)

def resnet18_LN_invariant(num_classes, std_weight=1., zero_init_residual=False, amp=True, eps=1e-05, width=1.0, **kwargs):
    return ResNet_LN_invariant(BasicBlock_LN, [2,2,2,2], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual, \
                                amp=amp, eps=eps, width=width, **kwargs)

def resnet18_GBN_inv_stl(num_classes, std_weight=1., zero_init_residual=False, amp=True, eps=1e-05):
    return ResNet_GBN_inv_stl(BasicBlock_GBN, [2,2,2,2], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual, \
                                amp=amp, eps=eps)
                                
def resnet18_GBN_invariant2_avg(num_classes, std_weight=1., zero_init_residual=False, amp=True, eps=1e-05):
    return ResNet_GBN_invariant2_avg(BasicBlock_GBN, [2,2,2,2], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual, \
                                amp=amp, eps=eps)
    
def resnet18_invariant(num_classes, std_weight=1., zero_init_residual=False, amp=True, eps=1e-05, momentum=0.1):
    return ResNet_invariant(BasicBlock, [2,2,2,2], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual, \
                                amp=amp, eps=eps, momentum=momentum)

def resnet18_invariant_avg(num_classes, std_weight=1., zero_init_residual=False, amp=True, eps=1e-05, momentum=0.1):
    return ResNet_invariant_avg(BasicBlock, [2,2,2,2], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual, \
                                amp=amp, eps=eps, momentum=momentum)

def resnet50_invariant(num_classes, std_weight=1., zero_init_residual=False, amp=True, eps=1e-05):
    return ResNet_invariant(Bottleneck, [3,4,6,3], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual, \
                                amp=amp, eps=eps)

def resnet101_invariant(num_classes, std_weight=1., zero_init_residual=False, amp=True, eps=1e-05):
    return ResNet_invariant(Bottleneck, [3,4,23,3], std_weight=std_weight, num_classes=num_classes, zero_init_residual=zero_init_residual, \
                                amp=amp, eps=eps)
                                    
def resnet50_GBN(num_classes, std_weight=1., zero_init_residual=True, amp=True):
    return ResNet_GBN_imagenet(Bottleneck_GBN, [3,4,6,3], std_weight=std_weight, num_classes=num_classes, \
                            zero_init_residual=zero_init_residual, amp=amp)

def resnet50_GBN_invariant(num_classes, std_weight=1., zero_init_residual=True, amp=True):
    return ResNet_GBN_imagenet_invariant(Bottleneck_GBN, [3,4,6,3], std_weight=std_weight, num_classes=num_classes, \
                            zero_init_residual=zero_init_residual, amp=amp)

def resnet50_GBN_invariant2(num_classes, std_weight=1., zero_init_residual=True, amp=True, eps=1e-05):
    return ResNet_GBN_imagenet_invariant2(Bottleneck_GBN, [3,4,6,3], std_weight=std_weight, num_classes=num_classes, \
                            zero_init_residual=zero_init_residual, amp=amp, eps=eps)

# def test():
#     net = ResNet18()
#     y = net(Variable(torch.randn(1,3,32,32)))
#     print(y.size())

# test()