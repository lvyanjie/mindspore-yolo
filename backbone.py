#coding:utf-8

# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

# Focus结构修改
from mindspore import nn
from mindspore import ops

from common import autopad


class SiLU(nn.Cell):
    def __init__(self):
        super(SiLU, self).__init__()
        
        self.sigmoid = ops.Sigmoid()

    def construct(self, x):
        return x * self.sigmoid(x)


class  Conv(nn.Cell):
    def __init__(self, c1, c2, k=1, s=1, p=None,
                momentum=0.97,
                eps=1e-3,
                act=True):
        super(Conv, self).__init__()
        self.padding = autopad(k, p)
        self.pad_mode = None
        if self.padding == 0:
            self.pad_mode = 'same'
        elif self.padding == 1:
            self.pad_mode = 'pad'
        self.conv = nn.Conv2d(in_channels=c1, out_channels=c2,
                            kernel_size=k, stride=s,
                            padding=self.padding,
                            pad_mode=self.pad_mode,
                            has_bias=False)
        self.bn = nn.BatchNorm2d(num_features=c2,
                                momentum=momentum,
                                eps=eps)
        # 网上材料都是leakyrelu激活方式
        self.act = SiLU() if act is True else (
            act if isinstance(act, nn.Cell) else ops.Identity()
        )

    def construct(self, x):
        return self.act(self.bn(self.conv(x)))
        


class Focus(nn.Cell):
    def __init__(self, c1, c2, k=1, s=1, p=None, act=True):
        # batch
        super(Focus, self).__init__()
        self.slice_op = ops.StridedSlice()
        self.concat_op = ops.Concat(axis=1)  # channel
        self.conv = Conv(c1 * 4, c2, k, s, p, act=act)

    def construct(self, x):
        b, c, h, w = x.shape   # 求取shape

        # b, c, h/2, w/2
        # split 4
        x1 = self.slice_op(x, (0, 0, 0, 0), (b, c, h, w), (1, 1, 2, 2))
        x2 = self.slice_op(x, (0, 0, 0, 1), (b, c, h, w), (1, 1, 2, 2))
        x3 = self.slice_op(x, (0, 0, 1, 0), (b, c, h, w), (1, 1, 2, 2))
        x4 = self.slice_op(x, (0, 0, 1, 1), (b, c, h, w), (1, 1, 2, 2))
        
        # 特征叠加
        x = self.concat_op((x1, x2, x3, x4))
        # 执行conv操作
        x = self.conv(x)
        return x


class Bottleneck(nn.Cell):
    # Standard bottlenect
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)   # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_, c2, 3, 1)
        self.add = shortcut and c1==c2

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2(c2)
        out = c2
        if self.add:
            out = x + out

        return out


class BottoleneckCSP(nn.Cell):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super(BottoleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.m = nn.SequentialCell([Bottleneck(c_, c_, shortcut, e) for _ in range(n)])
        self.conv2 = Conv(c1, c_, 1, 1)
        self.concat = ops.Concat(axis=1)
        self.conv3 = Conv(2*c_, c2, 1)
    

    def construct(self, x):
        c1 = self.conv1(x)
        c2 = self.m(c1)
        c3 = self.conv2(x)
        c4 = self.concat((c2, c3))
        c5 = self.conv3(c4)
        return c5


class SPP(nn.Cell):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=[5, 9, 13]):
        super(SPP, self).__init__()
        c_ = c1 // 2     # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=k[0], stride=1, pad_mode='same')
        self.maxpool2 = nn.MaxPool2d(kernel_size=k[1], stride=1, pad_mode='same')
        self.maxpool3 = nn.MaxPool2d(kernel_size=k[2], stride=1, pad_mode='same')
        self.concat = ops.Concat(axis=1)

        self.conv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)

    
    def construct(self, x):
        c1 = self.conv1(x)
        m1 = self.maxpool1(c1)
        m2 = self.maxpool2(c1)
        m3 = self.maxpool3(c1)
        c4 = self.concat((c1, m1, m2, m3))
        c5 = self.conv2(c4)
        return c5


class YOLOv5Backbone(nn.Cell):
    def __init__(self, shape):
        super(YOLOv5Backbone, self).__init__()

        # 用于确认输出维度
        c1, c2, c3, c4, c5, c6 = shape[0], shape[1], shape[2], \
                                 shape[3], shape[4], shape[5]
        short_num = shape[6]

        self.focus = Focus(c1, c2, k=3, s=1)
        self.conv1 = Conv(c2, c3, k=3, s=2)
        self.CSP1 = BottoleneckCSP(c3, c3, n=1 * short_num)
        self.conv2 = Conv(c3, c4, k=3, s=2)
        self.CSP2 = BottoleneckCSP(c4, c4, n=3 * short_num)
        self.conv3 = Conv(c4, c5, k=3, s=2)
        self.CSP3 = BottoleneckCSP(c5, c5, n=3 * short_num)
        self.conv4 = Conv(c5, c6, k=3, s=2)
        self.spp = SPP(c6, c6, k=[5, 9, 13])
        self.CSP4 = BottoleneckCSP(c6, c6, n=1 * short_num, shortcut=False)
    

    def construct(self, x):
        """
        :param x: shape is [batch, channel, height, width]
        """
        c1 = self.focus(x)    # out: batch, c2, height//2, width//2
        c2 = self.conv1(c1)   # out: batch, c3, height//4, width//4
        c3 = self.CSP1(c2)    # out: batch, c3, height//4, width//4
        c4 = self.conv2(c3)   # out: batch, c4, height//8, width//8
        # out
        c5 = self.CSP2(c4)    # out: batch, c4, height//8, width//8
        c6 = self.conv3(c5)   # out: batch, c5, height//16, width//16
        # out
        c7 = self.CSP3(c6)    # out: batch, c5, height//16, width//16
        c8 = self.conv4(c7)   # out: batch, c6, height//32, width//32
        c9 = self.spp(c8)     # out: batch, c6, height//32, width//32  --- 包含了多维特征
        # out
        c10 = self.CSP4(c9)   # out: batch, c6, height//32, width//32

        # 
        return c5, c7, c10

