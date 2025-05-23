# Copyright 2020 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore.nn as nn
from mindspore.nn import Conv2d, ReLU
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore import context
import mindspore

time_stamp_init = False
time_stamp_first = 0
loadvgg = 1


class OpenPoseNet(nn.Cell):
    insize = 368

    def __init__(self, vggpath='', vgg_with_bn=False):
        super(OpenPoseNet, self).__init__()
        self.base = Base_model(vgg_with_bn=vgg_with_bn)
        self.stage_1 = Stage_1()
        self.stage_2 = Stage_x()
        self.stage_3 = Stage_x()
        self.stage_4 = Stage_x()
        self.stage_5 = Stage_x()
        self.stage_6 = Stage_x()
        self.shape = P.Shape()
        self.cat = P.Concat(axis=1)
        self.add_Cascade_OPs = []
        self.Cascade_OPs = None
        self.Basic_OPS = None
        if loadvgg and vggpath:
            param_dict = load_checkpoint(vggpath)
            param_dict_new = {}
            trans_name = 'base.vgg_base.'
            for key, values in param_dict.items():
                if key.startswith('moments.'):
                    continue
                elif key.startswith('network.'):
                    param_dict_new[trans_name + key[17:]] = values
            load_param_into_net(self.base.vgg_base, param_dict_new)


    def construct(self, x):
        heatmaps = []
        pafs = []
        feature_map = self.base(x)
        h1, h2 = self.stage_1(feature_map)
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_2(self.cat((h1, h2, feature_map)))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_3(self.cat((h1, h2, feature_map)))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_4(self.cat((h1, h2, feature_map)))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_5(self.cat((h1, h2, feature_map)))
        pafs.append(h1)
        heatmaps.append(h2)
        h1, h2 = self.stage_6(self.cat((h1, h2, feature_map)))
        pafs.append(h1)
        heatmaps.append(h2)
        return pafs, heatmaps


class Vgg(nn.Cell):
    def __init__(self, cfg, batch_norm=False):
        # Important: When choose vgg, batch_size should <=64, otherwise will cause unknown error
        super(Vgg, self).__init__()
        # self.layers = self._make_layer(cfg, batch_norm=batch_norm)
        self.conv1 = Conv2d(in_channels=3,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')
        self.conv3 = Conv2d(in_channels=64,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
        self.relu3 = nn.ReLU()
        self.conv4 = Conv2d(in_channels=128,
                                out_channels=128,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')
        self.conv5 = Conv2d(in_channels=128,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
        self.relu5 = nn.ReLU()
        self.conv6 = Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
        self.relu6 = nn.ReLU()
        self.conv7 = Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
        self.relu7 = nn.ReLU()
        self.conv8 = Conv2d(in_channels=256,
                                out_channels=256,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
        self.relu8 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')
        self.conv9 = Conv2d(in_channels=256,
                                out_channels=512,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
        self.relu9 = nn.ReLU()
        self.conv10 = Conv2d(in_channels=512,
                                out_channels=512,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
        self.relu10 = nn.ReLU()

    def construct(self, x):
        # x = self.layers(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.maxpool3(x)
        x = self.conv9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.relu10(x)
        return x

    def _make_layer(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='same')]
            else:
                conv2d = Conv2d(in_channels=in_channels,
                                out_channels=v,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=True)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        return nn.SequentialCell(layers)


class Base_model(nn.Cell):
    def __init__(self, vgg_with_bn=False):
        super(Base_model, self).__init__()
        cfgs_zh = {'19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512]}
        self.vgg_base = Vgg(cfgs_zh['19'], batch_norm=vgg_with_bn)

        self.conv4_3_CPM = Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, pad_mode='same',
                                  has_bias=True)
        self.conv4_4_CPM = Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, pad_mode='same',
                                  has_bias=True)
        self.relu = ReLU()

    def construct(self, x):
        x = self.vgg_base(x)
        x = self.relu(self.conv4_3_CPM(x))
        x = self.relu(self.conv4_4_CPM(x))
        return x


class Stage_1(nn.Cell):
    def __init__(self):
        super(Stage_1, self).__init__()

        self.conv1_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same',
                                   has_bias=True)
        self.conv2_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same',
                                   has_bias=True)
        self.conv3_CPM_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same',
                                   has_bias=True)
        self.conv4_CPM_L1 = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, pad_mode='same',
                                   has_bias=True)
        self.conv5_CPM_L1 = Conv2d(in_channels=512, out_channels=38, kernel_size=1, stride=1, pad_mode='same',
                                   has_bias=True)

        self.conv1_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same',
                                   has_bias=True)
        self.conv2_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same',
                                   has_bias=True)
        self.conv3_CPM_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, pad_mode='same',
                                   has_bias=True)
        self.conv4_CPM_L2 = Conv2d(in_channels=128, out_channels=512, kernel_size=1, stride=1, pad_mode='same',
                                   has_bias=True)
        self.conv5_CPM_L2 = Conv2d(in_channels=512, out_channels=19, kernel_size=1, stride=1, pad_mode='same',
                                   has_bias=True)

        self.relu = ReLU()

    def construct(self, x):
        h1 = self.relu(self.conv1_CPM_L1(x))  # branch1
        h1 = self.relu(self.conv2_CPM_L1(h1))
        h1 = self.relu(self.conv3_CPM_L1(h1))
        h1 = self.relu(self.conv4_CPM_L1(h1))
        h1 = self.conv5_CPM_L1(h1)

        h2 = self.relu(self.conv1_CPM_L2(x))  # branch2
        h2 = self.relu(self.conv2_CPM_L2(h2))
        h2 = self.relu(self.conv3_CPM_L2(h2))
        h2 = self.relu(self.conv4_CPM_L2(h2))
        h2 = self.conv5_CPM_L2(h2)
        return h1, h2


class Stage_x(nn.Cell):
    def __init__(self):
        super(Stage_x, self).__init__()
        self.conv1_L1 = Conv2d(in_channels=185, out_channels=128, kernel_size=7, stride=1, pad_mode='same',
                               has_bias=True)

        self.conv2_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv3_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv4_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv5_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv6_L1 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv7_L1 = Conv2d(in_channels=128, out_channels=38, kernel_size=1, stride=1, pad_mode='same',
                               has_bias=True)

        self.conv1_L2 = Conv2d(in_channels=185, out_channels=128, kernel_size=7, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv2_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv3_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv4_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv5_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv6_L2 = Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, pad_mode='same',
                               has_bias=True)
        self.conv7_L2 = Conv2d(in_channels=128, out_channels=19, kernel_size=1, stride=1, pad_mode='same',
                               has_bias=True)
        self.relu = ReLU()

    def construct(self, x):
        h1 = self.relu(self.conv1_L1(x))  # branch1
        h1 = self.relu(self.conv2_L1(h1))
        h1 = self.relu(self.conv3_L1(h1))
        h1 = self.relu(self.conv4_L1(h1))
        h1 = self.relu(self.conv5_L1(h1))
        h1 = self.relu(self.conv6_L1(h1))
        h1 = self.conv7_L1(h1)
        h2 = self.relu(self.conv1_L2(x))  # branch2
        h2 = self.relu(self.conv2_L2(h2))
        h2 = self.relu(self.conv3_L2(h2))
        h2 = self.relu(self.conv4_L2(h2))
        h2 = self.relu(self.conv5_L2(h2))
        h2 = self.relu(self.conv6_L2(h2))
        h2 = self.conv7_L2(h2)
        return h1, h2
