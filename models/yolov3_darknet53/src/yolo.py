# Copyright 2020 Huawei Technologies Co., Ltd
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
"""YOLOv3 based on DarkNet."""
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C

from models.yolov3_darknet53.src.darknet import DarkNet, ResidualBlock
from models.yolov3_darknet53.src.loss import XYLoss, WHLoss, ConfidenceLoss, ClassLoss
from models.yolov3_darknet53.model_utils.config import config as default_config

def _conv_bn_relu(in_channel,
                  out_channel,
                  ksize,
                  stride=1,
                  padding=0,
                  dilation=1,
                  alpha=0.1,
                  momentum=0.9,
                  eps=1e-5,
                  pad_mode="same"):
    """Get a conv2d batchnorm and relu layer"""
    return nn.SequentialCell(
        [nn.Conv2d(in_channel,
                   out_channel,
                   kernel_size=ksize,
                   stride=stride,
                   padding=padding,
                   dilation=dilation,
                   pad_mode=pad_mode),
         nn.BatchNorm2d(out_channel, momentum=momentum, eps=eps),
         nn.LeakyReLU(alpha)]
    )


class YoloBlock(nn.Cell):
    """
    YoloBlock for YOLOv3.

    Args:
        in_channels: Integer. Input channel.
        out_chls: Integer. Middle channel.
        out_channels: Integer. Output channel.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3).

    Examples:
        YoloBlock(1024, 512, 255)

    """
    def __init__(self, in_channels, out_chls, out_channels):
        super(YoloBlock, self).__init__()
        out_chls_2 = out_chls*2

        self.conv0 = _conv_bn_relu(in_channels, out_chls, ksize=1)
        self.conv1 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv2 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv3 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv4 = _conv_bn_relu(out_chls_2, out_chls, ksize=1)
        self.conv5 = _conv_bn_relu(out_chls, out_chls_2, ksize=3)

        self.conv6 = nn.Conv2d(out_chls_2, out_channels, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        c1 = self.conv0(x)
        c2 = self.conv1(c1)

        c3 = self.conv2(c2)
        c4 = self.conv3(c3)

        c5 = self.conv4(c4)
        c6 = self.conv5(c5)

        out = self.conv6(c6)
        return c5, out


class YOLOv3(nn.Cell):
    """
     YOLOv3 Network.

     Note:
         backbone = darknet53

     Args:
         backbone_shape: List. Darknet output channels shape.
         backbone: Cell. Backbone Network.
         out_channel: Integer. Output channel.

     Returns:
         Tensor, output tensor.

     Examples:
         YOLOv3(backbone_shape=[64, 128, 256, 512, 1024]
                backbone=darknet53(),
                out_channel=255)
     """
    def __init__(self, backbone_shape, backbone, out_channel):
        super(YOLOv3, self).__init__()
        self.out_channel = out_channel
        self.backbone = backbone
        self.backblock0 = YoloBlock(backbone_shape[-1], out_chls=backbone_shape[-2], out_channels=out_channel)

        self.conv1 = _conv_bn_relu(in_channel=backbone_shape[-2], out_channel=backbone_shape[-2]//2, ksize=1)
        self.backblock1 = YoloBlock(in_channels=backbone_shape[-2]+backbone_shape[-3],
                                    out_chls=backbone_shape[-3],
                                    out_channels=out_channel)

        self.conv2 = _conv_bn_relu(in_channel=backbone_shape[-3], out_channel=backbone_shape[-3]//2, ksize=1)
        self.backblock2 = YoloBlock(in_channels=backbone_shape[-3]+backbone_shape[-4],
                                    out_chls=backbone_shape[-4],
                                    out_channels=out_channel)
        self.concat = P.Concat(axis=1)

        self.Cascade_OPs = None
        self.Basic_OPS = None

        self.add_Cascade_OPs = []
        self.in_shapes = {}
        self.orders = {}
        self.out_shapes = {}
        self.layer_names = {}
        self.origin_layer_names = {}




    def construct(self, x):
        # input_shape of x is (batch_size, 3, h, w)
        # feature_map1 is (batch_size, backbone_shape[2], h/8, w/8)
        # feature_map2 is (batch_size, backbone_shape[3], h/16, w/16)
        # feature_map3 is (batch_size, backbone_shape[4], h/32, w/32)
        img_hight = P.Shape()(x)[2]
        img_width = P.Shape()(x)[3]
        feature_map1, feature_map2, feature_map3 = self.backbone(x)
        con1, big_object_output = self.backblock0(feature_map3)

        con1 = self.conv1(con1)
        ups1 = P.ResizeNearestNeighbor((img_hight // 16, img_width // 16))(con1)
        con1 = self.concat((ups1, feature_map2))
        con2, medium_object_output = self.backblock1(con1)

        con2 = self.conv2(con2)
        ups2 = P.ResizeNearestNeighbor((img_hight // 8, img_width // 8))(con2)
        con3 = self.concat((ups2, feature_map1))
        _, small_object_output = self.backblock2(con3)

        return big_object_output, medium_object_output, small_object_output


class DetectionBlock(nn.Cell):
    """
     YOLOv3 detection Network. It will finally output the detection result.

     Args:
         scale: Character.
         config: Configuration.
         is_training: Bool, Whether train or not, default True.

     Returns:
         Tuple, tuple of output tensor,(f1,f2,f3).

     Examples:
         DetectionBlock(scale='l',stride=32,config=config)
     """

    def __init__(self, scale, config=None, is_training=True):
        super(DetectionBlock, self).__init__()
        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.num_anchors_per_scale = 3
        self.num_attrib = 4+1+self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = P.Reshape()
        self.tile = P.Tile()
        self.concat = P.Concat(axis=-1)
        self.conf_training = is_training

    def construct(self, x, input_shape):
        num_batch = P.Shape()(x)[0]
        grid_size = P.Shape()(x)[2:4]
        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = P.Reshape()(x, (num_batch,self.num_anchors_per_scale,self.num_attrib,grid_size[0],grid_size[1]))
        prediction = P.Transpose()(prediction, (0, 3, 4, 1, 2))
        range_x = range(grid_size[1])
        range_y = range(grid_size[0])
        grid_x = P.Cast()(F.tuple_to_array(range_x), ms.float32)
        grid_y = P.Cast()(F.tuple_to_array(range_y), ms.float32)
        # Tensor of shape [grid_size[0], grid_size[1], 1, 1] representing the coordinate of x/y axis for each grid
        # [batch, gridx, gridy, 1, 1]
        grid_x = self.tile(self.reshape(grid_x, (1, 1, -1, 1, 1)), (1, grid_size[0], 1, 1, 1))
        grid_y = self.tile(self.reshape(grid_y, (1, -1, 1, 1, 1)), (1, 1, grid_size[1], 1, 1))
        # Shape is [grid_size[0], grid_size[1], 1, 2]
        grid = self.concat((grid_x, grid_y))
        box_xy = prediction[:, :, :, :, :2]
        box_wh = prediction[:, :, :, :, 2:4]
        box_confidence = prediction[:, :, :, :, 4:5]
        box_probs = prediction[:, :, :, :, 5:]
        # gridsize1 is x
        # gridsize0 is y
        box_xy = (self.sigmoid(box_xy) + grid) / P.Cast()(F.tuple_to_array((grid_size[1], grid_size[0])), ms.float32)
        # box_wh is w->h
        box_wh = P.Exp()(box_wh) * self.anchors / input_shape
        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)
        if self.conf_training:
            return grid, prediction, box_xy, box_wh
        return self.concat((box_xy, box_wh, box_confidence, box_probs))


class Iou(nn.Cell):
    """Calculate the iou of boxes"""
    def __init__(self):
        super(Iou, self).__init__()
        self.min = P.Minimum()
        self.max = P.Maximum()

    def construct(self, box1, box2):
        # box1: pred_box [batch, gx, gy, anchors, 1,      4] ->4: [x_center, y_center, w, h]
        # box2: gt_box   [batch, 1,  1,  1,       maxbox, 4]
        # convert to topLeft and rightDown
        box1_xy = box1[:, :, :, :, :, :2]
        box1_wh = box1[:, :, :, :, :, 2:4]
        box1_mins = box1_xy - box1_wh / F.scalar_to_array(2.0) # topLeft
        box1_maxs = box1_xy + box1_wh / F.scalar_to_array(2.0) # rightDown

        box2_xy = box2[:, :, :, :, :, :2]
        box2_wh = box2[:, :, :, :, :, 2:4]
        box2_mins = box2_xy - box2_wh / F.scalar_to_array(2.0)
        box2_maxs = box2_xy + box2_wh / F.scalar_to_array(2.0)

        intersect_mins = self.max(box1_mins, box2_mins)
        intersect_maxs = self.min(box1_maxs, box2_maxs)
        intersect_wh = self.max(intersect_maxs - intersect_mins, F.scalar_to_array(0.0))
        # P.squeeze: for effiecient slice
        intersect_area = P.Squeeze(-1)(intersect_wh[:, :, :, :, :, 0:1]) * \
                         P.Squeeze(-1)(intersect_wh[:, :, :, :, :, 1:2])
        box1_area = P.Squeeze(-1)(box1_wh[:, :, :, :, :, 0:1]) * P.Squeeze(-1)(box1_wh[:, :, :, :, :, 1:2])
        box2_area = P.Squeeze(-1)(box2_wh[:, :, :, :, :, 0:1]) * P.Squeeze(-1)(box2_wh[:, :, :, :, :, 1:2])
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        # iou : [batch, gx, gy, anchors, maxboxes]
        return iou


class YoloLossBlock(nn.Cell):
    """
    Loss block cell of YOLOV3 network.
    """
    def __init__(self, scale, config=None):
        super(YoloLossBlock, self).__init__()
        self.config = config
        if scale == 's':
            # anchor mask
            idx = (0, 1, 2)
        elif scale == 'm':
            idx = (3, 4, 5)
        elif scale == 'l':
            idx = (6, 7, 8)
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        self.anchors = Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.ignore_threshold = Tensor(self.config.ignore_threshold, ms.float32)
        self.concat = P.Concat(axis=-1)
        self.iou = Iou()
        self.reduce_max = P.ReduceMax(keep_dims=False)
        self.xy_loss = XYLoss()
        self.wh_loss = WHLoss()
        self.confidenceLoss = ConfidenceLoss()
        self.classLoss = ClassLoss()

    def construct(self, grid, prediction, pred_xy, pred_wh, y_true, gt_box, input_shape):
        # prediction : origin output from yolo
        # pred_xy: (sigmoid(xy)+grid)/grid_size
        # pred_wh: (exp(wh)*anchors)/input_shape
        # y_true : after normalize
        # gt_box: [batch, maxboxes, xyhw] after normalize

        object_mask = y_true[:, :, :, :, 4:5]
        class_probs = y_true[:, :, :, :, 5:]

        grid_shape = P.Shape()(prediction)[1:3]
        grid_shape = P.Cast()(F.tuple_to_array(grid_shape[::-1]), ms.float32)

        pred_boxes = self.concat((pred_xy, pred_wh))
        true_xy = y_true[:, :, :, :, :2] * grid_shape - grid
        true_wh = y_true[:, :, :, :, 2:4]
        true_wh = P.Select()(P.Equal()(true_wh, 0.0),
                             P.Fill()(P.DType()(true_wh),
                                      P.Shape()(true_wh), 1.0),
                             true_wh)
        true_wh = P.Log()(true_wh / self.anchors * input_shape)
        # 2-w*h for large picture, use small scale, since small obj need more precise
        box_loss_scale = 2 - y_true[:, :, :, :, 2:3] * y_true[:, :, :, :, 3:4]

        gt_shape = P.Shape()(gt_box)
        gt_box = P.Reshape()(gt_box, (gt_shape[0], 1, 1, 1, gt_shape[1], gt_shape[2]))

        # add one more dimension for broadcast
        iou = self.iou(P.ExpandDims()(pred_boxes, -2), gt_box)
        # gt_box is x,y,h,w after normalize
        # [batch, grid[0], grid[1], num_anchor, num_gt]
        best_iou = self.reduce_max(iou, -1)
        # [batch, grid[0], grid[1], num_anchor]

        # ignore_mask IOU too small
        ignore_mask = best_iou < self.ignore_threshold
        ignore_mask = P.Cast()(ignore_mask, ms.float32)
        ignore_mask = P.ExpandDims()(ignore_mask, -1)
        # ignore_mask backpro will cause a lot maximunGrad and minimumGrad time consume.
        # so we turn off its gradient
        ignore_mask = F.stop_gradient(ignore_mask)

        xy_loss = self.xy_loss(object_mask, box_loss_scale, prediction[:, :, :, :, :2], true_xy)
        wh_loss = self.wh_loss(object_mask, box_loss_scale, prediction[:, :, :, :, 2:4], true_wh)
        confidence_loss = self.confidenceLoss(object_mask, prediction[:, :, :, :, 4:5], ignore_mask)
        class_loss = self.classLoss(object_mask, prediction[:, :, :, :, 5:], class_probs)
        loss = xy_loss + wh_loss + confidence_loss + class_loss
        batch_size = P.Shape()(prediction)[0]
        return loss / batch_size


class YOLOV3DarkNet53(nn.Cell):
    """
    Darknet based YOLOV3 network.

    Args:
        is_training: Bool. Whether train or not.

    Returns:
        Cell, cell instance of Darknet based YOLOV3 neural network.

    Examples:
        YOLOV3DarkNet53(True)
    """

    def __init__(self, is_training, config=default_config):
        super(YOLOV3DarkNet53, self).__init__()
        self.config = config
        self.keep_detect = self.config.keep_detect
        self.tenser_to_array = P.TupleToArray()

        # YOLOv3 network
        self.feature_map = YOLOv3(backbone=DarkNet(ResidualBlock, self.config.backbone_layers,
                                                   self.config.backbone_input_shape,
                                                   self.config.backbone_shape,
                                                   detect=True),
                                  backbone_shape=self.config.backbone_shape,
                                  out_channel=self.config.out_channel)

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock('l', is_training=is_training, config=self.config)
        self.detect_2 = DetectionBlock('m', is_training=is_training, config=self.config)
        self.detect_3 = DetectionBlock('s', is_training=is_training, config=self.config)

        self.add_Cascade_OPs = []
        self.Cascade_OPs = None
        self.Basic_OPS = None

        self.origin_layer_names = {
            "feature_map": self.feature_map,
            "feature_map.backbone": self.feature_map.backbone,
            "feature_map.backbone.conv0": self.feature_map.backbone.conv0,
            "feature_map.backbone.conv0.0": self.feature_map.backbone.conv0[0],
            "feature_map.backbone.conv0.1": self.feature_map.backbone.conv0[1],
            "feature_map.backbone.conv0.2": self.feature_map.backbone.conv0[2],
            "feature_map.backbone.conv1": self.feature_map.backbone.conv1,
            "feature_map.backbone.conv1.0": self.feature_map.backbone.conv1[0],
            "feature_map.backbone.conv1.1": self.feature_map.backbone.conv1[1],
            "feature_map.backbone.conv1.2": self.feature_map.backbone.conv1[2],
            "feature_map.backbone.layer1": self.feature_map.backbone.layer1,
            "feature_map.backbone.layer1.0": self.feature_map.backbone.layer1[0],
            "feature_map.backbone.layer1.0.conv1": self.feature_map.backbone.layer1[0].conv1,
            "feature_map.backbone.layer1.0.conv1.0": self.feature_map.backbone.layer1[0].conv1[0],
            "feature_map.backbone.layer1.0.conv1.1": self.feature_map.backbone.layer1[0].conv1[1],
            "feature_map.backbone.layer1.0.conv1.2": self.feature_map.backbone.layer1[0].conv1[2],
            "feature_map.backbone.layer1.0.conv2": self.feature_map.backbone.layer1[0].conv2,
            "feature_map.backbone.layer1.0.conv2.0": self.feature_map.backbone.layer1[0].conv2[0],
            "feature_map.backbone.layer1.0.conv2.1": self.feature_map.backbone.layer1[0].conv2[1],
            "feature_map.backbone.layer1.0.conv2.2": self.feature_map.backbone.layer1[0].conv2[2],
            "feature_map.backbone.conv2": self.feature_map.backbone.conv2,
            "feature_map.backbone.conv2.0": self.feature_map.backbone.conv2[0],
            "feature_map.backbone.conv2.1": self.feature_map.backbone.conv2[1],
            "feature_map.backbone.conv2.2": self.feature_map.backbone.conv2[2],
            "feature_map.backbone.layer2": self.feature_map.backbone.layer2,
            "feature_map.backbone.layer2.0": self.feature_map.backbone.layer2[0],
            "feature_map.backbone.layer2.0.conv1": self.feature_map.backbone.layer2[0].conv1,
            "feature_map.backbone.layer2.0.conv1.0": self.feature_map.backbone.layer2[0].conv1[0],
            "feature_map.backbone.layer2.0.conv1.1": self.feature_map.backbone.layer2[0].conv1[1],
            "feature_map.backbone.layer2.0.conv1.2": self.feature_map.backbone.layer2[0].conv1[2],
            "feature_map.backbone.layer2.0.conv2": self.feature_map.backbone.layer2[0].conv2,
            "feature_map.backbone.layer2.0.conv2.0": self.feature_map.backbone.layer2[0].conv2[0],
            "feature_map.backbone.layer2.0.conv2.1": self.feature_map.backbone.layer2[0].conv2[1],
            "feature_map.backbone.layer2.0.conv2.2": self.feature_map.backbone.layer2[0].conv2[2],
            "feature_map.backbone.layer2.1": self.feature_map.backbone.layer2[1],
            "feature_map.backbone.layer2.1.conv1": self.feature_map.backbone.layer2[1].conv1,
            "feature_map.backbone.layer2.1.conv1.0": self.feature_map.backbone.layer2[1].conv1[0],
            "feature_map.backbone.layer2.1.conv1.1": self.feature_map.backbone.layer2[1].conv1[1],
            "feature_map.backbone.layer2.1.conv1.2": self.feature_map.backbone.layer2[1].conv1[2],
            "feature_map.backbone.layer2.1.conv2": self.feature_map.backbone.layer2[1].conv2,
            "feature_map.backbone.layer2.1.conv2.0": self.feature_map.backbone.layer2[1].conv2[0],
            "feature_map.backbone.layer2.1.conv2.1": self.feature_map.backbone.layer2[1].conv2[1],
            "feature_map.backbone.layer2.1.conv2.2": self.feature_map.backbone.layer2[1].conv2[2],
            "feature_map.backbone.conv3": self.feature_map.backbone.conv3,
            "feature_map.backbone.conv3.0": self.feature_map.backbone.conv3[0],
            "feature_map.backbone.conv3.1": self.feature_map.backbone.conv3[1],
            "feature_map.backbone.conv3.2": self.feature_map.backbone.conv3[2],
            "feature_map.backbone.layer3": self.feature_map.backbone.layer3,
            "feature_map.backbone.layer3.0": self.feature_map.backbone.layer3[0],
            "feature_map.backbone.layer3.0.conv1": self.feature_map.backbone.layer3[0].conv1,
            "feature_map.backbone.layer3.0.conv1.0": self.feature_map.backbone.layer3[0].conv1[0],
            "feature_map.backbone.layer3.0.conv1.1": self.feature_map.backbone.layer3[0].conv1[1],
            "feature_map.backbone.layer3.0.conv1.2": self.feature_map.backbone.layer3[0].conv1[2],
            "feature_map.backbone.layer3.0.conv2": self.feature_map.backbone.layer3[0].conv2,
            "feature_map.backbone.layer3.0.conv2.0": self.feature_map.backbone.layer3[0].conv2[0],
            "feature_map.backbone.layer3.0.conv2.1": self.feature_map.backbone.layer3[0].conv2[1],
            "feature_map.backbone.layer3.0.conv2.2": self.feature_map.backbone.layer3[0].conv2[2],
            "feature_map.backbone.layer3.1": self.feature_map.backbone.layer3[1],
            "feature_map.backbone.layer3.1.conv1": self.feature_map.backbone.layer3[1].conv1,
            "feature_map.backbone.layer3.1.conv1.0": self.feature_map.backbone.layer3[1].conv1[0],
            "feature_map.backbone.layer3.1.conv1.1": self.feature_map.backbone.layer3[1].conv1[1],
            "feature_map.backbone.layer3.1.conv1.2": self.feature_map.backbone.layer3[1].conv1[2],
            "feature_map.backbone.layer3.1.conv2": self.feature_map.backbone.layer3[1].conv2,
            "feature_map.backbone.layer3.1.conv2.0": self.feature_map.backbone.layer3[1].conv2[0],
            "feature_map.backbone.layer3.1.conv2.1": self.feature_map.backbone.layer3[1].conv2[1],
            "feature_map.backbone.layer3.1.conv2.2": self.feature_map.backbone.layer3[1].conv2[2],
            "feature_map.backbone.layer3.2": self.feature_map.backbone.layer3[2],
            "feature_map.backbone.layer3.2.conv1": self.feature_map.backbone.layer3[2].conv1,
            "feature_map.backbone.layer3.2.conv1.0": self.feature_map.backbone.layer3[2].conv1[0],
            "feature_map.backbone.layer3.2.conv1.1": self.feature_map.backbone.layer3[2].conv1[1],
            "feature_map.backbone.layer3.2.conv1.2": self.feature_map.backbone.layer3[2].conv1[2],
            "feature_map.backbone.layer3.2.conv2": self.feature_map.backbone.layer3[2].conv2,
            "feature_map.backbone.layer3.2.conv2.0": self.feature_map.backbone.layer3[2].conv2[0],
            "feature_map.backbone.layer3.2.conv2.1": self.feature_map.backbone.layer3[2].conv2[1],
            "feature_map.backbone.layer3.2.conv2.2": self.feature_map.backbone.layer3[2].conv2[2],
            "feature_map.backbone.layer3.3": self.feature_map.backbone.layer3[3],
            "feature_map.backbone.layer3.3.conv1": self.feature_map.backbone.layer3[3].conv1,
            "feature_map.backbone.layer3.3.conv1.0": self.feature_map.backbone.layer3[3].conv1[0],
            "feature_map.backbone.layer3.3.conv1.1": self.feature_map.backbone.layer3[3].conv1[1],
            "feature_map.backbone.layer3.3.conv1.2": self.feature_map.backbone.layer3[3].conv1[2],
            "feature_map.backbone.layer3.3.conv2": self.feature_map.backbone.layer3[3].conv2,
            "feature_map.backbone.layer3.3.conv2.0": self.feature_map.backbone.layer3[3].conv2[0],
            "feature_map.backbone.layer3.3.conv2.1": self.feature_map.backbone.layer3[3].conv2[1],
            "feature_map.backbone.layer3.3.conv2.2": self.feature_map.backbone.layer3[3].conv2[2],
            "feature_map.backbone.layer3.4": self.feature_map.backbone.layer3[4],
            "feature_map.backbone.layer3.4.conv1": self.feature_map.backbone.layer3[4].conv1,
            "feature_map.backbone.layer3.4.conv1.0": self.feature_map.backbone.layer3[4].conv1[0],
            "feature_map.backbone.layer3.4.conv1.1": self.feature_map.backbone.layer3[4].conv1[1],
            "feature_map.backbone.layer3.4.conv1.2": self.feature_map.backbone.layer3[4].conv1[2],
            "feature_map.backbone.layer3.4.conv2": self.feature_map.backbone.layer3[4].conv2,
            "feature_map.backbone.layer3.4.conv2.0": self.feature_map.backbone.layer3[4].conv2[0],
            "feature_map.backbone.layer3.4.conv2.1": self.feature_map.backbone.layer3[4].conv2[1],
            "feature_map.backbone.layer3.4.conv2.2": self.feature_map.backbone.layer3[4].conv2[2],
            "feature_map.backbone.layer3.5": self.feature_map.backbone.layer3[5],
            "feature_map.backbone.layer3.5.conv1": self.feature_map.backbone.layer3[5].conv1,
            "feature_map.backbone.layer3.5.conv1.0": self.feature_map.backbone.layer3[5].conv1[0],
            "feature_map.backbone.layer3.5.conv1.1": self.feature_map.backbone.layer3[5].conv1[1],
            "feature_map.backbone.layer3.5.conv1.2": self.feature_map.backbone.layer3[5].conv1[2],
            "feature_map.backbone.layer3.5.conv2": self.feature_map.backbone.layer3[5].conv2,
            "feature_map.backbone.layer3.5.conv2.0": self.feature_map.backbone.layer3[5].conv2[0],
            "feature_map.backbone.layer3.5.conv2.1": self.feature_map.backbone.layer3[5].conv2[1],
            "feature_map.backbone.layer3.5.conv2.2": self.feature_map.backbone.layer3[5].conv2[2],
            "feature_map.backbone.layer3.6": self.feature_map.backbone.layer3[6],
            "feature_map.backbone.layer3.6.conv1": self.feature_map.backbone.layer3[6].conv1,
            "feature_map.backbone.layer3.6.conv1.0": self.feature_map.backbone.layer3[6].conv1[0],
            "feature_map.backbone.layer3.6.conv1.1": self.feature_map.backbone.layer3[6].conv1[1],
            "feature_map.backbone.layer3.6.conv1.2": self.feature_map.backbone.layer3[6].conv1[2],
            "feature_map.backbone.layer3.6.conv2": self.feature_map.backbone.layer3[6].conv2,
            "feature_map.backbone.layer3.6.conv2.0": self.feature_map.backbone.layer3[6].conv2[0],
            "feature_map.backbone.layer3.6.conv2.1": self.feature_map.backbone.layer3[6].conv2[1],
            "feature_map.backbone.layer3.6.conv2.2": self.feature_map.backbone.layer3[6].conv2[2],
            "feature_map.backbone.layer3.7": self.feature_map.backbone.layer3[7],
            "feature_map.backbone.layer3.7.conv1": self.feature_map.backbone.layer3[7].conv1,
            "feature_map.backbone.layer3.7.conv1.0": self.feature_map.backbone.layer3[7].conv1[0],
            "feature_map.backbone.layer3.7.conv1.1": self.feature_map.backbone.layer3[7].conv1[1],
            "feature_map.backbone.layer3.7.conv1.2": self.feature_map.backbone.layer3[7].conv1[2],
            "feature_map.backbone.layer3.7.conv2": self.feature_map.backbone.layer3[7].conv2,
            "feature_map.backbone.layer3.7.conv2.0": self.feature_map.backbone.layer3[7].conv2[0],
            "feature_map.backbone.layer3.7.conv2.1": self.feature_map.backbone.layer3[7].conv2[1],
            "feature_map.backbone.layer3.7.conv2.2": self.feature_map.backbone.layer3[7].conv2[2],
            "feature_map.backbone.conv4": self.feature_map.backbone.conv4,
            "feature_map.backbone.conv4.0": self.feature_map.backbone.conv4[0],
            "feature_map.backbone.conv4.1": self.feature_map.backbone.conv4[1],
            "feature_map.backbone.conv4.2": self.feature_map.backbone.conv4[2],
            "feature_map.backbone.layer4": self.feature_map.backbone.layer4,
            "feature_map.backbone.layer4.0": self.feature_map.backbone.layer4[0],
            "feature_map.backbone.layer4.0.conv1": self.feature_map.backbone.layer4[0].conv1,
            "feature_map.backbone.layer4.0.conv1.0": self.feature_map.backbone.layer4[0].conv1[0],
            "feature_map.backbone.layer4.0.conv1.1": self.feature_map.backbone.layer4[0].conv1[1],
            "feature_map.backbone.layer4.0.conv1.2": self.feature_map.backbone.layer4[0].conv1[2],
            "feature_map.backbone.layer4.0.conv2": self.feature_map.backbone.layer4[0].conv2,
            "feature_map.backbone.layer4.0.conv2.0": self.feature_map.backbone.layer4[0].conv2[0],
            "feature_map.backbone.layer4.0.conv2.1": self.feature_map.backbone.layer4[0].conv2[1],
            "feature_map.backbone.layer4.0.conv2.2": self.feature_map.backbone.layer4[0].conv2[2],
            "feature_map.backbone.layer4.1": self.feature_map.backbone.layer4[1],
            "feature_map.backbone.layer4.1.conv1": self.feature_map.backbone.layer4[1].conv1,
            "feature_map.backbone.layer4.1.conv1.0": self.feature_map.backbone.layer4[1].conv1[0],
            "feature_map.backbone.layer4.1.conv1.1": self.feature_map.backbone.layer4[1].conv1[1],
            "feature_map.backbone.layer4.1.conv1.2": self.feature_map.backbone.layer4[1].conv1[2],
            "feature_map.backbone.layer4.1.conv2": self.feature_map.backbone.layer4[1].conv2,
            "feature_map.backbone.layer4.1.conv2.0": self.feature_map.backbone.layer4[1].conv2[0],
            "feature_map.backbone.layer4.1.conv2.1": self.feature_map.backbone.layer4[1].conv2[1],
            "feature_map.backbone.layer4.1.conv2.2": self.feature_map.backbone.layer4[1].conv2[2],
            "feature_map.backbone.layer4.2": self.feature_map.backbone.layer4[2],
            "feature_map.backbone.layer4.2.conv1": self.feature_map.backbone.layer4[2].conv1,
            "feature_map.backbone.layer4.2.conv1.0": self.feature_map.backbone.layer4[2].conv1[0],
            "feature_map.backbone.layer4.2.conv1.1": self.feature_map.backbone.layer4[2].conv1[1],
            "feature_map.backbone.layer4.2.conv1.2": self.feature_map.backbone.layer4[2].conv1[2],
            "feature_map.backbone.layer4.2.conv2": self.feature_map.backbone.layer4[2].conv2,
            "feature_map.backbone.layer4.2.conv2.0": self.feature_map.backbone.layer4[2].conv2[0],
            "feature_map.backbone.layer4.2.conv2.1": self.feature_map.backbone.layer4[2].conv2[1],
            "feature_map.backbone.layer4.2.conv2.2": self.feature_map.backbone.layer4[2].conv2[2],
            "feature_map.backbone.layer4.3": self.feature_map.backbone.layer4[3],
            "feature_map.backbone.layer4.3.conv1": self.feature_map.backbone.layer4[3].conv1,
            "feature_map.backbone.layer4.3.conv1.0": self.feature_map.backbone.layer4[3].conv1[0],
            "feature_map.backbone.layer4.3.conv1.1": self.feature_map.backbone.layer4[3].conv1[1],
            "feature_map.backbone.layer4.3.conv1.2": self.feature_map.backbone.layer4[3].conv1[2],
            "feature_map.backbone.layer4.3.conv2": self.feature_map.backbone.layer4[3].conv2,
            "feature_map.backbone.layer4.3.conv2.0": self.feature_map.backbone.layer4[3].conv2[0],
            "feature_map.backbone.layer4.3.conv2.1": self.feature_map.backbone.layer4[3].conv2[1],
            "feature_map.backbone.layer4.3.conv2.2": self.feature_map.backbone.layer4[3].conv2[2],
            "feature_map.backbone.layer4.4": self.feature_map.backbone.layer4[4],
            "feature_map.backbone.layer4.4.conv1": self.feature_map.backbone.layer4[4].conv1,
            "feature_map.backbone.layer4.4.conv1.0": self.feature_map.backbone.layer4[4].conv1[0],
            "feature_map.backbone.layer4.4.conv1.1": self.feature_map.backbone.layer4[4].conv1[1],
            "feature_map.backbone.layer4.4.conv1.2": self.feature_map.backbone.layer4[4].conv1[2],
            "feature_map.backbone.layer4.4.conv2": self.feature_map.backbone.layer4[4].conv2,
            "feature_map.backbone.layer4.4.conv2.0": self.feature_map.backbone.layer4[4].conv2[0],
            "feature_map.backbone.layer4.4.conv2.1": self.feature_map.backbone.layer4[4].conv2[1],
            "feature_map.backbone.layer4.4.conv2.2": self.feature_map.backbone.layer4[4].conv2[2],
            "feature_map.backbone.layer4.5": self.feature_map.backbone.layer4[5],
            "feature_map.backbone.layer4.5.conv1": self.feature_map.backbone.layer4[5].conv1,
            "feature_map.backbone.layer4.5.conv1.0": self.feature_map.backbone.layer4[5].conv1[0],
            "feature_map.backbone.layer4.5.conv1.1": self.feature_map.backbone.layer4[5].conv1[1],
            "feature_map.backbone.layer4.5.conv1.2": self.feature_map.backbone.layer4[5].conv1[2],
            "feature_map.backbone.layer4.5.conv2": self.feature_map.backbone.layer4[5].conv2,
            "feature_map.backbone.layer4.5.conv2.0": self.feature_map.backbone.layer4[5].conv2[0],
            "feature_map.backbone.layer4.5.conv2.1": self.feature_map.backbone.layer4[5].conv2[1],
            "feature_map.backbone.layer4.5.conv2.2": self.feature_map.backbone.layer4[5].conv2[2],
            "feature_map.backbone.layer4.6": self.feature_map.backbone.layer4[6],
            "feature_map.backbone.layer4.6.conv1": self.feature_map.backbone.layer4[6].conv1,
            "feature_map.backbone.layer4.6.conv1.0": self.feature_map.backbone.layer4[6].conv1[0],
            "feature_map.backbone.layer4.6.conv1.1": self.feature_map.backbone.layer4[6].conv1[1],
            "feature_map.backbone.layer4.6.conv1.2": self.feature_map.backbone.layer4[6].conv1[2],
            "feature_map.backbone.layer4.6.conv2": self.feature_map.backbone.layer4[6].conv2,
            "feature_map.backbone.layer4.6.conv2.0": self.feature_map.backbone.layer4[6].conv2[0],
            "feature_map.backbone.layer4.6.conv2.1": self.feature_map.backbone.layer4[6].conv2[1],
            "feature_map.backbone.layer4.6.conv2.2": self.feature_map.backbone.layer4[6].conv2[2],
            "feature_map.backbone.layer4.7": self.feature_map.backbone.layer4[7],
            "feature_map.backbone.layer4.7.conv1": self.feature_map.backbone.layer4[7].conv1,
            "feature_map.backbone.layer4.7.conv1.0": self.feature_map.backbone.layer4[7].conv1[0],
            "feature_map.backbone.layer4.7.conv1.1": self.feature_map.backbone.layer4[7].conv1[1],
            "feature_map.backbone.layer4.7.conv1.2": self.feature_map.backbone.layer4[7].conv1[2],
            "feature_map.backbone.layer4.7.conv2": self.feature_map.backbone.layer4[7].conv2,
            "feature_map.backbone.layer4.7.conv2.0": self.feature_map.backbone.layer4[7].conv2[0],
            "feature_map.backbone.layer4.7.conv2.1": self.feature_map.backbone.layer4[7].conv2[1],
            "feature_map.backbone.layer4.7.conv2.2": self.feature_map.backbone.layer4[7].conv2[2],
            "feature_map.backbone.conv5": self.feature_map.backbone.conv5,
            "feature_map.backbone.conv5.0": self.feature_map.backbone.conv5[0],
            "feature_map.backbone.conv5.1": self.feature_map.backbone.conv5[1],
            "feature_map.backbone.conv5.2": self.feature_map.backbone.conv5[2],
            "feature_map.backbone.layer5": self.feature_map.backbone.layer5,
            "feature_map.backbone.layer5.0": self.feature_map.backbone.layer5[0],
            "feature_map.backbone.layer5.0.conv1": self.feature_map.backbone.layer5[0].conv1,
            "feature_map.backbone.layer5.0.conv1.0": self.feature_map.backbone.layer5[0].conv1[0],
            "feature_map.backbone.layer5.0.conv1.1": self.feature_map.backbone.layer5[0].conv1[1],
            "feature_map.backbone.layer5.0.conv1.2": self.feature_map.backbone.layer5[0].conv1[2],
            "feature_map.backbone.layer5.0.conv2": self.feature_map.backbone.layer5[0].conv2,
            "feature_map.backbone.layer5.0.conv2.0": self.feature_map.backbone.layer5[0].conv2[0],
            "feature_map.backbone.layer5.0.conv2.1": self.feature_map.backbone.layer5[0].conv2[1],
            "feature_map.backbone.layer5.0.conv2.2": self.feature_map.backbone.layer5[0].conv2[2],
            "feature_map.backbone.layer5.1": self.feature_map.backbone.layer5[1],
            "feature_map.backbone.layer5.1.conv1": self.feature_map.backbone.layer5[1].conv1,
            "feature_map.backbone.layer5.1.conv1.0": self.feature_map.backbone.layer5[1].conv1[0],
            "feature_map.backbone.layer5.1.conv1.1": self.feature_map.backbone.layer5[1].conv1[1],
            "feature_map.backbone.layer5.1.conv1.2": self.feature_map.backbone.layer5[1].conv1[2],
            "feature_map.backbone.layer5.1.conv2": self.feature_map.backbone.layer5[1].conv2,
            "feature_map.backbone.layer5.1.conv2.0": self.feature_map.backbone.layer5[1].conv2[0],
            "feature_map.backbone.layer5.1.conv2.1": self.feature_map.backbone.layer5[1].conv2[1],
            "feature_map.backbone.layer5.1.conv2.2": self.feature_map.backbone.layer5[1].conv2[2],
            "feature_map.backbone.layer5.2": self.feature_map.backbone.layer5[2],
            "feature_map.backbone.layer5.2.conv1": self.feature_map.backbone.layer5[2].conv1,
            "feature_map.backbone.layer5.2.conv1.0": self.feature_map.backbone.layer5[2].conv1[0],
            "feature_map.backbone.layer5.2.conv1.1": self.feature_map.backbone.layer5[2].conv1[1],
            "feature_map.backbone.layer5.2.conv1.2": self.feature_map.backbone.layer5[2].conv1[2],
            "feature_map.backbone.layer5.2.conv2": self.feature_map.backbone.layer5[2].conv2,
            "feature_map.backbone.layer5.2.conv2.0": self.feature_map.backbone.layer5[2].conv2[0],
            "feature_map.backbone.layer5.2.conv2.1": self.feature_map.backbone.layer5[2].conv2[1],
            "feature_map.backbone.layer5.2.conv2.2": self.feature_map.backbone.layer5[2].conv2[2],
            "feature_map.backbone.layer5.3": self.feature_map.backbone.layer5[3],
            "feature_map.backbone.layer5.3.conv1": self.feature_map.backbone.layer5[3].conv1,
            "feature_map.backbone.layer5.3.conv1.0": self.feature_map.backbone.layer5[3].conv1[0],
            "feature_map.backbone.layer5.3.conv1.1": self.feature_map.backbone.layer5[3].conv1[1],
            "feature_map.backbone.layer5.3.conv1.2": self.feature_map.backbone.layer5[3].conv1[2],
            "feature_map.backbone.layer5.3.conv2": self.feature_map.backbone.layer5[3].conv2,
            "feature_map.backbone.layer5.3.conv2.0": self.feature_map.backbone.layer5[3].conv2[0],
            "feature_map.backbone.layer5.3.conv2.1": self.feature_map.backbone.layer5[3].conv2[1],
            "feature_map.backbone.layer5.3.conv2.2": self.feature_map.backbone.layer5[3].conv2[2],
            "feature_map.backblock0": self.feature_map.backblock0,
            "feature_map.backblock0.conv0": self.feature_map.backblock0.conv0,
            "feature_map.backblock0.conv0.0": self.feature_map.backblock0.conv0[0],
            "feature_map.backblock0.conv0.1": self.feature_map.backblock0.conv0[1],
            "feature_map.backblock0.conv0.2": self.feature_map.backblock0.conv0[2],
            "feature_map.backblock0.conv1": self.feature_map.backblock0.conv1,
            "feature_map.backblock0.conv1.0": self.feature_map.backblock0.conv1[0],
            "feature_map.backblock0.conv1.1": self.feature_map.backblock0.conv1[1],
            "feature_map.backblock0.conv1.2": self.feature_map.backblock0.conv1[2],
            "feature_map.backblock0.conv2": self.feature_map.backblock0.conv2,
            "feature_map.backblock0.conv2.0": self.feature_map.backblock0.conv2[0],
            "feature_map.backblock0.conv2.1": self.feature_map.backblock0.conv2[1],
            "feature_map.backblock0.conv2.2": self.feature_map.backblock0.conv2[2],
            "feature_map.backblock0.conv3": self.feature_map.backblock0.conv3,
            "feature_map.backblock0.conv3.0": self.feature_map.backblock0.conv3[0],
            "feature_map.backblock0.conv3.1": self.feature_map.backblock0.conv3[1],
            "feature_map.backblock0.conv3.2": self.feature_map.backblock0.conv3[2],
            "feature_map.backblock0.conv4": self.feature_map.backblock0.conv4,
            "feature_map.backblock0.conv4.0": self.feature_map.backblock0.conv4[0],
            "feature_map.backblock0.conv4.1": self.feature_map.backblock0.conv4[1],
            "feature_map.backblock0.conv4.2": self.feature_map.backblock0.conv4[2],
            "feature_map.backblock0.conv5": self.feature_map.backblock0.conv5,
            "feature_map.backblock0.conv5.0": self.feature_map.backblock0.conv5[0],
            "feature_map.backblock0.conv5.1": self.feature_map.backblock0.conv5[1],
            "feature_map.backblock0.conv5.2": self.feature_map.backblock0.conv5[2],
            "feature_map.backblock0.conv6": self.feature_map.backblock0.conv6,
            "feature_map.conv1": self.feature_map.conv1,
            "feature_map.conv1.0": self.feature_map.conv1[0],
            "feature_map.conv1.1": self.feature_map.conv1[1],
            "feature_map.conv1.2": self.feature_map.conv1[2],
            "feature_map.backblock1": self.feature_map.backblock1,
            "feature_map.backblock1.conv0": self.feature_map.backblock1.conv0,
            "feature_map.backblock1.conv0.0": self.feature_map.backblock1.conv0[0],
            "feature_map.backblock1.conv0.1": self.feature_map.backblock1.conv0[1],
            "feature_map.backblock1.conv0.2": self.feature_map.backblock1.conv0[2],
            "feature_map.backblock1.conv1": self.feature_map.backblock1.conv1,
            "feature_map.backblock1.conv1.0": self.feature_map.backblock1.conv1[0],
            "feature_map.backblock1.conv1.1": self.feature_map.backblock1.conv1[1],
            "feature_map.backblock1.conv1.2": self.feature_map.backblock1.conv1[2],
            "feature_map.backblock1.conv2": self.feature_map.backblock1.conv2,
            "feature_map.backblock1.conv2.0": self.feature_map.backblock1.conv2[0],
            "feature_map.backblock1.conv2.1": self.feature_map.backblock1.conv2[1],
            "feature_map.backblock1.conv2.2": self.feature_map.backblock1.conv2[2],
            "feature_map.backblock1.conv3": self.feature_map.backblock1.conv3,
            "feature_map.backblock1.conv3.0": self.feature_map.backblock1.conv3[0],
            "feature_map.backblock1.conv3.1": self.feature_map.backblock1.conv3[1],
            "feature_map.backblock1.conv3.2": self.feature_map.backblock1.conv3[2],
            "feature_map.backblock1.conv4": self.feature_map.backblock1.conv4,
            "feature_map.backblock1.conv4.0": self.feature_map.backblock1.conv4[0],
            "feature_map.backblock1.conv4.1": self.feature_map.backblock1.conv4[1],
            "feature_map.backblock1.conv4.2": self.feature_map.backblock1.conv4[2],
            "feature_map.backblock1.conv5": self.feature_map.backblock1.conv5,
            "feature_map.backblock1.conv5.0": self.feature_map.backblock1.conv5[0],
            "feature_map.backblock1.conv5.1": self.feature_map.backblock1.conv5[1],
            "feature_map.backblock1.conv5.2": self.feature_map.backblock1.conv5[2],
            "feature_map.backblock1.conv6": self.feature_map.backblock1.conv6,
            "feature_map.conv2": self.feature_map.conv2,
            "feature_map.conv2.0": self.feature_map.conv2[0],
            "feature_map.conv2.1": self.feature_map.conv2[1],
            "feature_map.conv2.2": self.feature_map.conv2[2],
            "feature_map.backblock2": self.feature_map.backblock2,
            "feature_map.backblock2.conv0": self.feature_map.backblock2.conv0,
            "feature_map.backblock2.conv0.0": self.feature_map.backblock2.conv0[0],
            "feature_map.backblock2.conv0.1": self.feature_map.backblock2.conv0[1],
            "feature_map.backblock2.conv0.2": self.feature_map.backblock2.conv0[2],
            "feature_map.backblock2.conv1": self.feature_map.backblock2.conv1,
            "feature_map.backblock2.conv1.0": self.feature_map.backblock2.conv1[0],
            "feature_map.backblock2.conv1.1": self.feature_map.backblock2.conv1[1],
            "feature_map.backblock2.conv1.2": self.feature_map.backblock2.conv1[2],
            "feature_map.backblock2.conv2": self.feature_map.backblock2.conv2,
            "feature_map.backblock2.conv2.0": self.feature_map.backblock2.conv2[0],
            "feature_map.backblock2.conv2.1": self.feature_map.backblock2.conv2[1],
            "feature_map.backblock2.conv2.2": self.feature_map.backblock2.conv2[2],
            "feature_map.backblock2.conv3": self.feature_map.backblock2.conv3,
            "feature_map.backblock2.conv3.0": self.feature_map.backblock2.conv3[0],
            "feature_map.backblock2.conv3.1": self.feature_map.backblock2.conv3[1],
            "feature_map.backblock2.conv3.2": self.feature_map.backblock2.conv3[2],
            "feature_map.backblock2.conv4": self.feature_map.backblock2.conv4,
            "feature_map.backblock2.conv4.0": self.feature_map.backblock2.conv4[0],
            "feature_map.backblock2.conv4.1": self.feature_map.backblock2.conv4[1],
            "feature_map.backblock2.conv4.2": self.feature_map.backblock2.conv4[2],
            "feature_map.backblock2.conv5": self.feature_map.backblock2.conv5,
            "feature_map.backblock2.conv5.0": self.feature_map.backblock2.conv5[0],
            "feature_map.backblock2.conv5.1": self.feature_map.backblock2.conv5[1],
            "feature_map.backblock2.conv5.2": self.feature_map.backblock2.conv5[2],
            "feature_map.backblock2.conv6": self.feature_map.backblock2.conv6,
            "detect_1": self.detect_1,
            "detect_1.sigmoid": self.detect_1.sigmoid,
            "detect_2": self.detect_2,
            "detect_2.sigmoid": self.detect_2.sigmoid,
            "detect_3": self.detect_3,
            "detect_3.sigmoid": self.detect_3.sigmoid,

        }
        self.layer_names = {
            "feature_map": self.feature_map,
            "feature_map.backbone": self.feature_map.backbone,
            "feature_map.backbone.conv0": self.feature_map.backbone.conv0,
            "feature_map.backbone.conv0.0": self.feature_map.backbone.conv0[0],
            "feature_map.backbone.conv0.1": self.feature_map.backbone.conv0[1],
            "feature_map.backbone.conv0.2": self.feature_map.backbone.conv0[2],
            "feature_map.backbone.conv1": self.feature_map.backbone.conv1,
            "feature_map.backbone.conv1.0": self.feature_map.backbone.conv1[0],
            "feature_map.backbone.conv1.1": self.feature_map.backbone.conv1[1],
            "feature_map.backbone.conv1.2": self.feature_map.backbone.conv1[2],
            "feature_map.backbone.layer1": self.feature_map.backbone.layer1,
            "feature_map.backbone.layer1.0": self.feature_map.backbone.layer1[0],
            "feature_map.backbone.layer1.0.conv1": self.feature_map.backbone.layer1[0].conv1,
            "feature_map.backbone.layer1.0.conv1.0": self.feature_map.backbone.layer1[0].conv1[0],
            "feature_map.backbone.layer1.0.conv1.1": self.feature_map.backbone.layer1[0].conv1[1],
            "feature_map.backbone.layer1.0.conv1.2": self.feature_map.backbone.layer1[0].conv1[2],
            "feature_map.backbone.layer1.0.conv2": self.feature_map.backbone.layer1[0].conv2,
            "feature_map.backbone.layer1.0.conv2.0": self.feature_map.backbone.layer1[0].conv2[0],
            "feature_map.backbone.layer1.0.conv2.1": self.feature_map.backbone.layer1[0].conv2[1],
            "feature_map.backbone.layer1.0.conv2.2": self.feature_map.backbone.layer1[0].conv2[2],
            "feature_map.backbone.conv2": self.feature_map.backbone.conv2,
            "feature_map.backbone.conv2.0": self.feature_map.backbone.conv2[0],
            "feature_map.backbone.conv2.1": self.feature_map.backbone.conv2[1],
            "feature_map.backbone.conv2.2": self.feature_map.backbone.conv2[2],
            "feature_map.backbone.layer2": self.feature_map.backbone.layer2,
            "feature_map.backbone.layer2.0": self.feature_map.backbone.layer2[0],
            "feature_map.backbone.layer2.0.conv1": self.feature_map.backbone.layer2[0].conv1,
            "feature_map.backbone.layer2.0.conv1.0": self.feature_map.backbone.layer2[0].conv1[0],
            "feature_map.backbone.layer2.0.conv1.1": self.feature_map.backbone.layer2[0].conv1[1],
            "feature_map.backbone.layer2.0.conv1.2": self.feature_map.backbone.layer2[0].conv1[2],
            "feature_map.backbone.layer2.0.conv2": self.feature_map.backbone.layer2[0].conv2,
            "feature_map.backbone.layer2.0.conv2.0": self.feature_map.backbone.layer2[0].conv2[0],
            "feature_map.backbone.layer2.0.conv2.1": self.feature_map.backbone.layer2[0].conv2[1],
            "feature_map.backbone.layer2.0.conv2.2": self.feature_map.backbone.layer2[0].conv2[2],
            "feature_map.backbone.layer2.1": self.feature_map.backbone.layer2[1],
            "feature_map.backbone.layer2.1.conv1": self.feature_map.backbone.layer2[1].conv1,
            "feature_map.backbone.layer2.1.conv1.0": self.feature_map.backbone.layer2[1].conv1[0],
            "feature_map.backbone.layer2.1.conv1.1": self.feature_map.backbone.layer2[1].conv1[1],
            "feature_map.backbone.layer2.1.conv1.2": self.feature_map.backbone.layer2[1].conv1[2],
            "feature_map.backbone.layer2.1.conv2": self.feature_map.backbone.layer2[1].conv2,
            "feature_map.backbone.layer2.1.conv2.0": self.feature_map.backbone.layer2[1].conv2[0],
            "feature_map.backbone.layer2.1.conv2.1": self.feature_map.backbone.layer2[1].conv2[1],
            "feature_map.backbone.layer2.1.conv2.2": self.feature_map.backbone.layer2[1].conv2[2],
            "feature_map.backbone.conv3": self.feature_map.backbone.conv3,
            "feature_map.backbone.conv3.0": self.feature_map.backbone.conv3[0],
            "feature_map.backbone.conv3.1": self.feature_map.backbone.conv3[1],
            "feature_map.backbone.conv3.2": self.feature_map.backbone.conv3[2],
            "feature_map.backbone.layer3": self.feature_map.backbone.layer3,
            "feature_map.backbone.layer3.0": self.feature_map.backbone.layer3[0],
            "feature_map.backbone.layer3.0.conv1": self.feature_map.backbone.layer3[0].conv1,
            "feature_map.backbone.layer3.0.conv1.0": self.feature_map.backbone.layer3[0].conv1[0],
            "feature_map.backbone.layer3.0.conv1.1": self.feature_map.backbone.layer3[0].conv1[1],
            "feature_map.backbone.layer3.0.conv1.2": self.feature_map.backbone.layer3[0].conv1[2],
            "feature_map.backbone.layer3.0.conv2": self.feature_map.backbone.layer3[0].conv2,
            "feature_map.backbone.layer3.0.conv2.0": self.feature_map.backbone.layer3[0].conv2[0],
            "feature_map.backbone.layer3.0.conv2.1": self.feature_map.backbone.layer3[0].conv2[1],
            "feature_map.backbone.layer3.0.conv2.2": self.feature_map.backbone.layer3[0].conv2[2],
            "feature_map.backbone.layer3.1": self.feature_map.backbone.layer3[1],
            "feature_map.backbone.layer3.1.conv1": self.feature_map.backbone.layer3[1].conv1,
            "feature_map.backbone.layer3.1.conv1.0": self.feature_map.backbone.layer3[1].conv1[0],
            "feature_map.backbone.layer3.1.conv1.1": self.feature_map.backbone.layer3[1].conv1[1],
            "feature_map.backbone.layer3.1.conv1.2": self.feature_map.backbone.layer3[1].conv1[2],
            "feature_map.backbone.layer3.1.conv2": self.feature_map.backbone.layer3[1].conv2,
            "feature_map.backbone.layer3.1.conv2.0": self.feature_map.backbone.layer3[1].conv2[0],
            "feature_map.backbone.layer3.1.conv2.1": self.feature_map.backbone.layer3[1].conv2[1],
            "feature_map.backbone.layer3.1.conv2.2": self.feature_map.backbone.layer3[1].conv2[2],
            "feature_map.backbone.layer3.2": self.feature_map.backbone.layer3[2],
            "feature_map.backbone.layer3.2.conv1": self.feature_map.backbone.layer3[2].conv1,
            "feature_map.backbone.layer3.2.conv1.0": self.feature_map.backbone.layer3[2].conv1[0],
            "feature_map.backbone.layer3.2.conv1.1": self.feature_map.backbone.layer3[2].conv1[1],
            "feature_map.backbone.layer3.2.conv1.2": self.feature_map.backbone.layer3[2].conv1[2],
            "feature_map.backbone.layer3.2.conv2": self.feature_map.backbone.layer3[2].conv2,
            "feature_map.backbone.layer3.2.conv2.0": self.feature_map.backbone.layer3[2].conv2[0],
            "feature_map.backbone.layer3.2.conv2.1": self.feature_map.backbone.layer3[2].conv2[1],
            "feature_map.backbone.layer3.2.conv2.2": self.feature_map.backbone.layer3[2].conv2[2],
            "feature_map.backbone.layer3.3": self.feature_map.backbone.layer3[3],
            "feature_map.backbone.layer3.3.conv1": self.feature_map.backbone.layer3[3].conv1,
            "feature_map.backbone.layer3.3.conv1.0": self.feature_map.backbone.layer3[3].conv1[0],
            "feature_map.backbone.layer3.3.conv1.1": self.feature_map.backbone.layer3[3].conv1[1],
            "feature_map.backbone.layer3.3.conv1.2": self.feature_map.backbone.layer3[3].conv1[2],
            "feature_map.backbone.layer3.3.conv2": self.feature_map.backbone.layer3[3].conv2,
            "feature_map.backbone.layer3.3.conv2.0": self.feature_map.backbone.layer3[3].conv2[0],
            "feature_map.backbone.layer3.3.conv2.1": self.feature_map.backbone.layer3[3].conv2[1],
            "feature_map.backbone.layer3.3.conv2.2": self.feature_map.backbone.layer3[3].conv2[2],
            "feature_map.backbone.layer3.4": self.feature_map.backbone.layer3[4],
            "feature_map.backbone.layer3.4.conv1": self.feature_map.backbone.layer3[4].conv1,
            "feature_map.backbone.layer3.4.conv1.0": self.feature_map.backbone.layer3[4].conv1[0],
            "feature_map.backbone.layer3.4.conv1.1": self.feature_map.backbone.layer3[4].conv1[1],
            "feature_map.backbone.layer3.4.conv1.2": self.feature_map.backbone.layer3[4].conv1[2],
            "feature_map.backbone.layer3.4.conv2": self.feature_map.backbone.layer3[4].conv2,
            "feature_map.backbone.layer3.4.conv2.0": self.feature_map.backbone.layer3[4].conv2[0],
            "feature_map.backbone.layer3.4.conv2.1": self.feature_map.backbone.layer3[4].conv2[1],
            "feature_map.backbone.layer3.4.conv2.2": self.feature_map.backbone.layer3[4].conv2[2],
            "feature_map.backbone.layer3.5": self.feature_map.backbone.layer3[5],
            "feature_map.backbone.layer3.5.conv1": self.feature_map.backbone.layer3[5].conv1,
            "feature_map.backbone.layer3.5.conv1.0": self.feature_map.backbone.layer3[5].conv1[0],
            "feature_map.backbone.layer3.5.conv1.1": self.feature_map.backbone.layer3[5].conv1[1],
            "feature_map.backbone.layer3.5.conv1.2": self.feature_map.backbone.layer3[5].conv1[2],
            "feature_map.backbone.layer3.5.conv2": self.feature_map.backbone.layer3[5].conv2,
            "feature_map.backbone.layer3.5.conv2.0": self.feature_map.backbone.layer3[5].conv2[0],
            "feature_map.backbone.layer3.5.conv2.1": self.feature_map.backbone.layer3[5].conv2[1],
            "feature_map.backbone.layer3.5.conv2.2": self.feature_map.backbone.layer3[5].conv2[2],
            "feature_map.backbone.layer3.6": self.feature_map.backbone.layer3[6],
            "feature_map.backbone.layer3.6.conv1": self.feature_map.backbone.layer3[6].conv1,
            "feature_map.backbone.layer3.6.conv1.0": self.feature_map.backbone.layer3[6].conv1[0],
            "feature_map.backbone.layer3.6.conv1.1": self.feature_map.backbone.layer3[6].conv1[1],
            "feature_map.backbone.layer3.6.conv1.2": self.feature_map.backbone.layer3[6].conv1[2],
            "feature_map.backbone.layer3.6.conv2": self.feature_map.backbone.layer3[6].conv2,
            "feature_map.backbone.layer3.6.conv2.0": self.feature_map.backbone.layer3[6].conv2[0],
            "feature_map.backbone.layer3.6.conv2.1": self.feature_map.backbone.layer3[6].conv2[1],
            "feature_map.backbone.layer3.6.conv2.2": self.feature_map.backbone.layer3[6].conv2[2],
            "feature_map.backbone.layer3.7": self.feature_map.backbone.layer3[7],
            "feature_map.backbone.layer3.7.conv1": self.feature_map.backbone.layer3[7].conv1,
            "feature_map.backbone.layer3.7.conv1.0": self.feature_map.backbone.layer3[7].conv1[0],
            "feature_map.backbone.layer3.7.conv1.1": self.feature_map.backbone.layer3[7].conv1[1],
            "feature_map.backbone.layer3.7.conv1.2": self.feature_map.backbone.layer3[7].conv1[2],
            "feature_map.backbone.layer3.7.conv2": self.feature_map.backbone.layer3[7].conv2,
            "feature_map.backbone.layer3.7.conv2.0": self.feature_map.backbone.layer3[7].conv2[0],
            "feature_map.backbone.layer3.7.conv2.1": self.feature_map.backbone.layer3[7].conv2[1],
            "feature_map.backbone.layer3.7.conv2.2": self.feature_map.backbone.layer3[7].conv2[2],
            "feature_map.backbone.conv4": self.feature_map.backbone.conv4,
            "feature_map.backbone.conv4.0": self.feature_map.backbone.conv4[0],
            "feature_map.backbone.conv4.1": self.feature_map.backbone.conv4[1],
            "feature_map.backbone.conv4.2": self.feature_map.backbone.conv4[2],
            "feature_map.backbone.layer4": self.feature_map.backbone.layer4,
            "feature_map.backbone.layer4.0": self.feature_map.backbone.layer4[0],
            "feature_map.backbone.layer4.0.conv1": self.feature_map.backbone.layer4[0].conv1,
            "feature_map.backbone.layer4.0.conv1.0": self.feature_map.backbone.layer4[0].conv1[0],
            "feature_map.backbone.layer4.0.conv1.1": self.feature_map.backbone.layer4[0].conv1[1],
            "feature_map.backbone.layer4.0.conv1.2": self.feature_map.backbone.layer4[0].conv1[2],
            "feature_map.backbone.layer4.0.conv2": self.feature_map.backbone.layer4[0].conv2,
            "feature_map.backbone.layer4.0.conv2.0": self.feature_map.backbone.layer4[0].conv2[0],
            "feature_map.backbone.layer4.0.conv2.1": self.feature_map.backbone.layer4[0].conv2[1],
            "feature_map.backbone.layer4.0.conv2.2": self.feature_map.backbone.layer4[0].conv2[2],
            "feature_map.backbone.layer4.1": self.feature_map.backbone.layer4[1],
            "feature_map.backbone.layer4.1.conv1": self.feature_map.backbone.layer4[1].conv1,
            "feature_map.backbone.layer4.1.conv1.0": self.feature_map.backbone.layer4[1].conv1[0],
            "feature_map.backbone.layer4.1.conv1.1": self.feature_map.backbone.layer4[1].conv1[1],
            "feature_map.backbone.layer4.1.conv1.2": self.feature_map.backbone.layer4[1].conv1[2],
            "feature_map.backbone.layer4.1.conv2": self.feature_map.backbone.layer4[1].conv2,
            "feature_map.backbone.layer4.1.conv2.0": self.feature_map.backbone.layer4[1].conv2[0],
            "feature_map.backbone.layer4.1.conv2.1": self.feature_map.backbone.layer4[1].conv2[1],
            "feature_map.backbone.layer4.1.conv2.2": self.feature_map.backbone.layer4[1].conv2[2],
            "feature_map.backbone.layer4.2": self.feature_map.backbone.layer4[2],
            "feature_map.backbone.layer4.2.conv1": self.feature_map.backbone.layer4[2].conv1,
            "feature_map.backbone.layer4.2.conv1.0": self.feature_map.backbone.layer4[2].conv1[0],
            "feature_map.backbone.layer4.2.conv1.1": self.feature_map.backbone.layer4[2].conv1[1],
            "feature_map.backbone.layer4.2.conv1.2": self.feature_map.backbone.layer4[2].conv1[2],
            "feature_map.backbone.layer4.2.conv2": self.feature_map.backbone.layer4[2].conv2,
            "feature_map.backbone.layer4.2.conv2.0": self.feature_map.backbone.layer4[2].conv2[0],
            "feature_map.backbone.layer4.2.conv2.1": self.feature_map.backbone.layer4[2].conv2[1],
            "feature_map.backbone.layer4.2.conv2.2": self.feature_map.backbone.layer4[2].conv2[2],
            "feature_map.backbone.layer4.3": self.feature_map.backbone.layer4[3],
            "feature_map.backbone.layer4.3.conv1": self.feature_map.backbone.layer4[3].conv1,
            "feature_map.backbone.layer4.3.conv1.0": self.feature_map.backbone.layer4[3].conv1[0],
            "feature_map.backbone.layer4.3.conv1.1": self.feature_map.backbone.layer4[3].conv1[1],
            "feature_map.backbone.layer4.3.conv1.2": self.feature_map.backbone.layer4[3].conv1[2],
            "feature_map.backbone.layer4.3.conv2": self.feature_map.backbone.layer4[3].conv2,
            "feature_map.backbone.layer4.3.conv2.0": self.feature_map.backbone.layer4[3].conv2[0],
            "feature_map.backbone.layer4.3.conv2.1": self.feature_map.backbone.layer4[3].conv2[1],
            "feature_map.backbone.layer4.3.conv2.2": self.feature_map.backbone.layer4[3].conv2[2],
            "feature_map.backbone.layer4.4": self.feature_map.backbone.layer4[4],
            "feature_map.backbone.layer4.4.conv1": self.feature_map.backbone.layer4[4].conv1,
            "feature_map.backbone.layer4.4.conv1.0": self.feature_map.backbone.layer4[4].conv1[0],
            "feature_map.backbone.layer4.4.conv1.1": self.feature_map.backbone.layer4[4].conv1[1],
            "feature_map.backbone.layer4.4.conv1.2": self.feature_map.backbone.layer4[4].conv1[2],
            "feature_map.backbone.layer4.4.conv2": self.feature_map.backbone.layer4[4].conv2,
            "feature_map.backbone.layer4.4.conv2.0": self.feature_map.backbone.layer4[4].conv2[0],
            "feature_map.backbone.layer4.4.conv2.1": self.feature_map.backbone.layer4[4].conv2[1],
            "feature_map.backbone.layer4.4.conv2.2": self.feature_map.backbone.layer4[4].conv2[2],
            "feature_map.backbone.layer4.5": self.feature_map.backbone.layer4[5],
            "feature_map.backbone.layer4.5.conv1": self.feature_map.backbone.layer4[5].conv1,
            "feature_map.backbone.layer4.5.conv1.0": self.feature_map.backbone.layer4[5].conv1[0],
            "feature_map.backbone.layer4.5.conv1.1": self.feature_map.backbone.layer4[5].conv1[1],
            "feature_map.backbone.layer4.5.conv1.2": self.feature_map.backbone.layer4[5].conv1[2],
            "feature_map.backbone.layer4.5.conv2": self.feature_map.backbone.layer4[5].conv2,
            "feature_map.backbone.layer4.5.conv2.0": self.feature_map.backbone.layer4[5].conv2[0],
            "feature_map.backbone.layer4.5.conv2.1": self.feature_map.backbone.layer4[5].conv2[1],
            "feature_map.backbone.layer4.5.conv2.2": self.feature_map.backbone.layer4[5].conv2[2],
            "feature_map.backbone.layer4.6": self.feature_map.backbone.layer4[6],
            "feature_map.backbone.layer4.6.conv1": self.feature_map.backbone.layer4[6].conv1,
            "feature_map.backbone.layer4.6.conv1.0": self.feature_map.backbone.layer4[6].conv1[0],
            "feature_map.backbone.layer4.6.conv1.1": self.feature_map.backbone.layer4[6].conv1[1],
            "feature_map.backbone.layer4.6.conv1.2": self.feature_map.backbone.layer4[6].conv1[2],
            "feature_map.backbone.layer4.6.conv2": self.feature_map.backbone.layer4[6].conv2,
            "feature_map.backbone.layer4.6.conv2.0": self.feature_map.backbone.layer4[6].conv2[0],
            "feature_map.backbone.layer4.6.conv2.1": self.feature_map.backbone.layer4[6].conv2[1],
            "feature_map.backbone.layer4.6.conv2.2": self.feature_map.backbone.layer4[6].conv2[2],
            "feature_map.backbone.layer4.7": self.feature_map.backbone.layer4[7],
            "feature_map.backbone.layer4.7.conv1": self.feature_map.backbone.layer4[7].conv1,
            "feature_map.backbone.layer4.7.conv1.0": self.feature_map.backbone.layer4[7].conv1[0],
            "feature_map.backbone.layer4.7.conv1.1": self.feature_map.backbone.layer4[7].conv1[1],
            "feature_map.backbone.layer4.7.conv1.2": self.feature_map.backbone.layer4[7].conv1[2],
            "feature_map.backbone.layer4.7.conv2": self.feature_map.backbone.layer4[7].conv2,
            "feature_map.backbone.layer4.7.conv2.0": self.feature_map.backbone.layer4[7].conv2[0],
            "feature_map.backbone.layer4.7.conv2.1": self.feature_map.backbone.layer4[7].conv2[1],
            "feature_map.backbone.layer4.7.conv2.2": self.feature_map.backbone.layer4[7].conv2[2],
            "feature_map.backbone.conv5": self.feature_map.backbone.conv5,
            "feature_map.backbone.conv5.0": self.feature_map.backbone.conv5[0],
            "feature_map.backbone.conv5.1": self.feature_map.backbone.conv5[1],
            "feature_map.backbone.conv5.2": self.feature_map.backbone.conv5[2],
            "feature_map.backbone.layer5": self.feature_map.backbone.layer5,
            "feature_map.backbone.layer5.0": self.feature_map.backbone.layer5[0],
            "feature_map.backbone.layer5.0.conv1": self.feature_map.backbone.layer5[0].conv1,
            "feature_map.backbone.layer5.0.conv1.0": self.feature_map.backbone.layer5[0].conv1[0],
            "feature_map.backbone.layer5.0.conv1.1": self.feature_map.backbone.layer5[0].conv1[1],
            "feature_map.backbone.layer5.0.conv1.2": self.feature_map.backbone.layer5[0].conv1[2],
            "feature_map.backbone.layer5.0.conv2": self.feature_map.backbone.layer5[0].conv2,
            "feature_map.backbone.layer5.0.conv2.0": self.feature_map.backbone.layer5[0].conv2[0],
            "feature_map.backbone.layer5.0.conv2.1": self.feature_map.backbone.layer5[0].conv2[1],
            "feature_map.backbone.layer5.0.conv2.2": self.feature_map.backbone.layer5[0].conv2[2],
            "feature_map.backbone.layer5.1": self.feature_map.backbone.layer5[1],
            "feature_map.backbone.layer5.1.conv1": self.feature_map.backbone.layer5[1].conv1,
            "feature_map.backbone.layer5.1.conv1.0": self.feature_map.backbone.layer5[1].conv1[0],
            "feature_map.backbone.layer5.1.conv1.1": self.feature_map.backbone.layer5[1].conv1[1],
            "feature_map.backbone.layer5.1.conv1.2": self.feature_map.backbone.layer5[1].conv1[2],
            "feature_map.backbone.layer5.1.conv2": self.feature_map.backbone.layer5[1].conv2,
            "feature_map.backbone.layer5.1.conv2.0": self.feature_map.backbone.layer5[1].conv2[0],
            "feature_map.backbone.layer5.1.conv2.1": self.feature_map.backbone.layer5[1].conv2[1],
            "feature_map.backbone.layer5.1.conv2.2": self.feature_map.backbone.layer5[1].conv2[2],
            "feature_map.backbone.layer5.2": self.feature_map.backbone.layer5[2],
            "feature_map.backbone.layer5.2.conv1": self.feature_map.backbone.layer5[2].conv1,
            "feature_map.backbone.layer5.2.conv1.0": self.feature_map.backbone.layer5[2].conv1[0],
            "feature_map.backbone.layer5.2.conv1.1": self.feature_map.backbone.layer5[2].conv1[1],
            "feature_map.backbone.layer5.2.conv1.2": self.feature_map.backbone.layer5[2].conv1[2],
            "feature_map.backbone.layer5.2.conv2": self.feature_map.backbone.layer5[2].conv2,
            "feature_map.backbone.layer5.2.conv2.0": self.feature_map.backbone.layer5[2].conv2[0],
            "feature_map.backbone.layer5.2.conv2.1": self.feature_map.backbone.layer5[2].conv2[1],
            "feature_map.backbone.layer5.2.conv2.2": self.feature_map.backbone.layer5[2].conv2[2],
            "feature_map.backbone.layer5.3": self.feature_map.backbone.layer5[3],
            "feature_map.backbone.layer5.3.conv1": self.feature_map.backbone.layer5[3].conv1,
            "feature_map.backbone.layer5.3.conv1.0": self.feature_map.backbone.layer5[3].conv1[0],
            "feature_map.backbone.layer5.3.conv1.1": self.feature_map.backbone.layer5[3].conv1[1],
            "feature_map.backbone.layer5.3.conv1.2": self.feature_map.backbone.layer5[3].conv1[2],
            "feature_map.backbone.layer5.3.conv2": self.feature_map.backbone.layer5[3].conv2,
            "feature_map.backbone.layer5.3.conv2.0": self.feature_map.backbone.layer5[3].conv2[0],
            "feature_map.backbone.layer5.3.conv2.1": self.feature_map.backbone.layer5[3].conv2[1],
            "feature_map.backbone.layer5.3.conv2.2": self.feature_map.backbone.layer5[3].conv2[2],
            "feature_map.backblock0": self.feature_map.backblock0,
            "feature_map.backblock0.conv0": self.feature_map.backblock0.conv0,
            "feature_map.backblock0.conv0.0": self.feature_map.backblock0.conv0[0],
            "feature_map.backblock0.conv0.1": self.feature_map.backblock0.conv0[1],
            "feature_map.backblock0.conv0.2": self.feature_map.backblock0.conv0[2],
            "feature_map.backblock0.conv1": self.feature_map.backblock0.conv1,
            "feature_map.backblock0.conv1.0": self.feature_map.backblock0.conv1[0],
            "feature_map.backblock0.conv1.1": self.feature_map.backblock0.conv1[1],
            "feature_map.backblock0.conv1.2": self.feature_map.backblock0.conv1[2],
            "feature_map.backblock0.conv2": self.feature_map.backblock0.conv2,
            "feature_map.backblock0.conv2.0": self.feature_map.backblock0.conv2[0],
            "feature_map.backblock0.conv2.1": self.feature_map.backblock0.conv2[1],
            "feature_map.backblock0.conv2.2": self.feature_map.backblock0.conv2[2],
            "feature_map.backblock0.conv3": self.feature_map.backblock0.conv3,
            "feature_map.backblock0.conv3.0": self.feature_map.backblock0.conv3[0],
            "feature_map.backblock0.conv3.1": self.feature_map.backblock0.conv3[1],
            "feature_map.backblock0.conv3.2": self.feature_map.backblock0.conv3[2],
            "feature_map.backblock0.conv4": self.feature_map.backblock0.conv4,
            "feature_map.backblock0.conv4.0": self.feature_map.backblock0.conv4[0],
            "feature_map.backblock0.conv4.1": self.feature_map.backblock0.conv4[1],
            "feature_map.backblock0.conv4.2": self.feature_map.backblock0.conv4[2],
            "feature_map.backblock0.conv5": self.feature_map.backblock0.conv5,
            "feature_map.backblock0.conv5.0": self.feature_map.backblock0.conv5[0],
            "feature_map.backblock0.conv5.1": self.feature_map.backblock0.conv5[1],
            "feature_map.backblock0.conv5.2": self.feature_map.backblock0.conv5[2],
            "feature_map.backblock0.conv6": self.feature_map.backblock0.conv6,
            "feature_map.conv1": self.feature_map.conv1,
            "feature_map.conv1.0": self.feature_map.conv1[0],
            "feature_map.conv1.1": self.feature_map.conv1[1],
            "feature_map.conv1.2": self.feature_map.conv1[2],
            "feature_map.backblock1": self.feature_map.backblock1,
            "feature_map.backblock1.conv0": self.feature_map.backblock1.conv0,
            "feature_map.backblock1.conv0.0": self.feature_map.backblock1.conv0[0],
            "feature_map.backblock1.conv0.1": self.feature_map.backblock1.conv0[1],
            "feature_map.backblock1.conv0.2": self.feature_map.backblock1.conv0[2],
            "feature_map.backblock1.conv1": self.feature_map.backblock1.conv1,
            "feature_map.backblock1.conv1.0": self.feature_map.backblock1.conv1[0],
            "feature_map.backblock1.conv1.1": self.feature_map.backblock1.conv1[1],
            "feature_map.backblock1.conv1.2": self.feature_map.backblock1.conv1[2],
            "feature_map.backblock1.conv2": self.feature_map.backblock1.conv2,
            "feature_map.backblock1.conv2.0": self.feature_map.backblock1.conv2[0],
            "feature_map.backblock1.conv2.1": self.feature_map.backblock1.conv2[1],
            "feature_map.backblock1.conv2.2": self.feature_map.backblock1.conv2[2],
            "feature_map.backblock1.conv3": self.feature_map.backblock1.conv3,
            "feature_map.backblock1.conv3.0": self.feature_map.backblock1.conv3[0],
            "feature_map.backblock1.conv3.1": self.feature_map.backblock1.conv3[1],
            "feature_map.backblock1.conv3.2": self.feature_map.backblock1.conv3[2],
            "feature_map.backblock1.conv4": self.feature_map.backblock1.conv4,
            "feature_map.backblock1.conv4.0": self.feature_map.backblock1.conv4[0],
            "feature_map.backblock1.conv4.1": self.feature_map.backblock1.conv4[1],
            "feature_map.backblock1.conv4.2": self.feature_map.backblock1.conv4[2],
            "feature_map.backblock1.conv5": self.feature_map.backblock1.conv5,
            "feature_map.backblock1.conv5.0": self.feature_map.backblock1.conv5[0],
            "feature_map.backblock1.conv5.1": self.feature_map.backblock1.conv5[1],
            "feature_map.backblock1.conv5.2": self.feature_map.backblock1.conv5[2],
            "feature_map.backblock1.conv6": self.feature_map.backblock1.conv6,
            "feature_map.conv2": self.feature_map.conv2,
            "feature_map.conv2.0": self.feature_map.conv2[0],
            "feature_map.conv2.1": self.feature_map.conv2[1],
            "feature_map.conv2.2": self.feature_map.conv2[2],
            "feature_map.backblock2": self.feature_map.backblock2,
            "feature_map.backblock2.conv0": self.feature_map.backblock2.conv0,
            "feature_map.backblock2.conv0.0": self.feature_map.backblock2.conv0[0],
            "feature_map.backblock2.conv0.1": self.feature_map.backblock2.conv0[1],
            "feature_map.backblock2.conv0.2": self.feature_map.backblock2.conv0[2],
            "feature_map.backblock2.conv1": self.feature_map.backblock2.conv1,
            "feature_map.backblock2.conv1.0": self.feature_map.backblock2.conv1[0],
            "feature_map.backblock2.conv1.1": self.feature_map.backblock2.conv1[1],
            "feature_map.backblock2.conv1.2": self.feature_map.backblock2.conv1[2],
            "feature_map.backblock2.conv2": self.feature_map.backblock2.conv2,
            "feature_map.backblock2.conv2.0": self.feature_map.backblock2.conv2[0],
            "feature_map.backblock2.conv2.1": self.feature_map.backblock2.conv2[1],
            "feature_map.backblock2.conv2.2": self.feature_map.backblock2.conv2[2],
            "feature_map.backblock2.conv3": self.feature_map.backblock2.conv3,
            "feature_map.backblock2.conv3.0": self.feature_map.backblock2.conv3[0],
            "feature_map.backblock2.conv3.1": self.feature_map.backblock2.conv3[1],
            "feature_map.backblock2.conv3.2": self.feature_map.backblock2.conv3[2],
            "feature_map.backblock2.conv4": self.feature_map.backblock2.conv4,
            "feature_map.backblock2.conv4.0": self.feature_map.backblock2.conv4[0],
            "feature_map.backblock2.conv4.1": self.feature_map.backblock2.conv4[1],
            "feature_map.backblock2.conv4.2": self.feature_map.backblock2.conv4[2],
            "feature_map.backblock2.conv5": self.feature_map.backblock2.conv5,
            "feature_map.backblock2.conv5.0": self.feature_map.backblock2.conv5[0],
            "feature_map.backblock2.conv5.1": self.feature_map.backblock2.conv5[1],
            "feature_map.backblock2.conv5.2": self.feature_map.backblock2.conv5[2],
            "feature_map.backblock2.conv6": self.feature_map.backblock2.conv6,
            "detect_1": self.detect_1,
            "detect_1.sigmoid": self.detect_1.sigmoid,
            "detect_2": self.detect_2,
            "detect_2.sigmoid": self.detect_2.sigmoid,
            "detect_3": self.detect_3,
            "detect_3.sigmoid": self.detect_3.sigmoid,

        }

        self.in_shapes = {}

        self.out_shapes = {
            'INPUT': [-1, 3, 416, 416],
            'feature_map.backbone.conv0.0': [-1, 32, 416, 416],
            'feature_map.backbone.conv0.1': [-1, 32, 416, 416],
            'feature_map.backbone.conv0.2': [-1, 32, 416, 416],
            'feature_map.backbone.conv1.0': [-1, 64, 208, 208],
            'feature_map.backbone.conv1.1': [-1, 64, 208, 208],
            'feature_map.backbone.conv1.2': [-1, 64, 208, 208],
            'feature_map.backbone.layer1.0.conv1.0': [-1, 32, 208, 208],
            'feature_map.backbone.layer1.0.conv1.1': [-1, 32, 208, 208],
            'feature_map.backbone.layer1.0.conv1.2': [-1, 32, 208, 208],
            'feature_map.backbone.layer1.0.conv2.0': [-1, 64, 208, 208],
            'feature_map.backbone.layer1.0.conv2.1': [-1, 64, 208, 208],
            'feature_map.backbone.layer1.0.conv2.2': [-1, 64, 208, 208],
            'feature_map.backbone.conv2.0': [-1, 128, 104, 104],
            'feature_map.backbone.conv2.1': [-1, 128, 104, 104],
            'feature_map.backbone.conv2.2': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.0.conv1.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv1.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv1.2': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.0.conv2.0': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.0.conv2.1': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.0.conv2.2': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.1.conv1.0': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv1.1': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv1.2': [-1, 64, 104, 104],
            'feature_map.backbone.layer2.1.conv2.0': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.1.conv2.1': [-1, 128, 104, 104],
            'feature_map.backbone.layer2.1.conv2.2': [-1, 128, 104, 104],
            'feature_map.backbone.conv3.0': [-1, 256, 52, 52],
            'feature_map.backbone.conv3.1': [-1, 256, 52, 52],
            'feature_map.backbone.conv3.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.0.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.0.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.0.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.0.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.1.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.1.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.1.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.1.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.2.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.2.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.2.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.2.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.3.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.3.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.3.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.3.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.4.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.4.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.4.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.4.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.5.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.5.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.5.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.5.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.6.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.6.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.6.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.6.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.7.conv1.0': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv1.1': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv1.2': [-1, 128, 52, 52],
            'feature_map.backbone.layer3.7.conv2.0': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.7.conv2.1': [-1, 256, 52, 52],
            'feature_map.backbone.layer3.7.conv2.2': [-1, 256, 52, 52],
            'feature_map.backbone.conv4.0': [-1, 512, 26, 26],
            'feature_map.backbone.conv4.1': [-1, 512, 26, 26],
            'feature_map.backbone.conv4.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.0.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.0.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.0.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.0.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.1.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.1.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.1.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.1.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.2.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.2.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.2.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.2.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.3.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.3.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.3.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.3.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.4.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.4.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.4.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.4.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.5.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.5.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.5.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.5.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.6.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.6.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.6.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.6.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.7.conv1.0': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv1.1': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv1.2': [-1, 256, 26, 26],
            'feature_map.backbone.layer4.7.conv2.0': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.7.conv2.1': [-1, 512, 26, 26],
            'feature_map.backbone.layer4.7.conv2.2': [-1, 512, 26, 26],
            'feature_map.backbone.conv5.0': [-1, 1024, 13, 13],
            'feature_map.backbone.conv5.1': [-1, 1024, 13, 13],
            'feature_map.backbone.conv5.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.0.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.0.conv2.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.0.conv2.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.0.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.1.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.1.conv2.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.1.conv2.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.1.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.2.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.2.conv2.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.2.conv2.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.2.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.3.conv1.0': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv1.1': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv1.2': [-1, 512, 13, 13],
            'feature_map.backbone.layer5.3.conv2.0': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.3.conv2.1': [-1, 1024, 13, 13],
            'feature_map.backbone.layer5.3.conv2.2': [-1, 1024, 13, 13],
            'feature_map.backblock0.conv0.0': [-1, 512, 13, 13],
            'feature_map.backblock0.conv0.1': [-1, 512, 13, 13],
            'feature_map.backblock0.conv0.2': [-1, 512, 13, 13],
            'feature_map.backblock0.conv1.0': [-1, 1024, 13, 13],
            'feature_map.backblock0.conv1.1': [-1, 1024, 13, 13],
            'feature_map.backblock0.conv1.2': [-1, 1024, 13, 13],
            'feature_map.backblock0.conv2.0': [-1, 512, 13, 13],
            'feature_map.backblock0.conv2.1': [-1, 512, 13, 13],
            'feature_map.backblock0.conv2.2': [-1, 512, 13, 13],
            'feature_map.backblock0.conv3.0': [-1, 1024, 13, 13],
            'feature_map.backblock0.conv3.1': [-1, 1024, 13, 13],
            'feature_map.backblock0.conv3.2': [-1, 1024, 13, 13],
            'feature_map.backblock0.conv4.0': [-1, 512, 13, 13],
            'feature_map.backblock0.conv4.1': [-1, 512, 13, 13],
            'feature_map.backblock0.conv4.2': [-1, 512, 13, 13],
            'feature_map.backblock0.conv5.0': [-1, 1024, 13, 13],
            'feature_map.backblock0.conv5.1': [-1, 1024, 13, 13],
            'feature_map.backblock0.conv5.2': [-1, 1024, 13, 13],
            'feature_map.backblock0.conv6': [-1, 255, 13, 13],
            'feature_map.conv1.0': [-1, 256, 13, 13],
            'feature_map.conv1.1': [-1, 256, 13, 13],
            'feature_map.conv1.2': [-1, 256, 13, 13],
            'feature_map.backblock1.conv0.0': [-1, 256, 26, 26],
            'feature_map.backblock1.conv0.1': [-1, 256, 26, 26],
            'feature_map.backblock1.conv0.2': [-1, 256, 26, 26],
            'feature_map.backblock1.conv1.0': [-1, 512, 26, 26],
            'feature_map.backblock1.conv1.1': [-1, 512, 26, 26],
            'feature_map.backblock1.conv1.2': [-1, 512, 26, 26],
            'feature_map.backblock1.conv2.0': [-1, 256, 26, 26],
            'feature_map.backblock1.conv2.1': [-1, 256, 26, 26],
            'feature_map.backblock1.conv2.2': [-1, 256, 26, 26],
            'feature_map.backblock1.conv3.0': [-1, 512, 26, 26],
            'feature_map.backblock1.conv3.1': [-1, 512, 26, 26],
            'feature_map.backblock1.conv3.2': [-1, 512, 26, 26],
            'feature_map.backblock1.conv4.0': [-1, 256, 26, 26],
            'feature_map.backblock1.conv4.1': [-1, 256, 26, 26],
            'feature_map.backblock1.conv4.2': [-1, 256, 26, 26],
            'feature_map.backblock1.conv5.0': [-1, 512, 26, 26],
            'feature_map.backblock1.conv5.1': [-1, 512, 26, 26],
            'feature_map.backblock1.conv5.2': [-1, 512, 26, 26],
            'feature_map.backblock1.conv6': [-1, 255, 26, 26],
            'feature_map.conv2.0': [-1, 128, 26, 26],
            'feature_map.conv2.1': [-1, 128, 26, 26],
            'feature_map.conv2.2': [-1, 128, 26, 26],
            'feature_map.backblock2.conv0.0': [-1, 128, 52, 52],
            'feature_map.backblock2.conv0.1': [-1, 128, 52, 52],
            'feature_map.backblock2.conv0.2': [-1, 128, 52, 52],
            'feature_map.backblock2.conv1.0': [-1, 256, 52, 52],
            'feature_map.backblock2.conv1.1': [-1, 256, 52, 52],
            'feature_map.backblock2.conv1.2': [-1, 256, 52, 52],
            'feature_map.backblock2.conv2.0': [-1, 128, 52, 52],
            'feature_map.backblock2.conv2.1': [-1, 128, 52, 52],
            'feature_map.backblock2.conv2.2': [-1, 128, 52, 52],
            'feature_map.backblock2.conv3.0': [-1, 256, 52, 52],
            'feature_map.backblock2.conv3.1': [-1, 256, 52, 52],
            'feature_map.backblock2.conv3.2': [-1, 256, 52, 52],
            'feature_map.backblock2.conv4.0': [-1, 128, 52, 52],
            'feature_map.backblock2.conv4.1': [-1, 128, 52, 52],
            'feature_map.backblock2.conv4.2': [-1, 128, 52, 52],
            'feature_map.backblock2.conv5.0': [-1, 256, 52, 52],
            'feature_map.backblock2.conv5.1': [-1, 256, 52, 52],
            'feature_map.backblock2.conv5.2': [-1, 256, 52, 52],
            'feature_map.backblock2.conv6': [-1, 255, 52, 52],
            'detect_1.sigmoid': [-1, 13, 13, 3, 2],
            'detect_2.sigmoid': [-1, 26, 26, 3, 2],
            'detect_3.sigmoid': [-1, 52, 52, 3, 2],
            'OUTPUT1': [-1, 13, 13, 3, 2],
            'OUTPUT2':[-1, 26, 26, 3, 2],
            'OUTPUT3':[-1, 52, 52, 3, 2]
        }

        self.orders = {
            'feature_map.backbone.conv0.0': ["INPUT", "feature_map.backbone.conv0.1"],
            'feature_map.backbone.conv0.1': ["feature_map.backbone.conv0.0", "feature_map.backbone.conv0.2"],
            'feature_map.backbone.conv0.2': ["feature_map.backbone.conv0.1", "feature_map.backbone.conv1.0"],
            'feature_map.backbone.conv1.0': ["feature_map.backbone.conv0.2", "feature_map.backbone.conv1.1"],
            'feature_map.backbone.conv1.1': ["feature_map.backbone.conv1.0", "feature_map.backbone.conv1.2"],
            'feature_map.backbone.conv1.2': ["feature_map.backbone.conv1.1", "feature_map.backbone.layer1.0.conv1.0"],
            'feature_map.backbone.layer1.0.conv1.0': ["feature_map.backbone.conv1.2",
                                                      "feature_map.backbone.layer1.0.conv1.1"],
            'feature_map.backbone.layer1.0.conv1.1': ["feature_map.backbone.layer1.0.conv1.0",
                                                      "feature_map.backbone.layer1.0.conv1.2"],
            'feature_map.backbone.layer1.0.conv1.2': ["feature_map.backbone.layer1.0.conv1.1",
                                                      "feature_map.backbone.layer1.0.conv2.0"],
            'feature_map.backbone.layer1.0.conv2.0': ["feature_map.backbone.layer1.0.conv1.2",
                                                      "feature_map.backbone.layer1.0.conv2.1"],
            'feature_map.backbone.layer1.0.conv2.1': ["feature_map.backbone.layer1.0.conv2.0",
                                                      "feature_map.backbone.layer1.0.conv2.2"],
            'feature_map.backbone.layer1.0.conv2.2': ["feature_map.backbone.layer1.0.conv2.1",
                                                      "feature_map.backbone.conv2.0"],
            'feature_map.backbone.conv2.0': ["feature_map.backbone.layer1.0.conv2.2", "feature_map.backbone.conv2.1"],
            'feature_map.backbone.conv2.1': ["feature_map.backbone.conv2.0", "feature_map.backbone.conv2.2"],
            'feature_map.backbone.conv2.2': ["feature_map.backbone.conv2.1", "feature_map.backbone.layer2.0.conv1.0"],
            'feature_map.backbone.layer2.0.conv1.0': ["feature_map.backbone.conv2.2",
                                                      "feature_map.backbone.layer2.0.conv1.1"],
            'feature_map.backbone.layer2.0.conv1.1': ["feature_map.backbone.layer2.0.conv1.0",
                                                      "feature_map.backbone.layer2.0.conv1.2"],
            'feature_map.backbone.layer2.0.conv1.2': ["feature_map.backbone.layer2.0.conv1.1",
                                                      "feature_map.backbone.layer2.0.conv2.0"],
            'feature_map.backbone.layer2.0.conv2.0': ["feature_map.backbone.layer2.0.conv1.2",
                                                      "feature_map.backbone.layer2.0.conv2.1"],
            'feature_map.backbone.layer2.0.conv2.1': ["feature_map.backbone.layer2.0.conv2.0",
                                                      "feature_map.backbone.layer2.0.conv2.2"],
            'feature_map.backbone.layer2.0.conv2.2': ["feature_map.backbone.layer2.0.conv2.1",
                                                      "feature_map.backbone.layer2.1.conv1.0"],
            'feature_map.backbone.layer2.1.conv1.0': ["feature_map.backbone.layer2.0.conv2.2",
                                                      "feature_map.backbone.layer2.1.conv1.1"],
            'feature_map.backbone.layer2.1.conv1.1': ["feature_map.backbone.layer2.1.conv1.0",
                                                      "feature_map.backbone.layer2.1.conv1.2"],
            'feature_map.backbone.layer2.1.conv1.2': ["feature_map.backbone.layer2.1.conv1.1",
                                                      "feature_map.backbone.layer2.1.conv2.0"],
            'feature_map.backbone.layer2.1.conv2.0': ["feature_map.backbone.layer2.1.conv1.2",
                                                      "feature_map.backbone.layer2.1.conv2.1"],
            'feature_map.backbone.layer2.1.conv2.1': ["feature_map.backbone.layer2.1.conv2.0",
                                                      "feature_map.backbone.layer2.1.conv2.2"],
            'feature_map.backbone.layer2.1.conv2.2': ["feature_map.backbone.layer2.1.conv2.1",
                                                      "feature_map.backbone.conv3.0"],
            'feature_map.backbone.conv3.0': ["feature_map.backbone.layer2.1.conv2.2", "feature_map.backbone.conv3.1"],
            'feature_map.backbone.conv3.1': ["feature_map.backbone.conv3.0", "feature_map.backbone.conv3.2"],
            'feature_map.backbone.conv3.2': ["feature_map.backbone.conv3.1", "feature_map.backbone.layer3.0.conv1.0"],
            'feature_map.backbone.layer3.0.conv1.0': ["feature_map.backbone.conv3.2",
                                                      "feature_map.backbone.layer3.0.conv1.1"],
            'feature_map.backbone.layer3.0.conv1.1': ["feature_map.backbone.layer3.0.conv1.0",
                                                      "feature_map.backbone.layer3.0.conv1.2"],
            'feature_map.backbone.layer3.0.conv1.2': ["feature_map.backbone.layer3.0.conv1.1",
                                                      "feature_map.backbone.layer3.0.conv2.0"],
            'feature_map.backbone.layer3.0.conv2.0': ["feature_map.backbone.layer3.0.conv1.2",
                                                      "feature_map.backbone.layer3.0.conv2.1"],
            'feature_map.backbone.layer3.0.conv2.1': ["feature_map.backbone.layer3.0.conv2.0",
                                                      "feature_map.backbone.layer3.0.conv2.2"],
            'feature_map.backbone.layer3.0.conv2.2': ["feature_map.backbone.layer3.0.conv2.1",
                                                      "feature_map.backbone.layer3.1.conv1.0"],
            'feature_map.backbone.layer3.1.conv1.0': ["feature_map.backbone.layer3.0.conv2.2",
                                                      "feature_map.backbone.layer3.1.conv1.1"],
            'feature_map.backbone.layer3.1.conv1.1': ["feature_map.backbone.layer3.1.conv1.0",
                                                      "feature_map.backbone.layer3.1.conv1.2"],
            'feature_map.backbone.layer3.1.conv1.2': ["feature_map.backbone.layer3.1.conv1.1",
                                                      "feature_map.backbone.layer3.1.conv2.0"],
            'feature_map.backbone.layer3.1.conv2.0': ["feature_map.backbone.layer3.1.conv1.2",
                                                      "feature_map.backbone.layer3.1.conv2.1"],
            'feature_map.backbone.layer3.1.conv2.1': ["feature_map.backbone.layer3.1.conv2.0",
                                                      "feature_map.backbone.layer3.1.conv2.2"],
            'feature_map.backbone.layer3.1.conv2.2': ["feature_map.backbone.layer3.1.conv2.1",
                                                      "feature_map.backbone.layer3.2.conv1.0"],
            'feature_map.backbone.layer3.2.conv1.0': ["feature_map.backbone.layer3.1.conv2.2",
                                                      "feature_map.backbone.layer3.2.conv1.1"],
            'feature_map.backbone.layer3.2.conv1.1': ["feature_map.backbone.layer3.2.conv1.0",
                                                      "feature_map.backbone.layer3.2.conv1.2"],
            'feature_map.backbone.layer3.2.conv1.2': ["feature_map.backbone.layer3.2.conv1.1",
                                                      "feature_map.backbone.layer3.2.conv2.0"],
            'feature_map.backbone.layer3.2.conv2.0': ["feature_map.backbone.layer3.2.conv1.2",
                                                      "feature_map.backbone.layer3.2.conv2.1"],
            'feature_map.backbone.layer3.2.conv2.1': ["feature_map.backbone.layer3.2.conv2.0",
                                                      "feature_map.backbone.layer3.2.conv2.2"],
            'feature_map.backbone.layer3.2.conv2.2': ["feature_map.backbone.layer3.2.conv2.1",
                                                      "feature_map.backbone.layer3.3.conv1.0"],
            'feature_map.backbone.layer3.3.conv1.0': ["feature_map.backbone.layer3.2.conv2.2",
                                                      "feature_map.backbone.layer3.3.conv1.1"],
            'feature_map.backbone.layer3.3.conv1.1': ["feature_map.backbone.layer3.3.conv1.0",
                                                      "feature_map.backbone.layer3.3.conv1.2"],
            'feature_map.backbone.layer3.3.conv1.2': ["feature_map.backbone.layer3.3.conv1.1",
                                                      "feature_map.backbone.layer3.3.conv2.0"],
            'feature_map.backbone.layer3.3.conv2.0': ["feature_map.backbone.layer3.3.conv1.2",
                                                      "feature_map.backbone.layer3.3.conv2.1"],
            'feature_map.backbone.layer3.3.conv2.1': ["feature_map.backbone.layer3.3.conv2.0",
                                                      "feature_map.backbone.layer3.3.conv2.2"],
            'feature_map.backbone.layer3.3.conv2.2': ["feature_map.backbone.layer3.3.conv2.1",
                                                      "feature_map.backbone.layer3.4.conv1.0"],
            'feature_map.backbone.layer3.4.conv1.0': ["feature_map.backbone.layer3.3.conv2.2",
                                                      "feature_map.backbone.layer3.4.conv1.1"],
            'feature_map.backbone.layer3.4.conv1.1': ["feature_map.backbone.layer3.4.conv1.0",
                                                      "feature_map.backbone.layer3.4.conv1.2"],
            'feature_map.backbone.layer3.4.conv1.2': ["feature_map.backbone.layer3.4.conv1.1",
                                                      "feature_map.backbone.layer3.4.conv2.0"],
            'feature_map.backbone.layer3.4.conv2.0': ["feature_map.backbone.layer3.4.conv1.2",
                                                      "feature_map.backbone.layer3.4.conv2.1"],
            'feature_map.backbone.layer3.4.conv2.1': ["feature_map.backbone.layer3.4.conv2.0",
                                                      "feature_map.backbone.layer3.4.conv2.2"],
            'feature_map.backbone.layer3.4.conv2.2': ["feature_map.backbone.layer3.4.conv2.1",
                                                      "feature_map.backbone.layer3.5.conv1.0"],
            'feature_map.backbone.layer3.5.conv1.0': ["feature_map.backbone.layer3.4.conv2.2",
                                                      "feature_map.backbone.layer3.5.conv1.1"],
            'feature_map.backbone.layer3.5.conv1.1': ["feature_map.backbone.layer3.5.conv1.0",
                                                      "feature_map.backbone.layer3.5.conv1.2"],
            'feature_map.backbone.layer3.5.conv1.2': ["feature_map.backbone.layer3.5.conv1.1",
                                                      "feature_map.backbone.layer3.5.conv2.0"],
            'feature_map.backbone.layer3.5.conv2.0': ["feature_map.backbone.layer3.5.conv1.2",
                                                      "feature_map.backbone.layer3.5.conv2.1"],
            'feature_map.backbone.layer3.5.conv2.1': ["feature_map.backbone.layer3.5.conv2.0",
                                                      "feature_map.backbone.layer3.5.conv2.2"],
            'feature_map.backbone.layer3.5.conv2.2': ["feature_map.backbone.layer3.5.conv2.1",
                                                      "feature_map.backbone.layer3.6.conv1.0"],
            'feature_map.backbone.layer3.6.conv1.0': ["feature_map.backbone.layer3.5.conv2.2",
                                                      "feature_map.backbone.layer3.6.conv1.1"],
            'feature_map.backbone.layer3.6.conv1.1': ["feature_map.backbone.layer3.6.conv1.0",
                                                      "feature_map.backbone.layer3.6.conv1.2"],
            'feature_map.backbone.layer3.6.conv1.2': ["feature_map.backbone.layer3.6.conv1.1",
                                                      "feature_map.backbone.layer3.6.conv2.0"],
            'feature_map.backbone.layer3.6.conv2.0': ["feature_map.backbone.layer3.6.conv1.2",
                                                      "feature_map.backbone.layer3.6.conv2.1"],
            'feature_map.backbone.layer3.6.conv2.1': ["feature_map.backbone.layer3.6.conv2.0",
                                                      "feature_map.backbone.layer3.6.conv2.2"],
            'feature_map.backbone.layer3.6.conv2.2': ["feature_map.backbone.layer3.6.conv2.1",
                                                      "feature_map.backbone.layer3.7.conv1.0"],
            'feature_map.backbone.layer3.7.conv1.0': ["feature_map.backbone.layer3.6.conv2.2",
                                                      "feature_map.backbone.layer3.7.conv1.1"],
            'feature_map.backbone.layer3.7.conv1.1': ["feature_map.backbone.layer3.7.conv1.0",
                                                      "feature_map.backbone.layer3.7.conv1.2"],
            'feature_map.backbone.layer3.7.conv1.2': ["feature_map.backbone.layer3.7.conv1.1",
                                                      "feature_map.backbone.layer3.7.conv2.0"],
            'feature_map.backbone.layer3.7.conv2.0': ["feature_map.backbone.layer3.7.conv1.2",
                                                      "feature_map.backbone.layer3.7.conv2.1"],
            'feature_map.backbone.layer3.7.conv2.1': ["feature_map.backbone.layer3.7.conv2.0",
                                                      "feature_map.backbone.layer3.7.conv2.2"],
            'feature_map.backbone.layer3.7.conv2.2': ["feature_map.backbone.layer3.7.conv2.1",
                                                      ["feature_map.backbone.conv4.0",
                                                       "feature_map.backblock2.conv0.0"]],  # layer3输出接到最后一个yoloblock前
            'feature_map.backbone.conv4.0': ["feature_map.backbone.layer3.7.conv2.2", "feature_map.backbone.conv4.1"],
            'feature_map.backbone.conv4.1': ["feature_map.backbone.conv4.0", "feature_map.backbone.conv4.2"],
            'feature_map.backbone.conv4.2': ["feature_map.backbone.conv4.1", "feature_map.backbone.layer4.0.conv1.0"],
            'feature_map.backbone.layer4.0.conv1.0': ["feature_map.backbone.conv4.2",
                                                      "feature_map.backbone.layer4.0.conv1.1"],
            'feature_map.backbone.layer4.0.conv1.1': ["feature_map.backbone.layer4.0.conv1.0",
                                                      "feature_map.backbone.layer4.0.conv1.2"],
            'feature_map.backbone.layer4.0.conv1.2': ["feature_map.backbone.layer4.0.conv1.1",
                                                      "feature_map.backbone.layer4.0.conv2.0"],
            'feature_map.backbone.layer4.0.conv2.0': ["feature_map.backbone.layer4.0.conv1.2",
                                                      "feature_map.backbone.layer4.0.conv2.1"],
            'feature_map.backbone.layer4.0.conv2.1': ["feature_map.backbone.layer4.0.conv2.0",
                                                      "feature_map.backbone.layer4.0.conv2.2"],
            'feature_map.backbone.layer4.0.conv2.2': ["feature_map.backbone.layer4.0.conv2.1",
                                                      "feature_map.backbone.layer4.1.conv1.0"],
            'feature_map.backbone.layer4.1.conv1.0': ["feature_map.backbone.layer4.0.conv2.2",
                                                      "feature_map.backbone.layer4.1.conv1.1"],
            'feature_map.backbone.layer4.1.conv1.1': ["feature_map.backbone.layer4.1.conv1.0",
                                                      "feature_map.backbone.layer4.1.conv1.2"],
            'feature_map.backbone.layer4.1.conv1.2': ["feature_map.backbone.layer4.1.conv1.1",
                                                      "feature_map.backbone.layer4.1.conv2.0"],
            'feature_map.backbone.layer4.1.conv2.0': ["feature_map.backbone.layer4.1.conv1.2",
                                                      "feature_map.backbone.layer4.1.conv2.1"],
            'feature_map.backbone.layer4.1.conv2.1': ["feature_map.backbone.layer4.1.conv2.0",
                                                      "feature_map.backbone.layer4.1.conv2.2"],
            'feature_map.backbone.layer4.1.conv2.2': ["feature_map.backbone.layer4.1.conv2.1",
                                                      "feature_map.backbone.layer4.2.conv1.0"],
            'feature_map.backbone.layer4.2.conv1.0': ["feature_map.backbone.layer4.1.conv2.2",
                                                      "feature_map.backbone.layer4.2.conv1.1"],
            'feature_map.backbone.layer4.2.conv1.1': ["feature_map.backbone.layer4.2.conv1.0",
                                                      "feature_map.backbone.layer4.2.conv1.2"],
            'feature_map.backbone.layer4.2.conv1.2': ["feature_map.backbone.layer4.2.conv1.1",
                                                      "feature_map.backbone.layer4.2.conv2.0"],
            'feature_map.backbone.layer4.2.conv2.0': ["feature_map.backbone.layer4.2.conv1.2",
                                                      "feature_map.backbone.layer4.2.conv2.1"],
            'feature_map.backbone.layer4.2.conv2.1': ["feature_map.backbone.layer4.2.conv2.0",
                                                      "feature_map.backbone.layer4.2.conv2.2"],
            'feature_map.backbone.layer4.2.conv2.2': ["feature_map.backbone.layer4.2.conv2.1",
                                                      "feature_map.backbone.layer4.3.conv1.0"],
            'feature_map.backbone.layer4.3.conv1.0': ["feature_map.backbone.layer4.2.conv2.2",
                                                      "feature_map.backbone.layer4.3.conv1.1"],
            'feature_map.backbone.layer4.3.conv1.1': ["feature_map.backbone.layer4.3.conv1.0",
                                                      "feature_map.backbone.layer4.3.conv1.2"],
            'feature_map.backbone.layer4.3.conv1.2': ["feature_map.backbone.layer4.3.conv1.1",
                                                      "feature_map.backbone.layer4.3.conv2.0"],
            'feature_map.backbone.layer4.3.conv2.0': ["feature_map.backbone.layer4.3.conv1.2",
                                                      "feature_map.backbone.layer4.3.conv2.1"],
            'feature_map.backbone.layer4.3.conv2.1': ["feature_map.backbone.layer4.3.conv2.0",
                                                      "feature_map.backbone.layer4.3.conv2.2"],
            'feature_map.backbone.layer4.3.conv2.2': ["feature_map.backbone.layer4.3.conv2.1",
                                                      "feature_map.backbone.layer4.4.conv1.0"],
            'feature_map.backbone.layer4.4.conv1.0': ["feature_map.backbone.layer4.3.conv2.2",
                                                      "feature_map.backbone.layer4.4.conv1.1"],
            'feature_map.backbone.layer4.4.conv1.1': ["feature_map.backbone.layer4.4.conv1.0",
                                                      "feature_map.backbone.layer4.4.conv1.2"],
            'feature_map.backbone.layer4.4.conv1.2': ["feature_map.backbone.layer4.4.conv1.1",
                                                      "feature_map.backbone.layer4.4.conv2.0"],
            'feature_map.backbone.layer4.4.conv2.0': ["feature_map.backbone.layer4.4.conv1.2",
                                                      "feature_map.backbone.layer4.4.conv2.1"],
            'feature_map.backbone.layer4.4.conv2.1': ["feature_map.backbone.layer4.4.conv2.0",
                                                      "feature_map.backbone.layer4.4.conv2.2"],
            'feature_map.backbone.layer4.4.conv2.2': ["feature_map.backbone.layer4.4.conv2.1",
                                                      "feature_map.backbone.layer4.5.conv1.0"],
            'feature_map.backbone.layer4.5.conv1.0': ["feature_map.backbone.layer4.4.conv2.2",
                                                      "feature_map.backbone.layer4.5.conv1.1"],
            'feature_map.backbone.layer4.5.conv1.1': ["feature_map.backbone.layer4.5.conv1.0",
                                                      "feature_map.backbone.layer4.5.conv1.2"],
            'feature_map.backbone.layer4.5.conv1.2': ["feature_map.backbone.layer4.5.conv1.1",
                                                      "feature_map.backbone.layer4.5.conv2.0"],
            'feature_map.backbone.layer4.5.conv2.0': ["feature_map.backbone.layer4.5.conv1.2",
                                                      "feature_map.backbone.layer4.5.conv2.1"],
            'feature_map.backbone.layer4.5.conv2.1': ["feature_map.backbone.layer4.5.conv2.0",
                                                      "feature_map.backbone.layer4.5.conv2.2"],
            'feature_map.backbone.layer4.5.conv2.2': ["feature_map.backbone.layer4.5.conv2.1",
                                                      "feature_map.backbone.layer4.6.conv1.0"],
            'feature_map.backbone.layer4.6.conv1.0': ["feature_map.backbone.layer4.5.conv2.2",
                                                      "feature_map.backbone.layer4.6.conv1.1"],
            'feature_map.backbone.layer4.6.conv1.1': ["feature_map.backbone.layer4.6.conv1.0",
                                                      "feature_map.backbone.layer4.6.conv1.2"],
            'feature_map.backbone.layer4.6.conv1.2': ["feature_map.backbone.layer4.6.conv1.1",
                                                      "feature_map.backbone.layer4.6.conv2.0"],
            'feature_map.backbone.layer4.6.conv2.0': ["feature_map.backbone.layer4.6.conv1.2",
                                                      "feature_map.backbone.layer4.6.conv2.1"],
            'feature_map.backbone.layer4.6.conv2.1': ["feature_map.backbone.layer4.6.conv2.0",
                                                      "feature_map.backbone.layer4.6.conv2.2"],
            'feature_map.backbone.layer4.6.conv2.2': ["feature_map.backbone.layer4.6.conv2.1",
                                                      "feature_map.backbone.layer4.7.conv1.0"],
            'feature_map.backbone.layer4.7.conv1.0': ["feature_map.backbone.layer4.6.conv2.2",
                                                      "feature_map.backbone.layer4.7.conv1.1"],
            'feature_map.backbone.layer4.7.conv1.1': ["feature_map.backbone.layer4.7.conv1.0",
                                                      "feature_map.backbone.layer4.7.conv1.2"],
            'feature_map.backbone.layer4.7.conv1.2': ["feature_map.backbone.layer4.7.conv1.1",
                                                      "feature_map.backbone.layer4.7.conv2.0"],
            'feature_map.backbone.layer4.7.conv2.0': ["feature_map.backbone.layer4.7.conv1.2",
                                                      "feature_map.backbone.layer4.7.conv2.1"],
            'feature_map.backbone.layer4.7.conv2.1': ["feature_map.backbone.layer4.7.conv2.0",
                                                      "feature_map.backbone.layer4.7.conv2.2"],
            'feature_map.backbone.layer4.7.conv2.2': ["feature_map.backbone.layer4.7.conv2.1",
                                                      ["feature_map.backbone.conv5.0",
                                                       "feature_map.backblock1.conv0.0"]],  # layer4接第二个yoloblock开头
            'feature_map.backbone.conv5.0': ["feature_map.backbone.layer4.7.conv2.2", "feature_map.backbone.conv5.1"],
            'feature_map.backbone.conv5.1': ["feature_map.backbone.conv5.0", "feature_map.backbone.conv5.2"],
            'feature_map.backbone.conv5.2': ["feature_map.backbone.conv5.1", "feature_map.backbone.layer5.0.conv1.0"],
            'feature_map.backbone.layer5.0.conv1.0': ["feature_map.backbone.conv5.2",
                                                      "feature_map.backbone.layer5.0.conv1.1"],
            'feature_map.backbone.layer5.0.conv1.1': ["feature_map.backbone.layer5.0.conv1.0",
                                                      "feature_map.backbone.layer5.0.conv1.2"],
            'feature_map.backbone.layer5.0.conv1.2': ["feature_map.backbone.layer5.0.conv1.1",
                                                      "feature_map.backbone.layer5.0.conv2.0"],
            'feature_map.backbone.layer5.0.conv2.0': ["feature_map.backbone.layer5.0.conv1.2",
                                                      "feature_map.backbone.layer5.0.conv2.1"],
            'feature_map.backbone.layer5.0.conv2.1': ["feature_map.backbone.layer5.0.conv2.0",
                                                      "feature_map.backbone.layer5.0.conv2.2"],
            'feature_map.backbone.layer5.0.conv2.2': ["feature_map.backbone.layer5.0.conv2.1",
                                                      "feature_map.backbone.layer5.1.conv1.0"],
            'feature_map.backbone.layer5.1.conv1.0': ["feature_map.backbone.layer5.0.conv2.2",
                                                      "feature_map.backbone.layer5.1.conv1.1"],
            'feature_map.backbone.layer5.1.conv1.1': ["feature_map.backbone.layer5.1.conv1.0",
                                                      "feature_map.backbone.layer5.1.conv1.2"],
            'feature_map.backbone.layer5.1.conv1.2': ["feature_map.backbone.layer5.1.conv1.1",
                                                      "feature_map.backbone.layer5.1.conv2.0"],
            'feature_map.backbone.layer5.1.conv2.0': ["feature_map.backbone.layer5.1.conv1.2",
                                                      "feature_map.backbone.layer5.1.conv2.1"],
            'feature_map.backbone.layer5.1.conv2.1': ["feature_map.backbone.layer5.1.conv2.0",
                                                      "feature_map.backbone.layer5.1.conv2.2"],
            'feature_map.backbone.layer5.1.conv2.2': ["feature_map.backbone.layer5.1.conv2.1",
                                                      "feature_map.backbone.layer5.2.conv1.0"],
            'feature_map.backbone.layer5.2.conv1.0': ["feature_map.backbone.layer5.1.conv2.2",
                                                      "feature_map.backbone.layer5.2.conv1.1"],
            'feature_map.backbone.layer5.2.conv1.1': ["feature_map.backbone.layer5.2.conv1.0",
                                                      "feature_map.backbone.layer5.2.conv1.2"],
            'feature_map.backbone.layer5.2.conv1.2': ["feature_map.backbone.layer5.2.conv1.1",
                                                      "feature_map.backbone.layer5.2.conv2.0"],
            'feature_map.backbone.layer5.2.conv2.0': ["feature_map.backbone.layer5.2.conv1.2",
                                                      "feature_map.backbone.layer5.2.conv2.1"],
            'feature_map.backbone.layer5.2.conv2.1': ["feature_map.backbone.layer5.2.conv2.0",
                                                      "feature_map.backbone.layer5.2.conv2.2"],
            'feature_map.backbone.layer5.2.conv2.2': ["feature_map.backbone.layer5.2.conv2.1",
                                                      "feature_map.backbone.layer5.3.conv1.0"],
            'feature_map.backbone.layer5.3.conv1.0': ["feature_map.backbone.layer5.2.conv2.2",
                                                      "feature_map.backbone.layer5.3.conv1.1"],
            'feature_map.backbone.layer5.3.conv1.1': ["feature_map.backbone.layer5.3.conv1.0",
                                                      "feature_map.backbone.layer5.3.conv1.2"],
            'feature_map.backbone.layer5.3.conv1.2': ["feature_map.backbone.layer5.3.conv1.1",
                                                      "feature_map.backbone.layer5.3.conv2.0"],
            'feature_map.backbone.layer5.3.conv2.0': ["feature_map.backbone.layer5.3.conv1.2",
                                                      "feature_map.backbone.layer5.3.conv2.1"],
            'feature_map.backbone.layer5.3.conv2.1': ["feature_map.backbone.layer5.3.conv2.0",
                                                      "feature_map.backbone.layer5.3.conv2.2"],
            'feature_map.backbone.layer5.3.conv2.2': ["feature_map.backbone.layer5.3.conv2.1",
                                                      "feature_map.backblock0.conv0.0"],  # layer5直接往下接第一个yoloblock
            'feature_map.backblock0.conv0.0': ["feature_map.backbone.layer5.3.conv2.2",
                                               "feature_map.backblock0.conv0.1"],
            'feature_map.backblock0.conv0.1': ["feature_map.backblock0.conv0.0", "feature_map.backblock0.conv0.2"],
            'feature_map.backblock0.conv0.2': ["feature_map.backblock0.conv0.1", "feature_map.backblock0.conv1.0"],
            'feature_map.backblock0.conv1.0': ["feature_map.backblock0.conv0.2", "feature_map.backblock0.conv1.1"],
            'feature_map.backblock0.conv1.1': ["feature_map.backblock0.conv1.0", "feature_map.backblock0.conv1.2"],
            'feature_map.backblock0.conv1.2': ["feature_map.backblock0.conv1.1", "feature_map.backblock0.conv2.0"],
            'feature_map.backblock0.conv2.0': ["feature_map.backblock0.conv1.2", "feature_map.backblock0.conv2.1"],
            'feature_map.backblock0.conv2.1': ["feature_map.backblock0.conv2.0", "feature_map.backblock0.conv2.2"],
            'feature_map.backblock0.conv2.2': ["feature_map.backblock0.conv2.1", "feature_map.backblock0.conv3.0"],
            'feature_map.backblock0.conv3.0': ["feature_map.backblock0.conv2.2", "feature_map.backblock0.conv3.1"],
            'feature_map.backblock0.conv3.1': ["feature_map.backblock0.conv3.0", "feature_map.backblock0.conv3.2"],
            'feature_map.backblock0.conv3.2': ["feature_map.backblock0.conv3.1", "feature_map.backblock0.conv4.0"],
            'feature_map.backblock0.conv4.0': ["feature_map.backblock0.conv3.2", "feature_map.backblock0.conv4.1"],
            'feature_map.backblock0.conv4.1': ["feature_map.backblock0.conv4.0", "feature_map.backblock0.conv4.2"],
            'feature_map.backblock0.conv4.2': ["feature_map.backblock0.conv4.1",
                                               ["feature_map.backblock0.conv5.0", "feature_map.conv1.0"]],
            # 第一个yoloblock的分支处，接第二个yoloblock前的cbr
            'feature_map.backblock0.conv5.0': ["feature_map.backblock0.conv4.2", "feature_map.backblock0.conv5.1"],
            'feature_map.backblock0.conv5.1': ["feature_map.backblock0.conv5.0", "feature_map.backblock0.conv5.2"],
            'feature_map.backblock0.conv5.2': ["feature_map.backblock0.conv5.1", "feature_map.backblock0.conv6"],
            'feature_map.backblock0.conv6': ["feature_map.backblock0.conv5.2", "detect_1.sigmoid"],
            'feature_map.conv1.0': ["feature_map.backblock0.conv4.2", "feature_map.conv1.1"],  # 从第一个yoloblock分支直接接下来
            'feature_map.conv1.1': ["feature_map.conv1.0", "feature_map.conv1.2"],
            'feature_map.conv1.2': ["feature_map.conv1.1", "feature_map.backblock1.conv0.0"],
            'feature_map.backblock1.conv0.0': [["feature_map.conv1.2", "feature_map.backbone.layer4.7.conv2.2"],
                                               "feature_map.backblock1.conv0.1"],  # 第二个yoloblock的开头
            'feature_map.backblock1.conv0.1': ["feature_map.backblock1.conv0.0", "feature_map.backblock1.conv0.2"],
            'feature_map.backblock1.conv0.2': ["feature_map.backblock1.conv0.1", "feature_map.backblock1.conv1.0"],
            'feature_map.backblock1.conv1.0': ["feature_map.backblock1.conv0.2", "feature_map.backblock1.conv1.1"],
            'feature_map.backblock1.conv1.1': ["feature_map.backblock1.conv1.0", "feature_map.backblock1.conv1.2"],
            'feature_map.backblock1.conv1.2': ["feature_map.backblock1.conv1.1", "feature_map.backblock1.conv2.0"],
            'feature_map.backblock1.conv2.0': ["feature_map.backblock1.conv1.2", "feature_map.backblock1.conv2.1"],
            'feature_map.backblock1.conv2.1': ["feature_map.backblock1.conv2.0", "feature_map.backblock1.conv2.2"],
            'feature_map.backblock1.conv2.2': ["feature_map.backblock1.conv2.1", "feature_map.backblock1.conv3.0"],
            'feature_map.backblock1.conv3.0': ["feature_map.backblock1.conv2.2", "feature_map.backblock1.conv3.1"],
            'feature_map.backblock1.conv3.1': ["feature_map.backblock1.conv3.0", "feature_map.backblock1.conv3.2"],
            'feature_map.backblock1.conv3.2': ["feature_map.backblock1.conv3.1", "feature_map.backblock1.conv4.0"],
            'feature_map.backblock1.conv4.0': ["feature_map.backblock1.conv3.2", "feature_map.backblock1.conv4.1"],
            'feature_map.backblock1.conv4.1': ["feature_map.backblock1.conv4.0", "feature_map.backblock1.conv4.2"],
            'feature_map.backblock1.conv4.2': ["feature_map.backblock1.conv4.1",
                                               ["feature_map.backblock1.conv5.0", "feature_map.conv2.0"]],
            # 第二个yoloblock的分支处，接第二个yoloblock前的cbr
            'feature_map.backblock1.conv5.0': ["feature_map.backblock1.conv4.2", "feature_map.backblock1.conv5.1"],
            'feature_map.backblock1.conv5.1': ["feature_map.backblock1.conv5.0", "feature_map.backblock1.conv5.2"],
            'feature_map.backblock1.conv5.2': ["feature_map.backblock1.conv5.1", "feature_map.backblock1.conv6"],
            'feature_map.backblock1.conv6': ["feature_map.backblock1.conv5.2", "detect_2.sigmoid"],
            'feature_map.conv2.0': ["feature_map.backblock1.conv4.2", "feature_map.conv2.1"],  # 从第二个yoloblock分支直接接下来
            'feature_map.conv2.1': ["feature_map.conv2.0", "feature_map.conv2.2"],
            'feature_map.conv2.2': ["feature_map.conv2.1", "feature_map.backblock2.conv0.0"],
            'feature_map.backblock2.conv0.0': [["feature_map.conv2.2", "feature_map.backbone.layer3.7.conv2.2"],
                                               "feature_map.backblock2.conv0.1"],  # 第三个yoloblock的开头
            'feature_map.backblock2.conv0.1': ["feature_map.backblock2.conv0.0", "feature_map.backblock2.conv0.2"],
            'feature_map.backblock2.conv0.2': ["feature_map.backblock2.conv0.1", "feature_map.backblock2.conv1.0"],
            'feature_map.backblock2.conv1.0': ["feature_map.backblock2.conv0.2", "feature_map.backblock2.conv1.1"],
            'feature_map.backblock2.conv1.1': ["feature_map.backblock2.conv1.0", "feature_map.backblock2.conv1.2"],
            'feature_map.backblock2.conv1.2': ["feature_map.backblock2.conv1.1", "feature_map.backblock2.conv2.0"],
            'feature_map.backblock2.conv2.0': ["feature_map.backblock2.conv1.2", "feature_map.backblock2.conv2.1"],
            'feature_map.backblock2.conv2.1': ["feature_map.backblock2.conv2.0", "feature_map.backblock2.conv2.2"],
            'feature_map.backblock2.conv2.2': ["feature_map.backblock2.conv2.1", "feature_map.backblock2.conv3.0"],
            'feature_map.backblock2.conv3.0': ["feature_map.backblock2.conv2.2", "feature_map.backblock2.conv3.1"],
            'feature_map.backblock2.conv3.1': ["feature_map.backblock2.conv3.0", "feature_map.backblock2.conv3.2"],
            'feature_map.backblock2.conv3.2': ["feature_map.backblock2.conv3.1", "feature_map.backblock2.conv4.0"],
            'feature_map.backblock2.conv4.0': ["feature_map.backblock2.conv3.2", "feature_map.backblock2.conv4.1"],
            'feature_map.backblock2.conv4.1': ["feature_map.backblock2.conv4.0", "feature_map.backblock2.conv4.2"],
            'feature_map.backblock2.conv4.2': ["feature_map.backblock2.conv4.1", "feature_map.backblock2.conv5.0"],
            'feature_map.backblock2.conv5.0': ["feature_map.backblock2.conv4.2", "feature_map.backblock2.conv5.1"],
            'feature_map.backblock2.conv5.1': ["feature_map.backblock2.conv5.0", "feature_map.backblock2.conv5.2"],
            'feature_map.backblock2.conv5.2': ["feature_map.backblock2.conv5.1", "feature_map.backblock2.conv6"],
            'feature_map.backblock2.conv6': ["feature_map.backblock2.conv5.2", "detect_3.sigmoid"],
            'detect_1.sigmoid': ["feature_map.backblock0.conv6", "OUTPUT1"],  # 从第一个yoloblock分支接出来
            'detect_2.sigmoid': ["feature_map.backblock1.conv6", "OUTPUT2"],  # 从第二个yoloblock分支接出来
            'detect_3.sigmoid': ["feature_map.backblock2.conv6", "OUTPUT3"],
        }



    def get_layers(self, layer_name):
        if layer_name not in self.layer_names.keys():
            return False
        return self.layer_names[layer_name]


    def set_origin_layers(self,layer_name,new_layer):
        if 'feature_map' == layer_name:
            self.feature_map= new_layer
            self.origin_layer_names["feature_map"]=new_layer
        elif 'feature_map.backbone' == layer_name:
            self.feature_map.backbone= new_layer
            self.origin_layer_names["feature_map.backbone"]=new_layer
        elif 'feature_map.backbone.conv0' == layer_name:
            self.feature_map.backbone.conv0= new_layer
            self.origin_layer_names["feature_map.backbone.conv0"]=new_layer
        elif 'feature_map.backbone.conv0.0' == layer_name:
            self.feature_map.backbone.conv0[0]= new_layer
            self.origin_layer_names["feature_map.backbone.conv0.0"]=new_layer
        elif 'feature_map.backbone.conv0.1' == layer_name:
            self.feature_map.backbone.conv0[1]= new_layer
            self.origin_layer_names["feature_map.backbone.conv0.1"]=new_layer
        elif 'feature_map.backbone.conv0.2' == layer_name:
            self.feature_map.backbone.conv0[2]= new_layer
            self.origin_layer_names["feature_map.backbone.conv0.2"]=new_layer
        elif 'feature_map.backbone.conv1' == layer_name:
            self.feature_map.backbone.conv1= new_layer
            self.origin_layer_names["feature_map.backbone.conv1"]=new_layer
        elif 'feature_map.backbone.conv1.0' == layer_name:
            self.feature_map.backbone.conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.conv1.0"]=new_layer
        elif 'feature_map.backbone.conv1.1' == layer_name:
            self.feature_map.backbone.conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.conv1.1"]=new_layer
        elif 'feature_map.backbone.conv1.2' == layer_name:
            self.feature_map.backbone.conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer1' == layer_name:
            self.feature_map.backbone.layer1= new_layer
            self.origin_layer_names["feature_map.backbone.layer1"]=new_layer
        elif 'feature_map.backbone.layer1.0' == layer_name:
            self.feature_map.backbone.layer1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv1' == layer_name:
            self.feature_map.backbone.layer1[0].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer1[0].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer1[0].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer1[0].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv2' == layer_name:
            self.feature_map.backbone.layer1[0].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer1[0].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer1[0].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer1[0].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.conv2' == layer_name:
            self.feature_map.backbone.conv2= new_layer
            self.origin_layer_names["feature_map.backbone.conv2"]=new_layer
        elif 'feature_map.backbone.conv2.0' == layer_name:
            self.feature_map.backbone.conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.conv2.0"]=new_layer
        elif 'feature_map.backbone.conv2.1' == layer_name:
            self.feature_map.backbone.conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.conv2.1"]=new_layer
        elif 'feature_map.backbone.conv2.2' == layer_name:
            self.feature_map.backbone.conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer2' == layer_name:
            self.feature_map.backbone.layer2= new_layer
            self.origin_layer_names["feature_map.backbone.layer2"]=new_layer
        elif 'feature_map.backbone.layer2.0' == layer_name:
            self.feature_map.backbone.layer2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1' == layer_name:
            self.feature_map.backbone.layer2[0].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2' == layer_name:
            self.feature_map.backbone.layer2[0].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer2.1' == layer_name:
            self.feature_map.backbone.layer2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1' == layer_name:
            self.feature_map.backbone.layer2[1].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2' == layer_name:
            self.feature_map.backbone.layer2[1].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.conv3' == layer_name:
            self.feature_map.backbone.conv3= new_layer
            self.origin_layer_names["feature_map.backbone.conv3"]=new_layer
        elif 'feature_map.backbone.conv3.0' == layer_name:
            self.feature_map.backbone.conv3[0]= new_layer
            self.origin_layer_names["feature_map.backbone.conv3.0"]=new_layer
        elif 'feature_map.backbone.conv3.1' == layer_name:
            self.feature_map.backbone.conv3[1]= new_layer
            self.origin_layer_names["feature_map.backbone.conv3.1"]=new_layer
        elif 'feature_map.backbone.conv3.2' == layer_name:
            self.feature_map.backbone.conv3[2]= new_layer
            self.origin_layer_names["feature_map.backbone.conv3.2"]=new_layer
        elif 'feature_map.backbone.layer3' == layer_name:
            self.feature_map.backbone.layer3= new_layer
            self.origin_layer_names["feature_map.backbone.layer3"]=new_layer
        elif 'feature_map.backbone.layer3.0' == layer_name:
            self.feature_map.backbone.layer3[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1' == layer_name:
            self.feature_map.backbone.layer3[0].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2' == layer_name:
            self.feature_map.backbone.layer3[0].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.1' == layer_name:
            self.feature_map.backbone.layer3[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1' == layer_name:
            self.feature_map.backbone.layer3[1].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2' == layer_name:
            self.feature_map.backbone.layer3[1].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.2' == layer_name:
            self.feature_map.backbone.layer3[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1' == layer_name:
            self.feature_map.backbone.layer3[2].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2' == layer_name:
            self.feature_map.backbone.layer3[2].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.3' == layer_name:
            self.feature_map.backbone.layer3[3]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1' == layer_name:
            self.feature_map.backbone.layer3[3].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2' == layer_name:
            self.feature_map.backbone.layer3[3].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.4' == layer_name:
            self.feature_map.backbone.layer3[4]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1' == layer_name:
            self.feature_map.backbone.layer3[4].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2' == layer_name:
            self.feature_map.backbone.layer3[4].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.5' == layer_name:
            self.feature_map.backbone.layer3[5]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1' == layer_name:
            self.feature_map.backbone.layer3[5].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2' == layer_name:
            self.feature_map.backbone.layer3[5].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.6' == layer_name:
            self.feature_map.backbone.layer3[6]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1' == layer_name:
            self.feature_map.backbone.layer3[6].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2' == layer_name:
            self.feature_map.backbone.layer3[6].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.7' == layer_name:
            self.feature_map.backbone.layer3[7]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1' == layer_name:
            self.feature_map.backbone.layer3[7].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2' == layer_name:
            self.feature_map.backbone.layer3[7].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.2"]=new_layer
        elif 'feature_map.backbone.conv4' == layer_name:
            self.feature_map.backbone.conv4= new_layer
            self.origin_layer_names["feature_map.backbone.conv4"]=new_layer
        elif 'feature_map.backbone.conv4.0' == layer_name:
            self.feature_map.backbone.conv4[0]= new_layer
            self.origin_layer_names["feature_map.backbone.conv4.0"]=new_layer
        elif 'feature_map.backbone.conv4.1' == layer_name:
            self.feature_map.backbone.conv4[1]= new_layer
            self.origin_layer_names["feature_map.backbone.conv4.1"]=new_layer
        elif 'feature_map.backbone.conv4.2' == layer_name:
            self.feature_map.backbone.conv4[2]= new_layer
            self.origin_layer_names["feature_map.backbone.conv4.2"]=new_layer
        elif 'feature_map.backbone.layer4' == layer_name:
            self.feature_map.backbone.layer4= new_layer
            self.origin_layer_names["feature_map.backbone.layer4"]=new_layer
        elif 'feature_map.backbone.layer4.0' == layer_name:
            self.feature_map.backbone.layer4[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1' == layer_name:
            self.feature_map.backbone.layer4[0].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2' == layer_name:
            self.feature_map.backbone.layer4[0].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.1' == layer_name:
            self.feature_map.backbone.layer4[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1' == layer_name:
            self.feature_map.backbone.layer4[1].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2' == layer_name:
            self.feature_map.backbone.layer4[1].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.2' == layer_name:
            self.feature_map.backbone.layer4[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1' == layer_name:
            self.feature_map.backbone.layer4[2].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2' == layer_name:
            self.feature_map.backbone.layer4[2].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.3' == layer_name:
            self.feature_map.backbone.layer4[3]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1' == layer_name:
            self.feature_map.backbone.layer4[3].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2' == layer_name:
            self.feature_map.backbone.layer4[3].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.4' == layer_name:
            self.feature_map.backbone.layer4[4]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1' == layer_name:
            self.feature_map.backbone.layer4[4].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2' == layer_name:
            self.feature_map.backbone.layer4[4].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.5' == layer_name:
            self.feature_map.backbone.layer4[5]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1' == layer_name:
            self.feature_map.backbone.layer4[5].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2' == layer_name:
            self.feature_map.backbone.layer4[5].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.6' == layer_name:
            self.feature_map.backbone.layer4[6]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1' == layer_name:
            self.feature_map.backbone.layer4[6].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2' == layer_name:
            self.feature_map.backbone.layer4[6].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.7' == layer_name:
            self.feature_map.backbone.layer4[7]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1' == layer_name:
            self.feature_map.backbone.layer4[7].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2' == layer_name:
            self.feature_map.backbone.layer4[7].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.2"]=new_layer
        elif 'feature_map.backbone.conv5' == layer_name:
            self.feature_map.backbone.conv5= new_layer
            self.origin_layer_names["feature_map.backbone.conv5"]=new_layer
        elif 'feature_map.backbone.conv5.0' == layer_name:
            self.feature_map.backbone.conv5[0]= new_layer
            self.origin_layer_names["feature_map.backbone.conv5.0"]=new_layer
        elif 'feature_map.backbone.conv5.1' == layer_name:
            self.feature_map.backbone.conv5[1]= new_layer
            self.origin_layer_names["feature_map.backbone.conv5.1"]=new_layer
        elif 'feature_map.backbone.conv5.2' == layer_name:
            self.feature_map.backbone.conv5[2]= new_layer
            self.origin_layer_names["feature_map.backbone.conv5.2"]=new_layer
        elif 'feature_map.backbone.layer5' == layer_name:
            self.feature_map.backbone.layer5= new_layer
            self.origin_layer_names["feature_map.backbone.layer5"]=new_layer
        elif 'feature_map.backbone.layer5.0' == layer_name:
            self.feature_map.backbone.layer5[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1' == layer_name:
            self.feature_map.backbone.layer5[0].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2' == layer_name:
            self.feature_map.backbone.layer5[0].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer5.1' == layer_name:
            self.feature_map.backbone.layer5[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1' == layer_name:
            self.feature_map.backbone.layer5[1].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2' == layer_name:
            self.feature_map.backbone.layer5[1].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer5.2' == layer_name:
            self.feature_map.backbone.layer5[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1' == layer_name:
            self.feature_map.backbone.layer5[2].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2' == layer_name:
            self.feature_map.backbone.layer5[2].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer5.3' == layer_name:
            self.feature_map.backbone.layer5[3]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1' == layer_name:
            self.feature_map.backbone.layer5[3].conv1= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2' == layer_name:
            self.feature_map.backbone.layer5[3].conv2= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[0]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[1]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[2]= new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.2"]=new_layer
        elif 'feature_map.backblock0' == layer_name:
            self.feature_map.backblock0= new_layer
            self.origin_layer_names["feature_map.backblock0"]=new_layer
        elif 'feature_map.backblock0.conv0' == layer_name:
            self.feature_map.backblock0.conv0= new_layer
            self.origin_layer_names["feature_map.backblock0.conv0"]=new_layer
        elif 'feature_map.backblock0.conv0.0' == layer_name:
            self.feature_map.backblock0.conv0[0]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.0"]=new_layer
        elif 'feature_map.backblock0.conv0.1' == layer_name:
            self.feature_map.backblock0.conv0[1]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.1"]=new_layer
        elif 'feature_map.backblock0.conv0.2' == layer_name:
            self.feature_map.backblock0.conv0[2]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.2"]=new_layer
        elif 'feature_map.backblock0.conv1' == layer_name:
            self.feature_map.backblock0.conv1= new_layer
            self.origin_layer_names["feature_map.backblock0.conv1"]=new_layer
        elif 'feature_map.backblock0.conv1.0' == layer_name:
            self.feature_map.backblock0.conv1[0]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.0"]=new_layer
        elif 'feature_map.backblock0.conv1.1' == layer_name:
            self.feature_map.backblock0.conv1[1]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.1"]=new_layer
        elif 'feature_map.backblock0.conv1.2' == layer_name:
            self.feature_map.backblock0.conv1[2]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.2"]=new_layer
        elif 'feature_map.backblock0.conv2' == layer_name:
            self.feature_map.backblock0.conv2= new_layer
            self.origin_layer_names["feature_map.backblock0.conv2"]=new_layer
        elif 'feature_map.backblock0.conv2.0' == layer_name:
            self.feature_map.backblock0.conv2[0]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.0"]=new_layer
        elif 'feature_map.backblock0.conv2.1' == layer_name:
            self.feature_map.backblock0.conv2[1]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.1"]=new_layer
        elif 'feature_map.backblock0.conv2.2' == layer_name:
            self.feature_map.backblock0.conv2[2]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.2"]=new_layer
        elif 'feature_map.backblock0.conv3' == layer_name:
            self.feature_map.backblock0.conv3= new_layer
            self.origin_layer_names["feature_map.backblock0.conv3"]=new_layer
        elif 'feature_map.backblock0.conv3.0' == layer_name:
            self.feature_map.backblock0.conv3[0]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.0"]=new_layer
        elif 'feature_map.backblock0.conv3.1' == layer_name:
            self.feature_map.backblock0.conv3[1]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.1"]=new_layer
        elif 'feature_map.backblock0.conv3.2' == layer_name:
            self.feature_map.backblock0.conv3[2]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.2"]=new_layer
        elif 'feature_map.backblock0.conv4' == layer_name:
            self.feature_map.backblock0.conv4= new_layer
            self.origin_layer_names["feature_map.backblock0.conv4"]=new_layer
        elif 'feature_map.backblock0.conv4.0' == layer_name:
            self.feature_map.backblock0.conv4[0]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.0"]=new_layer
        elif 'feature_map.backblock0.conv4.1' == layer_name:
            self.feature_map.backblock0.conv4[1]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.1"]=new_layer
        elif 'feature_map.backblock0.conv4.2' == layer_name:
            self.feature_map.backblock0.conv4[2]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.2"]=new_layer
        elif 'feature_map.backblock0.conv5' == layer_name:
            self.feature_map.backblock0.conv5= new_layer
            self.origin_layer_names["feature_map.backblock0.conv5"]=new_layer
        elif 'feature_map.backblock0.conv5.0' == layer_name:
            self.feature_map.backblock0.conv5[0]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv5.0"]=new_layer
        elif 'feature_map.backblock0.conv5.1' == layer_name:
            self.feature_map.backblock0.conv5[1]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv5.1"]=new_layer
        elif 'feature_map.backblock0.conv5.2' == layer_name:
            self.feature_map.backblock0.conv5[2]= new_layer
            self.origin_layer_names["feature_map.backblock0.conv5.2"]=new_layer
        elif 'feature_map.backblock0.conv6' == layer_name:
            self.feature_map.backblock0.conv6= new_layer
            self.origin_layer_names["feature_map.backblock0.conv6"]=new_layer
        elif 'feature_map.conv1' == layer_name:
            self.feature_map.conv1= new_layer
            self.origin_layer_names["feature_map.conv1"]=new_layer
        elif 'feature_map.conv1.0' == layer_name:
            self.feature_map.conv1[0]= new_layer
            self.origin_layer_names["feature_map.conv1.0"]=new_layer
        elif 'feature_map.conv1.1' == layer_name:
            self.feature_map.conv1[1]= new_layer
            self.origin_layer_names["feature_map.conv1.1"]=new_layer
        elif 'feature_map.conv1.2' == layer_name:
            self.feature_map.conv1[2]= new_layer
            self.origin_layer_names["feature_map.conv1.2"]=new_layer
        elif 'feature_map.backblock1' == layer_name:
            self.feature_map.backblock1= new_layer
            self.origin_layer_names["feature_map.backblock1"]=new_layer
        elif 'feature_map.backblock1.conv0' == layer_name:
            self.feature_map.backblock1.conv0= new_layer
            self.origin_layer_names["feature_map.backblock1.conv0"]=new_layer
        elif 'feature_map.backblock1.conv0.0' == layer_name:
            self.feature_map.backblock1.conv0[0]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.0"]=new_layer
        elif 'feature_map.backblock1.conv0.1' == layer_name:
            self.feature_map.backblock1.conv0[1]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.1"]=new_layer
        elif 'feature_map.backblock1.conv0.2' == layer_name:
            self.feature_map.backblock1.conv0[2]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.2"]=new_layer
        elif 'feature_map.backblock1.conv1' == layer_name:
            self.feature_map.backblock1.conv1= new_layer
            self.origin_layer_names["feature_map.backblock1.conv1"]=new_layer
        elif 'feature_map.backblock1.conv1.0' == layer_name:
            self.feature_map.backblock1.conv1[0]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.0"]=new_layer
        elif 'feature_map.backblock1.conv1.1' == layer_name:
            self.feature_map.backblock1.conv1[1]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.1"]=new_layer
        elif 'feature_map.backblock1.conv1.2' == layer_name:
            self.feature_map.backblock1.conv1[2]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.2"]=new_layer
        elif 'feature_map.backblock1.conv2' == layer_name:
            self.feature_map.backblock1.conv2= new_layer
            self.origin_layer_names["feature_map.backblock1.conv2"]=new_layer
        elif 'feature_map.backblock1.conv2.0' == layer_name:
            self.feature_map.backblock1.conv2[0]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.0"]=new_layer
        elif 'feature_map.backblock1.conv2.1' == layer_name:
            self.feature_map.backblock1.conv2[1]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.1"]=new_layer
        elif 'feature_map.backblock1.conv2.2' == layer_name:
            self.feature_map.backblock1.conv2[2]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.2"]=new_layer
        elif 'feature_map.backblock1.conv3' == layer_name:
            self.feature_map.backblock1.conv3= new_layer
            self.origin_layer_names["feature_map.backblock1.conv3"]=new_layer
        elif 'feature_map.backblock1.conv3.0' == layer_name:
            self.feature_map.backblock1.conv3[0]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.0"]=new_layer
        elif 'feature_map.backblock1.conv3.1' == layer_name:
            self.feature_map.backblock1.conv3[1]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.1"]=new_layer
        elif 'feature_map.backblock1.conv3.2' == layer_name:
            self.feature_map.backblock1.conv3[2]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.2"]=new_layer
        elif 'feature_map.backblock1.conv4' == layer_name:
            self.feature_map.backblock1.conv4= new_layer
            self.origin_layer_names["feature_map.backblock1.conv4"]=new_layer
        elif 'feature_map.backblock1.conv4.0' == layer_name:
            self.feature_map.backblock1.conv4[0]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.0"]=new_layer
        elif 'feature_map.backblock1.conv4.1' == layer_name:
            self.feature_map.backblock1.conv4[1]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.1"]=new_layer
        elif 'feature_map.backblock1.conv4.2' == layer_name:
            self.feature_map.backblock1.conv4[2]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.2"]=new_layer
        elif 'feature_map.backblock1.conv5' == layer_name:
            self.feature_map.backblock1.conv5= new_layer
            self.origin_layer_names["feature_map.backblock1.conv5"]=new_layer
        elif 'feature_map.backblock1.conv5.0' == layer_name:
            self.feature_map.backblock1.conv5[0]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.0"]=new_layer
        elif 'feature_map.backblock1.conv5.1' == layer_name:
            self.feature_map.backblock1.conv5[1]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.1"]=new_layer
        elif 'feature_map.backblock1.conv5.2' == layer_name:
            self.feature_map.backblock1.conv5[2]= new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.2"]=new_layer
        elif 'feature_map.backblock1.conv6' == layer_name:
            self.feature_map.backblock1.conv6= new_layer
            self.origin_layer_names["feature_map.backblock1.conv6"]=new_layer
        elif 'feature_map.conv2' == layer_name:
            self.feature_map.conv2= new_layer
            self.origin_layer_names["feature_map.conv2"]=new_layer
        elif 'feature_map.conv2.0' == layer_name:
            self.feature_map.conv2[0]= new_layer
            self.origin_layer_names["feature_map.conv2.0"]=new_layer
        elif 'feature_map.conv2.1' == layer_name:
            self.feature_map.conv2[1]= new_layer
            self.origin_layer_names["feature_map.conv2.1"]=new_layer
        elif 'feature_map.conv2.2' == layer_name:
            self.feature_map.conv2[2]= new_layer
            self.origin_layer_names["feature_map.conv2.2"]=new_layer
        elif 'feature_map.backblock2' == layer_name:
            self.feature_map.backblock2= new_layer
            self.origin_layer_names["feature_map.backblock2"]=new_layer
        elif 'feature_map.backblock2.conv0' == layer_name:
            self.feature_map.backblock2.conv0= new_layer
            self.origin_layer_names["feature_map.backblock2.conv0"]=new_layer
        elif 'feature_map.backblock2.conv0.0' == layer_name:
            self.feature_map.backblock2.conv0[0]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.0"]=new_layer
        elif 'feature_map.backblock2.conv0.1' == layer_name:
            self.feature_map.backblock2.conv0[1]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.1"]=new_layer
        elif 'feature_map.backblock2.conv0.2' == layer_name:
            self.feature_map.backblock2.conv0[2]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.2"]=new_layer
        elif 'feature_map.backblock2.conv1' == layer_name:
            self.feature_map.backblock2.conv1= new_layer
            self.origin_layer_names["feature_map.backblock2.conv1"]=new_layer
        elif 'feature_map.backblock2.conv1.0' == layer_name:
            self.feature_map.backblock2.conv1[0]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.0"]=new_layer
        elif 'feature_map.backblock2.conv1.1' == layer_name:
            self.feature_map.backblock2.conv1[1]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.1"]=new_layer
        elif 'feature_map.backblock2.conv1.2' == layer_name:
            self.feature_map.backblock2.conv1[2]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.2"]=new_layer
        elif 'feature_map.backblock2.conv2' == layer_name:
            self.feature_map.backblock2.conv2= new_layer
            self.origin_layer_names["feature_map.backblock2.conv2"]=new_layer
        elif 'feature_map.backblock2.conv2.0' == layer_name:
            self.feature_map.backblock2.conv2[0]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.0"]=new_layer
        elif 'feature_map.backblock2.conv2.1' == layer_name:
            self.feature_map.backblock2.conv2[1]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.1"]=new_layer
        elif 'feature_map.backblock2.conv2.2' == layer_name:
            self.feature_map.backblock2.conv2[2]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.2"]=new_layer
        elif 'feature_map.backblock2.conv3' == layer_name:
            self.feature_map.backblock2.conv3= new_layer
            self.origin_layer_names["feature_map.backblock2.conv3"]=new_layer
        elif 'feature_map.backblock2.conv3.0' == layer_name:
            self.feature_map.backblock2.conv3[0]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.0"]=new_layer
        elif 'feature_map.backblock2.conv3.1' == layer_name:
            self.feature_map.backblock2.conv3[1]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.1"]=new_layer
        elif 'feature_map.backblock2.conv3.2' == layer_name:
            self.feature_map.backblock2.conv3[2]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.2"]=new_layer
        elif 'feature_map.backblock2.conv4' == layer_name:
            self.feature_map.backblock2.conv4= new_layer
            self.origin_layer_names["feature_map.backblock2.conv4"]=new_layer
        elif 'feature_map.backblock2.conv4.0' == layer_name:
            self.feature_map.backblock2.conv4[0]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.0"]=new_layer
        elif 'feature_map.backblock2.conv4.1' == layer_name:
            self.feature_map.backblock2.conv4[1]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.1"]=new_layer
        elif 'feature_map.backblock2.conv4.2' == layer_name:
            self.feature_map.backblock2.conv4[2]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.2"]=new_layer
        elif 'feature_map.backblock2.conv5' == layer_name:
            self.feature_map.backblock2.conv5= new_layer
            self.origin_layer_names["feature_map.backblock2.conv5"]=new_layer
        elif 'feature_map.backblock2.conv5.0' == layer_name:
            self.feature_map.backblock2.conv5[0]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.0"]=new_layer
        elif 'feature_map.backblock2.conv5.1' == layer_name:
            self.feature_map.backblock2.conv5[1]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.1"]=new_layer
        elif 'feature_map.backblock2.conv5.2' == layer_name:
            self.feature_map.backblock2.conv5[2]= new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.2"]=new_layer
        elif 'feature_map.backblock2.conv6' == layer_name:
            self.feature_map.backblock2.conv6= new_layer
            self.origin_layer_names["feature_map.backblock2.conv6"]=new_layer
        elif 'detect_1' == layer_name:
            self.detect_1= new_layer
            self.origin_layer_names["detect_1"]=new_layer
        elif 'detect_1.sigmoid' == layer_name:
            self.detect_1.sigmoid= new_layer
            self.origin_layer_names["detect_1.sigmoid"]=new_layer
        elif 'detect_2' == layer_name:
            self.detect_2= new_layer
            self.origin_layer_names["detect_2"]=new_layer
        elif 'detect_2.sigmoid' == layer_name:
            self.detect_2.sigmoid= new_layer
            self.origin_layer_names["detect_2.sigmoid"]=new_layer
        elif 'detect_3' == layer_name:
            self.detect_3= new_layer
            self.origin_layer_names["detect_3"]=new_layer
        elif 'detect_3.sigmoid' == layer_name:
            self.detect_3.sigmoid= new_layer
            self.origin_layer_names["detect_3.sigmoid"]=new_layer




    def set_layers(self,layer_name,new_layer):
        if 'feature_map' == layer_name:
            self.feature_map= new_layer
            self.layer_names["feature_map"]=new_layer
            self.origin_layer_names["feature_map"]=new_layer
        elif 'feature_map.backbone' == layer_name:
            self.feature_map.backbone= new_layer
            self.layer_names["feature_map.backbone"]=new_layer
            self.origin_layer_names["feature_map.backbone"]=new_layer
        elif 'feature_map.backbone.conv0' == layer_name:
            self.feature_map.backbone.conv0= new_layer
            self.layer_names["feature_map.backbone.conv0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv0"]=new_layer
        elif 'feature_map.backbone.conv0.0' == layer_name:
            self.feature_map.backbone.conv0[0]= new_layer
            self.layer_names["feature_map.backbone.conv0.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv0.0"]=new_layer
        elif 'feature_map.backbone.conv0.1' == layer_name:
            self.feature_map.backbone.conv0[1]= new_layer
            self.layer_names["feature_map.backbone.conv0.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv0.1"]=new_layer
        elif 'feature_map.backbone.conv0.2' == layer_name:
            self.feature_map.backbone.conv0[2]= new_layer
            self.layer_names["feature_map.backbone.conv0.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv0.2"]=new_layer
        elif 'feature_map.backbone.conv1' == layer_name:
            self.feature_map.backbone.conv1= new_layer
            self.layer_names["feature_map.backbone.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv1"]=new_layer
        elif 'feature_map.backbone.conv1.0' == layer_name:
            self.feature_map.backbone.conv1[0]= new_layer
            self.layer_names["feature_map.backbone.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv1.0"]=new_layer
        elif 'feature_map.backbone.conv1.1' == layer_name:
            self.feature_map.backbone.conv1[1]= new_layer
            self.layer_names["feature_map.backbone.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv1.1"]=new_layer
        elif 'feature_map.backbone.conv1.2' == layer_name:
            self.feature_map.backbone.conv1[2]= new_layer
            self.layer_names["feature_map.backbone.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer1' == layer_name:
            self.feature_map.backbone.layer1= new_layer
            self.layer_names["feature_map.backbone.layer1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer1"]=new_layer
        elif 'feature_map.backbone.layer1.0' == layer_name:
            self.feature_map.backbone.layer1[0]= new_layer
            self.layer_names["feature_map.backbone.layer1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv1' == layer_name:
            self.feature_map.backbone.layer1[0].conv1= new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer1[0].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer1[0].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer1[0].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv2' == layer_name:
            self.feature_map.backbone.layer1[0].conv2= new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer1[0].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer1[0].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer1.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer1[0].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer1.0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer1.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.conv2' == layer_name:
            self.feature_map.backbone.conv2= new_layer
            self.layer_names["feature_map.backbone.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv2"]=new_layer
        elif 'feature_map.backbone.conv2.0' == layer_name:
            self.feature_map.backbone.conv2[0]= new_layer
            self.layer_names["feature_map.backbone.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv2.0"]=new_layer
        elif 'feature_map.backbone.conv2.1' == layer_name:
            self.feature_map.backbone.conv2[1]= new_layer
            self.layer_names["feature_map.backbone.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv2.1"]=new_layer
        elif 'feature_map.backbone.conv2.2' == layer_name:
            self.feature_map.backbone.conv2[2]= new_layer
            self.layer_names["feature_map.backbone.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer2' == layer_name:
            self.feature_map.backbone.layer2= new_layer
            self.layer_names["feature_map.backbone.layer2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2"]=new_layer
        elif 'feature_map.backbone.layer2.0' == layer_name:
            self.feature_map.backbone.layer2[0]= new_layer
            self.layer_names["feature_map.backbone.layer2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1' == layer_name:
            self.feature_map.backbone.layer2[0].conv1= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer2[0].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2' == layer_name:
            self.feature_map.backbone.layer2[0].conv2= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer2.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer2[0].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer2.0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer2.1' == layer_name:
            self.feature_map.backbone.layer2[1]= new_layer
            self.layer_names["feature_map.backbone.layer2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1' == layer_name:
            self.feature_map.backbone.layer2[1].conv1= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer2[1].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2' == layer_name:
            self.feature_map.backbone.layer2[1].conv2= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer2.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer2[1].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer2.1.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer2.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.conv3' == layer_name:
            self.feature_map.backbone.conv3= new_layer
            self.layer_names["feature_map.backbone.conv3"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv3"]=new_layer
        elif 'feature_map.backbone.conv3.0' == layer_name:
            self.feature_map.backbone.conv3[0]= new_layer
            self.layer_names["feature_map.backbone.conv3.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv3.0"]=new_layer
        elif 'feature_map.backbone.conv3.1' == layer_name:
            self.feature_map.backbone.conv3[1]= new_layer
            self.layer_names["feature_map.backbone.conv3.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv3.1"]=new_layer
        elif 'feature_map.backbone.conv3.2' == layer_name:
            self.feature_map.backbone.conv3[2]= new_layer
            self.layer_names["feature_map.backbone.conv3.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv3.2"]=new_layer
        elif 'feature_map.backbone.layer3' == layer_name:
            self.feature_map.backbone.layer3= new_layer
            self.layer_names["feature_map.backbone.layer3"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3"]=new_layer
        elif 'feature_map.backbone.layer3.0' == layer_name:
            self.feature_map.backbone.layer3[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1' == layer_name:
            self.feature_map.backbone.layer3[0].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[0].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2' == layer_name:
            self.feature_map.backbone.layer3[0].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[0].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.1' == layer_name:
            self.feature_map.backbone.layer3[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1' == layer_name:
            self.feature_map.backbone.layer3[1].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[1].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2' == layer_name:
            self.feature_map.backbone.layer3[1].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[1].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.1.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.2' == layer_name:
            self.feature_map.backbone.layer3[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1' == layer_name:
            self.feature_map.backbone.layer3[2].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[2].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2' == layer_name:
            self.feature_map.backbone.layer3[2].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[2].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.2.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.2.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.3' == layer_name:
            self.feature_map.backbone.layer3[3]= new_layer
            self.layer_names["feature_map.backbone.layer3.3"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1' == layer_name:
            self.feature_map.backbone.layer3[3].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[3].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2' == layer_name:
            self.feature_map.backbone.layer3[3].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[3].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.3.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.3.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.4' == layer_name:
            self.feature_map.backbone.layer3[4]= new_layer
            self.layer_names["feature_map.backbone.layer3.4"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1' == layer_name:
            self.feature_map.backbone.layer3[4].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[4].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2' == layer_name:
            self.feature_map.backbone.layer3[4].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.4.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[4].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.4.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.4.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.5' == layer_name:
            self.feature_map.backbone.layer3[5]= new_layer
            self.layer_names["feature_map.backbone.layer3.5"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1' == layer_name:
            self.feature_map.backbone.layer3[5].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[5].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2' == layer_name:
            self.feature_map.backbone.layer3[5].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.5.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[5].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.5.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.5.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.6' == layer_name:
            self.feature_map.backbone.layer3[6]= new_layer
            self.layer_names["feature_map.backbone.layer3.6"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1' == layer_name:
            self.feature_map.backbone.layer3[6].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[6].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2' == layer_name:
            self.feature_map.backbone.layer3[6].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.6.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[6].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.6.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.6.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer3.7' == layer_name:
            self.feature_map.backbone.layer3[7]= new_layer
            self.layer_names["feature_map.backbone.layer3.7"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1' == layer_name:
            self.feature_map.backbone.layer3[7].conv1= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1.0' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1.1' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv1.2' == layer_name:
            self.feature_map.backbone.layer3[7].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2' == layer_name:
            self.feature_map.backbone.layer3[7].conv2= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2.0' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2.1' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer3.7.conv2.2' == layer_name:
            self.feature_map.backbone.layer3[7].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer3.7.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer3.7.conv2.2"]=new_layer
        elif 'feature_map.backbone.conv4' == layer_name:
            self.feature_map.backbone.conv4= new_layer
            self.layer_names["feature_map.backbone.conv4"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv4"]=new_layer
        elif 'feature_map.backbone.conv4.0' == layer_name:
            self.feature_map.backbone.conv4[0]= new_layer
            self.layer_names["feature_map.backbone.conv4.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv4.0"]=new_layer
        elif 'feature_map.backbone.conv4.1' == layer_name:
            self.feature_map.backbone.conv4[1]= new_layer
            self.layer_names["feature_map.backbone.conv4.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv4.1"]=new_layer
        elif 'feature_map.backbone.conv4.2' == layer_name:
            self.feature_map.backbone.conv4[2]= new_layer
            self.layer_names["feature_map.backbone.conv4.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv4.2"]=new_layer
        elif 'feature_map.backbone.layer4' == layer_name:
            self.feature_map.backbone.layer4= new_layer
            self.layer_names["feature_map.backbone.layer4"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4"]=new_layer
        elif 'feature_map.backbone.layer4.0' == layer_name:
            self.feature_map.backbone.layer4[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1' == layer_name:
            self.feature_map.backbone.layer4[0].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[0].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2' == layer_name:
            self.feature_map.backbone.layer4[0].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[0].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.1' == layer_name:
            self.feature_map.backbone.layer4[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1' == layer_name:
            self.feature_map.backbone.layer4[1].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[1].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2' == layer_name:
            self.feature_map.backbone.layer4[1].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[1].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.1.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.2' == layer_name:
            self.feature_map.backbone.layer4[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1' == layer_name:
            self.feature_map.backbone.layer4[2].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[2].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2' == layer_name:
            self.feature_map.backbone.layer4[2].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[2].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.2.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.2.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.3' == layer_name:
            self.feature_map.backbone.layer4[3]= new_layer
            self.layer_names["feature_map.backbone.layer4.3"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1' == layer_name:
            self.feature_map.backbone.layer4[3].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[3].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2' == layer_name:
            self.feature_map.backbone.layer4[3].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[3].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.3.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.3.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.4' == layer_name:
            self.feature_map.backbone.layer4[4]= new_layer
            self.layer_names["feature_map.backbone.layer4.4"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1' == layer_name:
            self.feature_map.backbone.layer4[4].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[4].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2' == layer_name:
            self.feature_map.backbone.layer4[4].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.4.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[4].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.4.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.4.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.5' == layer_name:
            self.feature_map.backbone.layer4[5]= new_layer
            self.layer_names["feature_map.backbone.layer4.5"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1' == layer_name:
            self.feature_map.backbone.layer4[5].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[5].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2' == layer_name:
            self.feature_map.backbone.layer4[5].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.5.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[5].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.5.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.5.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.6' == layer_name:
            self.feature_map.backbone.layer4[6]= new_layer
            self.layer_names["feature_map.backbone.layer4.6"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1' == layer_name:
            self.feature_map.backbone.layer4[6].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[6].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2' == layer_name:
            self.feature_map.backbone.layer4[6].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.6.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[6].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.6.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.6.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer4.7' == layer_name:
            self.feature_map.backbone.layer4[7]= new_layer
            self.layer_names["feature_map.backbone.layer4.7"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1' == layer_name:
            self.feature_map.backbone.layer4[7].conv1= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1.0' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1.1' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv1.2' == layer_name:
            self.feature_map.backbone.layer4[7].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2' == layer_name:
            self.feature_map.backbone.layer4[7].conv2= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2.0' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2.1' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer4.7.conv2.2' == layer_name:
            self.feature_map.backbone.layer4[7].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer4.7.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer4.7.conv2.2"]=new_layer
        elif 'feature_map.backbone.conv5' == layer_name:
            self.feature_map.backbone.conv5= new_layer
            self.layer_names["feature_map.backbone.conv5"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv5"]=new_layer
        elif 'feature_map.backbone.conv5.0' == layer_name:
            self.feature_map.backbone.conv5[0]= new_layer
            self.layer_names["feature_map.backbone.conv5.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv5.0"]=new_layer
        elif 'feature_map.backbone.conv5.1' == layer_name:
            self.feature_map.backbone.conv5[1]= new_layer
            self.layer_names["feature_map.backbone.conv5.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv5.1"]=new_layer
        elif 'feature_map.backbone.conv5.2' == layer_name:
            self.feature_map.backbone.conv5[2]= new_layer
            self.layer_names["feature_map.backbone.conv5.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.conv5.2"]=new_layer
        elif 'feature_map.backbone.layer5' == layer_name:
            self.feature_map.backbone.layer5= new_layer
            self.layer_names["feature_map.backbone.layer5"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5"]=new_layer
        elif 'feature_map.backbone.layer5.0' == layer_name:
            self.feature_map.backbone.layer5[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1' == layer_name:
            self.feature_map.backbone.layer5[0].conv1= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[0].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2' == layer_name:
            self.feature_map.backbone.layer5[0].conv2= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.0.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[0].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.0.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer5.1' == layer_name:
            self.feature_map.backbone.layer5[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1' == layer_name:
            self.feature_map.backbone.layer5[1].conv1= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[1].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2' == layer_name:
            self.feature_map.backbone.layer5[1].conv2= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.1.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[1].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.1.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.1.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer5.2' == layer_name:
            self.feature_map.backbone.layer5[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1' == layer_name:
            self.feature_map.backbone.layer5[2].conv1= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[2].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2' == layer_name:
            self.feature_map.backbone.layer5[2].conv2= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.2.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[2].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.2.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.2.conv2.2"]=new_layer
        elif 'feature_map.backbone.layer5.3' == layer_name:
            self.feature_map.backbone.layer5[3]= new_layer
            self.layer_names["feature_map.backbone.layer5.3"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1' == layer_name:
            self.feature_map.backbone.layer5[3].conv1= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1.0' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.0"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1.1' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.1"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv1.2' == layer_name:
            self.feature_map.backbone.layer5[3].conv1[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv1.2"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2' == layer_name:
            self.feature_map.backbone.layer5[3].conv2= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2.0' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[0]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.0"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2.1' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[1]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.1"]=new_layer
        elif 'feature_map.backbone.layer5.3.conv2.2' == layer_name:
            self.feature_map.backbone.layer5[3].conv2[2]= new_layer
            self.layer_names["feature_map.backbone.layer5.3.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backbone.layer5.3.conv2.2"]=new_layer
        elif 'feature_map.backblock0' == layer_name:
            self.feature_map.backblock0= new_layer
            self.layer_names["feature_map.backblock0"]=new_layer
            self.origin_layer_names["feature_map.backblock0"]=new_layer
        elif 'feature_map.backblock0.conv0' == layer_name:
            self.feature_map.backblock0.conv0= new_layer
            self.layer_names["feature_map.backblock0.conv0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv0"]=new_layer
        elif 'feature_map.backblock0.conv0.0' == layer_name:
            self.feature_map.backblock0.conv0[0]= new_layer
            self.layer_names["feature_map.backblock0.conv0.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.0"]=new_layer
        elif 'feature_map.backblock0.conv0.1' == layer_name:
            self.feature_map.backblock0.conv0[1]= new_layer
            self.layer_names["feature_map.backblock0.conv0.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.1"]=new_layer
        elif 'feature_map.backblock0.conv0.2' == layer_name:
            self.feature_map.backblock0.conv0[2]= new_layer
            self.layer_names["feature_map.backblock0.conv0.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv0.2"]=new_layer
        elif 'feature_map.backblock0.conv1' == layer_name:
            self.feature_map.backblock0.conv1= new_layer
            self.layer_names["feature_map.backblock0.conv1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv1"]=new_layer
        elif 'feature_map.backblock0.conv1.0' == layer_name:
            self.feature_map.backblock0.conv1[0]= new_layer
            self.layer_names["feature_map.backblock0.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.0"]=new_layer
        elif 'feature_map.backblock0.conv1.1' == layer_name:
            self.feature_map.backblock0.conv1[1]= new_layer
            self.layer_names["feature_map.backblock0.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.1"]=new_layer
        elif 'feature_map.backblock0.conv1.2' == layer_name:
            self.feature_map.backblock0.conv1[2]= new_layer
            self.layer_names["feature_map.backblock0.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv1.2"]=new_layer
        elif 'feature_map.backblock0.conv2' == layer_name:
            self.feature_map.backblock0.conv2= new_layer
            self.layer_names["feature_map.backblock0.conv2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv2"]=new_layer
        elif 'feature_map.backblock0.conv2.0' == layer_name:
            self.feature_map.backblock0.conv2[0]= new_layer
            self.layer_names["feature_map.backblock0.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.0"]=new_layer
        elif 'feature_map.backblock0.conv2.1' == layer_name:
            self.feature_map.backblock0.conv2[1]= new_layer
            self.layer_names["feature_map.backblock0.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.1"]=new_layer
        elif 'feature_map.backblock0.conv2.2' == layer_name:
            self.feature_map.backblock0.conv2[2]= new_layer
            self.layer_names["feature_map.backblock0.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv2.2"]=new_layer
        elif 'feature_map.backblock0.conv3' == layer_name:
            self.feature_map.backblock0.conv3= new_layer
            self.layer_names["feature_map.backblock0.conv3"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv3"]=new_layer
        elif 'feature_map.backblock0.conv3.0' == layer_name:
            self.feature_map.backblock0.conv3[0]= new_layer
            self.layer_names["feature_map.backblock0.conv3.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.0"]=new_layer
        elif 'feature_map.backblock0.conv3.1' == layer_name:
            self.feature_map.backblock0.conv3[1]= new_layer
            self.layer_names["feature_map.backblock0.conv3.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.1"]=new_layer
        elif 'feature_map.backblock0.conv3.2' == layer_name:
            self.feature_map.backblock0.conv3[2]= new_layer
            self.layer_names["feature_map.backblock0.conv3.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv3.2"]=new_layer
        elif 'feature_map.backblock0.conv4' == layer_name:
            self.feature_map.backblock0.conv4= new_layer
            self.layer_names["feature_map.backblock0.conv4"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv4"]=new_layer
        elif 'feature_map.backblock0.conv4.0' == layer_name:
            self.feature_map.backblock0.conv4[0]= new_layer
            self.layer_names["feature_map.backblock0.conv4.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.0"]=new_layer
        elif 'feature_map.backblock0.conv4.1' == layer_name:
            self.feature_map.backblock0.conv4[1]= new_layer
            self.layer_names["feature_map.backblock0.conv4.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.1"]=new_layer
        elif 'feature_map.backblock0.conv4.2' == layer_name:
            self.feature_map.backblock0.conv4[2]= new_layer
            self.layer_names["feature_map.backblock0.conv4.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv4.2"]=new_layer
        elif 'feature_map.backblock0.conv5' == layer_name:
            self.feature_map.backblock0.conv5= new_layer
            self.layer_names["feature_map.backblock0.conv5"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv5"]=new_layer
        elif 'feature_map.backblock0.conv5.0' == layer_name:
            self.feature_map.backblock0.conv5[0]= new_layer
            self.layer_names["feature_map.backblock0.conv5.0"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv5.0"]=new_layer
        elif 'feature_map.backblock0.conv5.1' == layer_name:
            self.feature_map.backblock0.conv5[1]= new_layer
            self.layer_names["feature_map.backblock0.conv5.1"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv5.1"]=new_layer
        elif 'feature_map.backblock0.conv5.2' == layer_name:
            self.feature_map.backblock0.conv5[2]= new_layer
            self.layer_names["feature_map.backblock0.conv5.2"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv5.2"]=new_layer
        elif 'feature_map.backblock0.conv6' == layer_name:
            self.feature_map.backblock0.conv6= new_layer
            self.layer_names["feature_map.backblock0.conv6"]=new_layer
            self.origin_layer_names["feature_map.backblock0.conv6"]=new_layer
        elif 'feature_map.conv1' == layer_name:
            self.feature_map.conv1= new_layer
            self.layer_names["feature_map.conv1"]=new_layer
            self.origin_layer_names["feature_map.conv1"]=new_layer
        elif 'feature_map.conv1.0' == layer_name:
            self.feature_map.conv1[0]= new_layer
            self.layer_names["feature_map.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.conv1.0"]=new_layer
        elif 'feature_map.conv1.1' == layer_name:
            self.feature_map.conv1[1]= new_layer
            self.layer_names["feature_map.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.conv1.1"]=new_layer
        elif 'feature_map.conv1.2' == layer_name:
            self.feature_map.conv1[2]= new_layer
            self.layer_names["feature_map.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.conv1.2"]=new_layer
        elif 'feature_map.backblock1' == layer_name:
            self.feature_map.backblock1= new_layer
            self.layer_names["feature_map.backblock1"]=new_layer
            self.origin_layer_names["feature_map.backblock1"]=new_layer
        elif 'feature_map.backblock1.conv0' == layer_name:
            self.feature_map.backblock1.conv0= new_layer
            self.layer_names["feature_map.backblock1.conv0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv0"]=new_layer
        elif 'feature_map.backblock1.conv0.0' == layer_name:
            self.feature_map.backblock1.conv0[0]= new_layer
            self.layer_names["feature_map.backblock1.conv0.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.0"]=new_layer
        elif 'feature_map.backblock1.conv0.1' == layer_name:
            self.feature_map.backblock1.conv0[1]= new_layer
            self.layer_names["feature_map.backblock1.conv0.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.1"]=new_layer
        elif 'feature_map.backblock1.conv0.2' == layer_name:
            self.feature_map.backblock1.conv0[2]= new_layer
            self.layer_names["feature_map.backblock1.conv0.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv0.2"]=new_layer
        elif 'feature_map.backblock1.conv1' == layer_name:
            self.feature_map.backblock1.conv1= new_layer
            self.layer_names["feature_map.backblock1.conv1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv1"]=new_layer
        elif 'feature_map.backblock1.conv1.0' == layer_name:
            self.feature_map.backblock1.conv1[0]= new_layer
            self.layer_names["feature_map.backblock1.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.0"]=new_layer
        elif 'feature_map.backblock1.conv1.1' == layer_name:
            self.feature_map.backblock1.conv1[1]= new_layer
            self.layer_names["feature_map.backblock1.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.1"]=new_layer
        elif 'feature_map.backblock1.conv1.2' == layer_name:
            self.feature_map.backblock1.conv1[2]= new_layer
            self.layer_names["feature_map.backblock1.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv1.2"]=new_layer
        elif 'feature_map.backblock1.conv2' == layer_name:
            self.feature_map.backblock1.conv2= new_layer
            self.layer_names["feature_map.backblock1.conv2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv2"]=new_layer
        elif 'feature_map.backblock1.conv2.0' == layer_name:
            self.feature_map.backblock1.conv2[0]= new_layer
            self.layer_names["feature_map.backblock1.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.0"]=new_layer
        elif 'feature_map.backblock1.conv2.1' == layer_name:
            self.feature_map.backblock1.conv2[1]= new_layer
            self.layer_names["feature_map.backblock1.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.1"]=new_layer
        elif 'feature_map.backblock1.conv2.2' == layer_name:
            self.feature_map.backblock1.conv2[2]= new_layer
            self.layer_names["feature_map.backblock1.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv2.2"]=new_layer
        elif 'feature_map.backblock1.conv3' == layer_name:
            self.feature_map.backblock1.conv3= new_layer
            self.layer_names["feature_map.backblock1.conv3"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv3"]=new_layer
        elif 'feature_map.backblock1.conv3.0' == layer_name:
            self.feature_map.backblock1.conv3[0]= new_layer
            self.layer_names["feature_map.backblock1.conv3.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.0"]=new_layer
        elif 'feature_map.backblock1.conv3.1' == layer_name:
            self.feature_map.backblock1.conv3[1]= new_layer
            self.layer_names["feature_map.backblock1.conv3.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.1"]=new_layer
        elif 'feature_map.backblock1.conv3.2' == layer_name:
            self.feature_map.backblock1.conv3[2]= new_layer
            self.layer_names["feature_map.backblock1.conv3.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv3.2"]=new_layer
        elif 'feature_map.backblock1.conv4' == layer_name:
            self.feature_map.backblock1.conv4= new_layer
            self.layer_names["feature_map.backblock1.conv4"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv4"]=new_layer
        elif 'feature_map.backblock1.conv4.0' == layer_name:
            self.feature_map.backblock1.conv4[0]= new_layer
            self.layer_names["feature_map.backblock1.conv4.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.0"]=new_layer
        elif 'feature_map.backblock1.conv4.1' == layer_name:
            self.feature_map.backblock1.conv4[1]= new_layer
            self.layer_names["feature_map.backblock1.conv4.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.1"]=new_layer
        elif 'feature_map.backblock1.conv4.2' == layer_name:
            self.feature_map.backblock1.conv4[2]= new_layer
            self.layer_names["feature_map.backblock1.conv4.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv4.2"]=new_layer
        elif 'feature_map.backblock1.conv5' == layer_name:
            self.feature_map.backblock1.conv5= new_layer
            self.layer_names["feature_map.backblock1.conv5"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv5"]=new_layer
        elif 'feature_map.backblock1.conv5.0' == layer_name:
            self.feature_map.backblock1.conv5[0]= new_layer
            self.layer_names["feature_map.backblock1.conv5.0"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.0"]=new_layer
        elif 'feature_map.backblock1.conv5.1' == layer_name:
            self.feature_map.backblock1.conv5[1]= new_layer
            self.layer_names["feature_map.backblock1.conv5.1"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.1"]=new_layer
        elif 'feature_map.backblock1.conv5.2' == layer_name:
            self.feature_map.backblock1.conv5[2]= new_layer
            self.layer_names["feature_map.backblock1.conv5.2"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv5.2"]=new_layer
        elif 'feature_map.backblock1.conv6' == layer_name:
            self.feature_map.backblock1.conv6= new_layer
            self.layer_names["feature_map.backblock1.conv6"]=new_layer
            self.origin_layer_names["feature_map.backblock1.conv6"]=new_layer
        elif 'feature_map.conv2' == layer_name:
            self.feature_map.conv2= new_layer
            self.layer_names["feature_map.conv2"]=new_layer
            self.origin_layer_names["feature_map.conv2"]=new_layer
        elif 'feature_map.conv2.0' == layer_name:
            self.feature_map.conv2[0]= new_layer
            self.layer_names["feature_map.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.conv2.0"]=new_layer
        elif 'feature_map.conv2.1' == layer_name:
            self.feature_map.conv2[1]= new_layer
            self.layer_names["feature_map.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.conv2.1"]=new_layer
        elif 'feature_map.conv2.2' == layer_name:
            self.feature_map.conv2[2]= new_layer
            self.layer_names["feature_map.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.conv2.2"]=new_layer
        elif 'feature_map.backblock2' == layer_name:
            self.feature_map.backblock2= new_layer
            self.layer_names["feature_map.backblock2"]=new_layer
            self.origin_layer_names["feature_map.backblock2"]=new_layer
        elif 'feature_map.backblock2.conv0' == layer_name:
            self.feature_map.backblock2.conv0= new_layer
            self.layer_names["feature_map.backblock2.conv0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv0"]=new_layer
        elif 'feature_map.backblock2.conv0.0' == layer_name:
            self.feature_map.backblock2.conv0[0]= new_layer
            self.layer_names["feature_map.backblock2.conv0.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.0"]=new_layer
        elif 'feature_map.backblock2.conv0.1' == layer_name:
            self.feature_map.backblock2.conv0[1]= new_layer
            self.layer_names["feature_map.backblock2.conv0.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.1"]=new_layer
        elif 'feature_map.backblock2.conv0.2' == layer_name:
            self.feature_map.backblock2.conv0[2]= new_layer
            self.layer_names["feature_map.backblock2.conv0.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv0.2"]=new_layer
        elif 'feature_map.backblock2.conv1' == layer_name:
            self.feature_map.backblock2.conv1= new_layer
            self.layer_names["feature_map.backblock2.conv1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv1"]=new_layer
        elif 'feature_map.backblock2.conv1.0' == layer_name:
            self.feature_map.backblock2.conv1[0]= new_layer
            self.layer_names["feature_map.backblock2.conv1.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.0"]=new_layer
        elif 'feature_map.backblock2.conv1.1' == layer_name:
            self.feature_map.backblock2.conv1[1]= new_layer
            self.layer_names["feature_map.backblock2.conv1.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.1"]=new_layer
        elif 'feature_map.backblock2.conv1.2' == layer_name:
            self.feature_map.backblock2.conv1[2]= new_layer
            self.layer_names["feature_map.backblock2.conv1.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv1.2"]=new_layer
        elif 'feature_map.backblock2.conv2' == layer_name:
            self.feature_map.backblock2.conv2= new_layer
            self.layer_names["feature_map.backblock2.conv2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv2"]=new_layer
        elif 'feature_map.backblock2.conv2.0' == layer_name:
            self.feature_map.backblock2.conv2[0]= new_layer
            self.layer_names["feature_map.backblock2.conv2.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.0"]=new_layer
        elif 'feature_map.backblock2.conv2.1' == layer_name:
            self.feature_map.backblock2.conv2[1]= new_layer
            self.layer_names["feature_map.backblock2.conv2.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.1"]=new_layer
        elif 'feature_map.backblock2.conv2.2' == layer_name:
            self.feature_map.backblock2.conv2[2]= new_layer
            self.layer_names["feature_map.backblock2.conv2.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv2.2"]=new_layer
        elif 'feature_map.backblock2.conv3' == layer_name:
            self.feature_map.backblock2.conv3= new_layer
            self.layer_names["feature_map.backblock2.conv3"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv3"]=new_layer
        elif 'feature_map.backblock2.conv3.0' == layer_name:
            self.feature_map.backblock2.conv3[0]= new_layer
            self.layer_names["feature_map.backblock2.conv3.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.0"]=new_layer
        elif 'feature_map.backblock2.conv3.1' == layer_name:
            self.feature_map.backblock2.conv3[1]= new_layer
            self.layer_names["feature_map.backblock2.conv3.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.1"]=new_layer
        elif 'feature_map.backblock2.conv3.2' == layer_name:
            self.feature_map.backblock2.conv3[2]= new_layer
            self.layer_names["feature_map.backblock2.conv3.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv3.2"]=new_layer
        elif 'feature_map.backblock2.conv4' == layer_name:
            self.feature_map.backblock2.conv4= new_layer
            self.layer_names["feature_map.backblock2.conv4"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv4"]=new_layer
        elif 'feature_map.backblock2.conv4.0' == layer_name:
            self.feature_map.backblock2.conv4[0]= new_layer
            self.layer_names["feature_map.backblock2.conv4.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.0"]=new_layer
        elif 'feature_map.backblock2.conv4.1' == layer_name:
            self.feature_map.backblock2.conv4[1]= new_layer
            self.layer_names["feature_map.backblock2.conv4.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.1"]=new_layer
        elif 'feature_map.backblock2.conv4.2' == layer_name:
            self.feature_map.backblock2.conv4[2]= new_layer
            self.layer_names["feature_map.backblock2.conv4.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv4.2"]=new_layer
        elif 'feature_map.backblock2.conv5' == layer_name:
            self.feature_map.backblock2.conv5= new_layer
            self.layer_names["feature_map.backblock2.conv5"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv5"]=new_layer
        elif 'feature_map.backblock2.conv5.0' == layer_name:
            self.feature_map.backblock2.conv5[0]= new_layer
            self.layer_names["feature_map.backblock2.conv5.0"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.0"]=new_layer
        elif 'feature_map.backblock2.conv5.1' == layer_name:
            self.feature_map.backblock2.conv5[1]= new_layer
            self.layer_names["feature_map.backblock2.conv5.1"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.1"]=new_layer
        elif 'feature_map.backblock2.conv5.2' == layer_name:
            self.feature_map.backblock2.conv5[2]= new_layer
            self.layer_names["feature_map.backblock2.conv5.2"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv5.2"]=new_layer
        elif 'feature_map.backblock2.conv6' == layer_name:
            self.feature_map.backblock2.conv6= new_layer
            self.layer_names["feature_map.backblock2.conv6"]=new_layer
            self.origin_layer_names["feature_map.backblock2.conv6"]=new_layer
        elif 'detect_1' == layer_name:
            self.detect_1= new_layer
            self.layer_names["detect_1"]=new_layer
            self.origin_layer_names["detect_1"]=new_layer
        elif 'detect_1.sigmoid' == layer_name:
            self.detect_1.sigmoid= new_layer
            self.layer_names["detect_1.sigmoid"]=new_layer
            self.origin_layer_names["detect_1.sigmoid"]=new_layer
        elif 'detect_2' == layer_name:
            self.detect_2= new_layer
            self.layer_names["detect_2"]=new_layer
            self.origin_layer_names["detect_2"]=new_layer
        elif 'detect_2.sigmoid' == layer_name:
            self.detect_2.sigmoid= new_layer
            self.layer_names["detect_2.sigmoid"]=new_layer
            self.origin_layer_names["detect_2.sigmoid"]=new_layer
        elif 'detect_3' == layer_name:
            self.detect_3= new_layer
            self.layer_names["detect_3"]=new_layer
            self.origin_layer_names["detect_3"]=new_layer
        elif 'detect_3.sigmoid' == layer_name:
            self.detect_3.sigmoid= new_layer
            self.layer_names["detect_3.sigmoid"]=new_layer
            self.origin_layer_names["detect_3.sigmoid"]=new_layer




    def get_order(self, layer_name):
        if layer_name not in self.orders.keys():
            return False
        return self.orders[layer_name]

    def set_order(self, layer_name,order):
        if layer_name not in self.orders.keys():
            return False
        self.orders[layer_name]=order

    def get_outshape(self, layer_name):

        if layer_name not in self.out_shapes.keys():
            return False

        return self.out_shapes[layer_name]

    def set_outshape(self, layer_name,out):

        if layer_name not in self.out_shapes.keys():
            return False

        self.out_shapes[layer_name]=out

    def get_inshape(self, layer_name):
        if layer_name not in self.in_shapes.keys():
            return False

        return self.in_shapes[layer_name]

    def set_inshape(self, layer_name,out):
        if layer_name not in self.in_shapes.keys():
            return False

        self.in_shapes[layer_name]=out

    def set_Basic_OPS(self,b):
        self.Basic_OPS=b
    def get_Cascade_OPs(self):
        return self.Cascade_OPs

    def get_Basic_OPS(self):
        return self.Basic_OPS

    def set_Cascade_OPS(self,c):
        self.Cascade_OPs=c


    def construct(self, x):
        input_shape = F.shape(x)[2:4]
        input_shape = F.cast(self.tenser_to_array(input_shape), ms.float32)
        big_object_output, medium_object_output, small_object_output = self.feature_map(x)
        if not self.keep_detect:
            return big_object_output, medium_object_output, small_object_output
        output_big = self.detect_1(big_object_output, input_shape)
        output_me = self.detect_2(medium_object_output, input_shape)
        output_small = self.detect_3(small_object_output, input_shape)
        # big is the final output which has smallest feature map
        return output_big, output_me, output_small





class YoloWithLossCell(nn.Cell):
    """YOLOV3 loss."""
    def __init__(self, network, config=default_config):
        super(YoloWithLossCell, self).__init__()
        self.yolo_network = network
        self.config = config
        self.tenser_to_array = P.TupleToArray()
        self.loss_big = YoloLossBlock('l', self.config)
        self.loss_me = YoloLossBlock('m', self.config)
        self.loss_small = YoloLossBlock('s', self.config)

    def construct(self, x, y_true_0, y_true_1, y_true_2, gt_0, gt_1, gt_2):
        input_shape = F.shape(x)[2:4]
        input_shape = F.cast(self.tenser_to_array(input_shape), ms.float32)
        yolo_out = self.yolo_network(x)
        loss_l = self.loss_big(*yolo_out[0], y_true_0, gt_0, input_shape)
        loss_m = self.loss_me(*yolo_out[1], y_true_1, gt_1, input_shape)
        loss_s = self.loss_small(*yolo_out[2], y_true_2, gt_2, input_shape)
        return loss_l + loss_m + loss_s


class TrainingWrapper(nn.Cell):
    """Training wrapper."""
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        self.optimizer(grads)
        return loss


if __name__ == '__main__':
    from models.yolov3_darknet53.model_utils.config import config
    model=YOLOV3DarkNet53(is_training=True,config=config)