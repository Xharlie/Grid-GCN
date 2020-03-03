"""
PointCNN models for ModelNet40 classification and ScanNet segmentation
"""

import mxnet as mx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models import pointcnn_module
from utils.ops import conv1d


def get_cls_inputs():
    data = mx.symbol.Variable(name='data')
    label = mx.symbol.Variable(name='label')
    return data, label

def get_cls_head(features, label, dropout_ratio=0, bn_decay=0.9, weights=None):
    """
    Get symbol for ModelNet40 shape classification
    """
    net = mx.symbol.transpose(features, axes=(0,2,1))
    net = conv1d(net, num_filter=384, kernel=(1,), stride=(1,), bn_decay=bn_decay, layout='NCW', scope='fc1')
    net = conv1d(net, num_filter=192, kernel=(1,), stride=(1,), bn_decay=bn_decay, layout='NCW', scope='fc2')
    if dropout_ratio > 0:
        net = mx.symbol.Dropout(net, p=dropout_ratio, name='fc2/dropout')
    net = conv1d(net, num_filter=40, kernel=(1,), stride=(1,), layout='NCW', use_bn=False, use_relu=False, scope='fc3')
    if weights is not None:
        net = mx.symbol.Custom(data=net, weight=weights, input_dim=3, name='fc3/weighted', op_type='weighted_gradient')
    net = mx.symbol.SoftmaxOutput(data=net, label=label, multi_output=True, normalization='batch', name='pred')
    return net

def get_feature_cls(data, B, N, bn_decay=0.9):
    points, features = pointcnn_module(data, None, shape=(B, N, 0, 48),
            sampling_method='random', fps_npoints=N,
            grouping_method='knn', rs_npoints=8, knn_dilation=1,
            C_delta=24, depth_multiplier=4,
            bn_decay=bn_decay, with_global=False, scope='layer1')
    points, features = pointcnn_module(points, features, shape=(B, N, 48, 96),
            sampling_method='random', fps_npoints=384,
            grouping_method='knn', rs_npoints=12, knn_dilation=2,
            C_delta=12, depth_multiplier=2,
            bn_decay=bn_decay, with_global=False, scope='layer2')
    points, features = pointcnn_module(points, features, shape=(B, 384, 96, 192),
            sampling_method='random', fps_npoints=128,
            grouping_method='knn', rs_npoints=16, knn_dilation=2,
            C_delta=24, depth_multiplier=2,
            bn_decay=bn_decay, with_global=False, scope='layer3')
    points, features = pointcnn_module(points, features, shape=(B, 128, 192, 384),
            sampling_method='random', fps_npoints=128,
            grouping_method='knn', rs_npoints=16, knn_dilation=3,
            C_delta=48, depth_multiplier=2,
            bn_decay=bn_decay, with_global=True, scope='layer4')
    return points, features

def get_symbol_cls(B, N, bn_decay=0.9, weights=None):
    """
    Get symbol for PointCNN ModelNet40 classification
    B: batch_size
    N: #points in each batch
    is_training: bool
    weights: weights for weighted_gradient
    The model has inputs:
      data: (B, N, 3), point clouds
      label: (B,), ground truth labels
    returns: symbol, (B, 40)
    """
    # inputs
    data, label = get_cls_inputs()
    # get features
    points, features = get_feature_cls(data, B, N, bn_decay=bn_decay)
    # get cls head
    net = get_cls_head(features, label, dropout_ratio=0, bn_decay=bn_decay, weights=weights)
    return net


if __name__ == "__main__":
    net = get_symbol_cls(B=64, N=1024)
    t = mx.viz.plot_network(net)
    t.render()


