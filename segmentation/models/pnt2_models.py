"""
PointNet++ model for ModelNet40 classification and ScanNet semantic labelling
"""

import mxnet as mx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from configs.configs import configs_pnt as configs
from models.pnt2_module import pointnet_sa_module, pointnet_fp_module
from utils.ops import fully_connected, conv1d
import custom_op.weighted_gradient

print("configs for models",configs)
def get_cls_inputs():
    data = mx.symbol.Variable(name='data')
    label = mx.symbol.Variable(name='label')
    return data, label

def get_cls_head(features, label, dropout_ratio=0.5, bn_decay=0.9, weights=None):
    """
    Get symbol for ModelNet40 shape classification
    """
    net = fully_connected(features, num_hidden=512, flatten=True, bn_decay=bn_decay, dropout_ratio=dropout_ratio, scope='fc1')
    net = fully_connected(net, num_hidden=256, bn_decay=bn_decay, dropout_ratio=dropout_ratio, scope='fc2')
    net = mx.symbol.FullyConnected(net, num_hidden=40, name='fc3')
    if weights is not None:
        net = mx.symbol.Custom(data=net, weight=weights, input_dim=2, name='fc3_weighted', op_type='weighted_gradient')
    net = mx.symbol.SoftmaxOutput(data=net, label=label, normalization='batch', name='pred')
    return net

def get_seg_inputs():
    data = mx.symbol.Variable(name='data')
    label = mx.symbol.Variable(name='label')
    return data, label

def get_seg_head(features, label, dropout_ratio=0.5, bn_decay=0.9, weights=None):
    """
    Get symbol for ScanNet semantic labelling
    """
    net = mx.symbol.transpose(features, axes=(0,2,1))
    net = conv1d(net, num_filter=128, kernel=(1,), stride=(1,), bn_decay=bn_decay, layout='NCW', scope='fc1')
    if dropout_ratio > 0:
        net = mx.symbol.Dropout(net, p=dropout_ratio, name='fc1/dropout')
    net = conv1d(net, num_filter=21, kernel=(1,), stride=(1,), layout='NCW', use_bn=False, use_relu=False, scope='fc2')
    if weights is not None:
        net = mx.symbol.Custom(data=net, weight=weights, input_dim=3, name='fc2/weighted', op_type='weighted_gradient')
    net = mx.symbol.SoftmaxOutput(data=net, label=label, use_ignore=True, ignore_label=0, multi_output=True, normalization='valid', name='pred')
    return net

def get_feature_cls_ssg(data, B, N, bn_decay=0.9):
    """
    centeroids: (B, 128, CC)
    features: (B, 128, 256)
    """
    # different settings as for whether use_normal or not
    if configs['num_points'] > 2048:
        rs_npoints = [128, 128]
    else:
        rs_npoints = [32, 64]
    # PointNet++ set abstraction
    # centeroids, features = pointnet_sa_module(data, None, shape=(B, N, 0), fps_npoints=512, radius_list=0.2, rs_npoints_list=rs_npoints[0], xyz_mlp=[32,32], mlp_list=[64,64,128], bn_decay=bn_decay, scope='layer1')
    # centeroids, features = pointnet_sa_module(centeroids, features, shape=(B, 512, 128), fps_npoints=128, radius_list=0.4, rs_npoints_list=rs_npoints[1], xyz_mlp=[32,32], mlp_list=[128,128,256], bn_decay=bn_decay, scope='layer2')
    centeroids, features = pointnet_sa_module(data, None, shape=(B, N, 0), fps_npoints=512, radius_list=0.2, rs_npoints_list=rs_npoints[0], mlp_list=[64,64,128], bn_decay=bn_decay, scope='layer1')
    centeroids, features = pointnet_sa_module(centeroids, features, shape=(B, 512, 128), fps_npoints=128, radius_list=0.4, rs_npoints_list=rs_npoints[1], mlp_list=[128,128,256], bn_decay=bn_decay, scope='layer2')
    return centeroids, features

def get_feature_cls_msg(data, B, N, bn_decay=0.9):
    """
    centeroids: (B, 128, CC)
    features: (B, 128, 640)
    """
    # PointNet++ set abstraction
    centeroids, features = pointnet_sa_module(data, None, shape=(B, N, 0), fps_npoints=512, radius_list=[0.1,0.2,0.4], rs_npoints_list=[16,32,128], xyz_mlp=[32,32], mlp_list=[[32,32,64],[64,64,128],[64,96,128]], bn_decay=bn_decay, scope='layer1')
    centeroids, features = pointnet_sa_module(centeroids, features, shape=(B, 512, 320), fps_npoints=128, radius_list=[0.2,0.4,0.8], rs_npoints_list=[32,64,128], xyz_mlp=[32,32], mlp_list=[[64,64,128],[128,128,256],[128,128,256]], bn_decay=bn_decay, scope='layer2')
    return centeroids, features

def get_feature_seg(data, B, N, bn_decay=0.9, lr_mult=None):
    """
    centeroids1: (B, 1024, 3)
    centeroids2: (B, 256, 3)
    centeroids3: (B, 64, 3)
    centeroids4: (B, 16, 3)
    features1: (B, 1024, 64)
    features2: (B, 256, 128)
    features3: (B, 64, 256)
    features4: (B, 16, 512)
    """
    # PointNet++ set abstraction
    # print()
    centeroids1, features1 = pointnet_sa_module(data, None, shape=(B, N, 0), sampling_method='fps', fps_npoints=8192, radius_list=0.1, rs_npoints_list=32, xyz_mlp=[32,32], mlp_list=[32,32,64], bn_decay=bn_decay, lr_mult=lr_mult, scope='layer1')
    centeroids2, features2 = pointnet_sa_module(centeroids1, features1, shape=(B, 8192, 64), sampling_method='fps', fps_npoints=256, radius_list=0.2, rs_npoints_list=32, xyz_mlp=[32,32], mlp_list=[64,64,128], bn_decay=bn_decay, lr_mult=lr_mult, scope='layer2')
    centeroids3, features3 = pointnet_sa_module(centeroids2, features2, shape=(B, 256, 128), sampling_method='fps', fps_npoints=64, radius_list=0.4, rs_npoints_list=32, xyz_mlp=[32,32], mlp_list=[128,128,256], bn_decay=bn_decay, lr_mult=lr_mult, scope='layer3')
    centeroids4, features4 = pointnet_sa_module(centeroids3, features3, shape=(B, 64, 256), sampling_method='fps', fps_npoints=16, radius_list=0.8, rs_npoints_list=32, xyz_mlp=[32,32], mlp_list=[256,256,512], bn_decay=bn_decay, lr_mult=lr_mult, scope='layer4')
    return [centeroids1, centeroids2, centeroids3, centeroids4], [features1, features2, features3, features4]


def get_symbol_cls_ssg(B, N, bn_decay=0.9, weights=None):
    """
    Get symbol for PointNet++ ModelNet40 classification (Single-Scale Grouping)
    B: batch_size
    N: #points in each batch
    weights: weights for weighted_gradient
    The model has inputs:
      data: (B, N, 3), point clouds
      label: (B,), ground truth labels
    returns: symbol, (B, 40)
    """
    # inputs
    data, label = get_cls_inputs()
    # get features
    centeroids, features = get_feature_cls_ssg(data, B, N, bn_decay)
    # centeroids, features = pointnet_sa_module(centeroids, features, shape=(B, 128, 256), group_all=True, xyz_mlp=[32,32], mlp_list=[256,512,1024], bn_decay=bn_decay, scope='layer3')
    centeroids, features = pointnet_sa_module(centeroids, features, shape=(B, 128, 256), group_all=True, mlp_list=[256,512,1024], bn_decay=bn_decay, scope='layer3')
    # get features -- pretrain for segmentation
    #(_, _, _, centeroids), (_, _, _, features) = get_feature_seg(data, B, N, bn_decay, lr_mult=None)
    #centeroids, features = pointnet_sa_module(centeroids, features, shape=(B, 16, 512), group_all=True, mlp_list=[512,512,1024], bn_decay=bn_decay, scope='layer5')
    # classification head
    net = get_cls_head(features, label, bn_decay=bn_decay, weights=weights)
    return net

def get_symbol_cls_msg(B, N, bn_decay=0.9, weights=None):
    """
    Get symbol for PointNet++ ModelNet40 classification (Multi-Scale Grouping)
    B: batch_size
    N: #points in each batch
    weights: weights for weighted_gradient
    The model has inputs:
      data: (B, N, 3), point clouds
      label: (B,), ground truth labels
    returns: symbol, (B, 40)
    """
    # inputs
    data, label = get_cls_inputs()
    # get features
    centeroids, features = get_feature_cls_msg(data, B, N, bn_decay)
    centeroids, features = pointnet_sa_module(centeroids, features, shape=(B, 128, 640), group_all=True, xyz_mlp=[32,32], mlp_list=[256,512,1024], bn_decay=bn_decay, scope='layer3')
    # classification
    net = get_cls_head(features, label, bn_decay=bn_decay, weights=weights)
    return net


def get_symbol_seg(B, N, bn_decay=0.9, weights=None):
    """
    Get symbol for PointNet++ ScanNet semantic labelling
    B: batch size
    N: #points in each batch
    The model has inputs:
      data: (B, N, 3), point clouds
      label: (B, N), ground truth labels
    returns: symbol, (B, 21, N)
    """
    # inputs
    print(B,N,"B!,N!")
    data, label =  get_seg_inputs()
    # get features
    (centeroids1, centeroids2, centeroids3, centeroids4), (features1, features2, features3, features4) = get_feature_seg(data, B, N, bn_decay=bn_decay, lr_mult=None)
    # PointNet++ feature propagation
    features3 = pointnet_fp_module(centeroids4, features4, centeroids3, features3, shape=(B, 64, 16, 512), mlp_list=[256,256], bn_decay=bn_decay, scope='layer5')
    features2 = pointnet_fp_module(centeroids3, features3, centeroids2, features2, shape=(B, 256, 64, 256), mlp_list=[256,256], bn_decay=bn_decay, scope='layer6')
    features1 = pointnet_fp_module(centeroids2, features2, centeroids1, features1, shape=(B, 8192, 256, 256), mlp_list=[256,128], bn_decay=bn_decay, scope='layer7')
    features = pointnet_fp_module(centeroids1, features1, data, None, shape=(B, N, 8192, 128), mlp_list=[128,128,128], bn_decay=bn_decay, scope='layer8')
    # segmentation head
    net = get_seg_head(features, label, bn_decay=bn_decay, weights=weights)
    return net


if __name__ == "__main__":
    '''
    #net = get_symbol_cls_ssg(B=64, N=1024)
    net = get_symbol_seg(B=64, N=1024)
    t = mx.viz.plot_network(net)
    t.render()
    '''

    # test pointnet_fp_module
    context = mx.cpu()
    data_data = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                data_data.append([i,j,k])
    data_data = mx.nd.array(data_data)
    data_data = mx.nd.stack(data_data, data_data*2, axis=0)  # (2, 27, 3)
    query_data = mx.nd.array([[[0.5, 0.5, 0.5]],[[0.5,0.5,0.5]]])  # (2, 1, 3)
    feature_data = mx.nd.arange(27).reshape((27, 1))
    feature_data = mx.nd.stack(feature_data, feature_data, axis=0)
    query = mx.symbol.Variable('query')
    data = mx.symbol.Variable('data')
    feature = mx.symbol.Variable('feature')
    output = pointnet_fp_module(data, feature, query, None, (2,1,27,1), mlp_list=[])
    mod = mx.mod.Module(output, data_names=['query', 'data', 'feature'], label_names=[], context=context)
    mod.bind(data_shapes=[('query', query_data.shape), ('data', data_data.shape), ('feature', feature_data.shape)])
    mod.init_params()
    mod.forward(mx.io.DataBatch([query_data, data_data, feature_data]))
    print(mod.get_outputs()[0].asnumpy())  # [[[4/3]], [[12/17]]]

