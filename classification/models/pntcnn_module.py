import mxnet as mx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from models import sampling, grouping
from utils import mlp1d, conv2d, depthwise_conv2d, separable_conv2d

def xconv(points, features, shape, depth_multiplier=1, bn_decay=0.9, scope=''):
    """
    PointCNN X-Conv operator
    Input:
        points: Symbol, shape=(B, M, K, 3), relative xyz (subtracted from centeroids)
        features: Symbol, shape=(B, M, K, C1), concatenated features of xyz and prev features
        shape: specify (B, M, K, C1, C2)
        depth_multiplier: depth_multiplier in separable_conv2d (the normal convolution in paper)
        bn_decay: batchnorm momentum
    Output:
        new_features: Symbol, shape=(B, M, C2)
    """
    B, M, K, C1, C2 = shape
    # prepare X transformation matrix
    points = points.transpose(axes=(0,3,2,1))  # (B, 3, K, M)
    X = conv2d(points, num_filter=K*K, kernel=(K,1), stride=(1,1),
            layout='NCHW', use_bn=True, scope=scope+'/X0')
    X = X.reshape((B, K, K, M))
    X = depthwise_conv2d(X, in_channels=K, depth_multiplier=K, kernel=(K,1), stride=(1,1),
            layout='NCHW', use_bn=True, scope=scope+'/X1')
    X = X.reshape((B, K, K, M))
    X = depthwise_conv2d(X, in_channels=K, depth_multiplier=K, kernel=(K,1), stride=(1,1),
            layout='NCHW', use_bn=True, use_relu=False, scope=scope+'/X2')
    X = X.reshape((B, K, K, M))
    X = X.transpose(axes=(0,3,2,1))  # (B, M, K, K)
    '''
    points = points.transpose(axes=(0,3,1,2))
    points = mx.symbol.BatchNorm(data=points, momentum=bn_decay, axis=1, fix_gamma=False, use_global_stats=False, name=scope+'/Xbn')
    X = conv2d(points, num_filter=K*K, kernel=(1,K), stride=(1,1),
            layout='NCHW', use_bn=False, scope=scope+'/X0')
    X = conv2d(X, num_filter=K*K, kernel=(1,1), stride=(1,1),
            layout='NCHW', use_bn=False, scope=scope+'/X1')
    X = conv2d(X, num_filter=K*K, kernel=(1,1), stride=(1,1),
            layout='NCHW', use_bn=False, use_relu=False, scope=scope+'/X2')
    X = X.transpose(axes=(0,2,3,1)).reshape((B, M, K, K))
    '''
    # X-Conv
    X_features = mx.symbol.linalg.gemm2(X, features, name=scope+'/xconv')  # (B, M, K, C1)
    # normal convolution
    X_features = X_features.transpose(axes=(0,3,1,2))  # (B, C1, M, K)
    X_features = separable_conv2d(X_features, in_channels=C1, num_filter=C2,
            depth_multiplier=depth_multiplier, kernel=(1,K), stride=(1,1),
            bn_decay=bn_decay, layout='NCHW', use_bn=True, scope=scope+'/conv')
    X_features = X_features.reshape((B, C2, M)).transpose(axes=(0,2,1))  # (B, M, C2)
    return X_features


def pointcnn_module(points, features, shape, sampling_method='random', fps_npoints=0, grouping_method='knn', rs_npoints=0, knn_dilation=1, C_delta=0, depth_multiplier=1, bn_decay=0.9, with_global=False, scope=''):
    """
    PointCNN X-Conv Module
    Input:
        points: Symbol, shape=(B, N, 3), xyz
        features: Symbol, shape=(B, N, C1), additional feature
        shape: specify (B, N, C1, C2)
        sampling_method: 'fps' or 'random'
        fps_npoints: number of representative points in fps/random sampling
        grouping_method: 'rs' or 'knn'
        rs_npoints: number of neighbors in rs/knn grouping
        knn_dilation: dilation rate in knn grouping
        C_delta: number of intermediate channels (see paper)
        depth_multiplier: depth_multiplier in separable_conv2d (the normal convolution in paper)
        bn_decay: bn_decay in the normal convolution
        with_global: whether to concatenate final features with knn xyz features
    Output:
        new_points: Symbol, shape=(B, fps_npoints, 3)
        new_features: Symbol. If with_global flag is set, shape=(B, fps_npoints, 1.25*C2); otherwise, shape=(B, fps_npoints, C2)
    """
    B, N, C1, C2 = shape

    # sampling
    centeroids = sampling(points, shape=(B, N, 3), method=sampling_method, fps_npoints=fps_npoints, scope=scope+'/sampling')  # (B, fps_npoints, 3)

    # grouping
    local_xyz, grouped_features = grouping(points, features, centeroids, (B, N, C1),
            fps_npoints, rs_npoints, xyz_mlp=[C_delta, C_delta], scope=scope+'/grouping')
    # local_xyz: (B, fps_npoints, rs_npoints, 3)
    # grouped_features: (B, fps_npoints, rs_npoints, C_delta+C1)

    # X-Conv
    new_features = xconv(local_xyz, grouped_features,
            shape=(B, fps_npoints, rs_npoints, C_delta+C1, C2),
            depth_multiplier=depth_multiplier,
            bn_decay=bn_decay, scope=scope+'/xconv')  # (B, fps_npoints, C2)
    if with_global:
        final_C2 = C2 // 4
        centeroids_features = mlp1d(centeroids, [final_C2, final_C2], use_bn=True, scope=scope+'/global')
        new_features = mx.symbol.concat(centeroids_features, new_features, dim=2)  # (B, fps_npoints, 1.25*C2)
    return centeroids, new_features


