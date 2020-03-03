import mxnet as mx

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from configs.configs import configs
from custom_op.farthest_point_sampler import FarthestPointSampler
from custom_op.random_sampler import RandomSampler
from custom_op.radius_search import RadiusSearch
from utils.ops import batch_take, knn, conv1d, conv2d, mlp1d, mlp2d

CC = configs['num_channel']


def sampling(data, shape, method='fps', fps_npoints=0, scope=''):
    """
    PointNet++ sampling operation
    Input:
        data: (B, N, CC), Symbol, CC = 3 or 6
        shape: specify (B, N, _)
        method: 'fps' or 'random'
        fps_npoints: int32, #points sampled in farthest point sampling
    Output:
        centeroids: (B, fps_npoints, CC), Symbol
    """
    B, N, _ = shape
    if N == fps_npoints:
        return data
    if method == 'fps':
        centeroids_ids = mx.symbol.Custom(data=data, npoints=fps_npoints, name=scope+'/fps_idx', op_type='farthest_point_sampler')  # (B, fps_npoints)
    elif method == 'random':
        centeroids_ids = mx.symbol.Custom(total=N, B=B, N=fps_npoints, name=scope+'/rand_idx', op_type='random_sampler')  # (B, fps_npoints)
    else:
        raise NotImplementedError
    centeroids = batch_take(data, centeroids_ids, shape=(B, fps_npoints, N, CC), scope=scope+'/fps')  # (B, fps_npoints, CC)
    return centeroids

def grouping(data, feature, centeroids, shape, fps_npoints, rs_npoints, radius=0, xyz_mlp=[], method='rs', knn_dilation=1, scope='', conxyz=True):
    """
    PointNet++ grouping operation
    Input:
        data: (B, N, CC), Symbol, CC = 3 or 6
        feature: (B, N, C), Symbol
        centeroids: (B, fps_npoints, CC), Symbol
        shape: specify (B, N, C)
        fps_npoints: int32, #points sampled in farthest point sampling
        rs_npoints: #points sampled in radius search
        radius: float32, search radius
        xyz_mlp: list of int32, MLP for relative data before concatenating with feature
        method: 'rs' or 'knn'
        knn_dilation: dilation factor in knn. Takes effect if method=='knn'
        scope: str, name_scope
    Output:
        local_xyz: (B, fps_npoints, rs_npoints, CC), Symbol, relative xyz
        new_feature: (B, fps_npoints, rs_npoints, CC+C), Symbol, new features
        If len(xyz_mlp)>0, CC=xyz_mlp[-1]
    """
    B, N, C = shape
    if method == 'rs':
        grouped_ids = mx.symbol.Custom(centeroids=centeroids, data=data, radius=radius, npoints=rs_npoints, name=scope+'_rs_idx', op_type='radius_search')  # (B, fps_npoints, rs_npoints)
    elif method == 'knn':
        _, grouped_ids = knn(query=centeroids, data=data, shape=(fps_npoints, N), k=rs_npoints*knn_dilation, scope=scope+'/knn')
        if knn_dilation > 1:
            grouped_ids = mx.symbol.slice(grouped_ids, begin=(0,0,0), end=(-1,-1,-1), step=(1,1,knn_dilation))  # (B, fps_npoints, rs_npoints)
    else:
        raise NotImplementedError
    local_xyz = batch_take(data, grouped_ids, shape=(B, fps_npoints, rs_npoints, N, CC), scope=scope+'/grouped_xyz')  # (B, fps_npoints, rs_npoints, CC)
    local_xyz_f = mx.symbol.broadcast_sub(local_xyz, centeroids.expand_dims(axis=2), name=scope+"/broad_sub")  # (B, fps_npoints, rs_npoints, CC), translation normalization
    if xyz_mlp:
        print("has xyz_mlp: ", scope)
        local_xyz_f = mlp2d(local_xyz_f, xyz_mlp, use_bn=False, scope=scope+'/grouped_xyz')
    grouped_feature = None
    if feature is not None:
        grouped_feature = batch_take(feature, grouped_ids, shape=(B, fps_npoints, rs_npoints, N, C), scope=scope+'/grouped_feat')  # (B, fps_npoints, rs_npoints, C)
    if conxyz:
        if feature is not None:
            grouped_feature = mx.symbol.concat(local_xyz_f, grouped_feature, dim=3)  # (B, fps_npoints, rs_npoints, CC+C)
        else:
            grouped_feature = local_xyz_f
    else:
        local_xyz_f = local_xyz
    return local_xyz_f, grouped_feature

def grouping_abs(data, feature, centeroids, shape, fps_npoints, rs_npoints, radius=0, xyz_mlp=[], method='rs', knn_dilation=1, scope=''):
    """
    PointNet++ grouping operation
    Input:
        data: (B, N, CC), Symbol, CC = 3 or 6
        feature: (B, N, C), Symbol
        centeroids: (B, fps_npoints, CC), Symbol
        shape: specify (B, N, C)
        fps_npoints: int32, #points sampled in farthest point sampling
        rs_npoints: #points sampled in radius search
        radius: float32, search radius
        xyz_mlp: list of int32, MLP for relative data before concatenating with feature
        method: 'rs' or 'knn'
        knn_dilation: dilation factor in knn. Takes effect if method=='knn'
        scope: str, name_scope
    Output:
        local_xyz: (B, fps_npoints, rs_npoints, CC), Symbol, relative xyz
        new_feature: (B, fps_npoints, rs_npoints, CC+C), Symbol, new features
        If len(xyz_mlp)>0, CC=xyz_mlp[-1]
    """
    B, N, C = shape
    if method == 'rs':
        grouped_ids = mx.symbol.Custom(centeroids=centeroids, data=data, radius=radius, npoints=rs_npoints, name=scope+'_rs_idx', op_type='radius_search')  # (B, fps_npoints, rs_npoints)
    elif method == 'knn':
        _, grouped_ids = knn(query=centeroids, data=data, shape=(fps_npoints, N), k=rs_npoints*knn_dilation, scope=scope+'/knn')
        if knn_dilation > 1:
            grouped_ids = mx.symbol.slice(grouped_ids, begin=(0,0,0), end=(-1,-1,-1), step=(1,1,knn_dilation))  # (B, fps_npoints, rs_npoints)
    else:
        raise NotImplementedError
    grouped_data = batch_take(data, grouped_ids, shape=(B, fps_npoints, rs_npoints, N, CC), scope=scope+'/grouped_xyz')  # (B, fps_npoints, rs_npoints, CC)
    local_xyz = grouped_data
    if xyz_mlp:
        grouped_data = mlp2d(grouped_data, xyz_mlp, use_bn=False, scope=scope+'/grouped_xyz')
    if feature is not None:
        grouped_feature = batch_take(feature, grouped_ids, shape=(B, fps_npoints, rs_npoints, N, C), scope=scope+'/grouped_feat')  # (B, fps_npoints, rs_npoints, C)
        grouped_data = mx.symbol.concat(grouped_data, grouped_feature, dim=3)  # (B, fps_npoints, rs_npoints, CC+C)
    return local_xyz, grouped_data

def grouping_all(data, feature, xyz_mlp=[], scope=''):
    """
    PointNet++ group_all operation
    Input:
        data: (B, N, CC), Symbol
        feature: (B, N, C), Symbol
        xyz_mlp: list of int32, MLP for relative data before concatenating with feature
    Output:
        new_feature: (B, 1, N, CC+C), Symbol
    """
    if xyz_mlp:
        data = mlp1d(data, xyz_mlp, use_bn=False, scope=scope+'grouped _xyz')
    if feature is not None:
        data = mx.symbol.concat(data, feature, dim=2)
    return data.expand_dims(axis=1, name=scope+'/group_all')

def interpolate(query, data, feature, shape, k=3, scope=''):
    """
    perform distance based interpolation
    Input:
        query: (B, M, 3), Symbol, locations to interpolate
        data: (B, N, 3), Symbol, locations of existing points
        feature: (B, N, C), Symbol, feature of existing points
        shape: specify (B, M, N, C)
    Output:
        interp_feature: (B, M, C), Symbol, feature of interpolated points
    """
    B, M, N, C = shape
    dist, ids = knn(query, data, (M,N), k)  # (B, M, k)
    weight = 1. / mx.symbol.maximum(dist, 1e-10)
    norm = mx.symbol.sum(weight, axis=2, keepdims=True)  # (B, M, 1)
    weight = mx.symbol.broadcast_div(weight, norm).expand_dims(axis=3)  # (B, M, k, 1)
    interp_feature = batch_take(feature, ids, shape=(B, M, k, N, C), scope=scope)  # (B, M, k, C)
    interp_feature = mx.symbol.broadcast_mul(interp_feature, weight)  # (B, M, k, C)
    interp_feature = mx.symbol.sum(interp_feature, axis=2)  # (B, M, C)
    return interp_feature

def mlp_and_pool(data, mlp_list, use_bn=True, bn_decay=0.9, attr=None, scope=''):
    """
    mlp and max-pooling
    Input:
        data: (B, fps_npoints, rs_npoints, CC+C), Symbol
        mlp: list of int32, output size of MLP
        bn_decay: decay parameter in batch normalization
    Output:
        pooled_data: (B, fps_npoints, mlp[-1]), Symbol
    """
    # mlp
    data = mlp2d(data, mlp_list, use_bn=use_bn, bn_decay=bn_decay, scope=scope)
    # pooling
    data = mx.symbol.max(data, axis=2, attr=attr, name=scope+'/pool')
    return data

def mlp_and_pooling(data, mlp_list, rs_npoints, use_bn=True, bn_decay=0.9, attr=None, scope=''):
    """
    mlp and max-pooling
    Input:
        data: (B, fps_npoints, rs_npoints, CC+C), Symbol
        mlp: list of int32, output size of MLP
        bn_decay: decay parameter in batch normalization
    Output:
        pooled_data: (B, fps_npoints, mlp[-1]), Symbol
    """
    # mlp
    data = mlp2d(data, mlp_list, use_bn=use_bn, bn_decay=bn_decay, scope=scope)
    # pooling
    data = mx.symbol.transpose(data, axes=(0,3,1,2)) # change to B C H W

    data = mx.sym.Pooling(name=scope + "/max_pooling", data=data,
                   kernel=(1, rs_npoints), pool_type="max")

    return mx.sym.squeeze(mx.symbol.transpose(data, axes=(0,2,3,1)), axis=2)

def pointnet_sa_module(data, feature, shape, sampling_method='fps', fps_npoints=None, radius_list=None,
        rs_npoints_list=None, group_all=False, xyz_mlp=[], mlp_list=[], bn_decay=0.9, lr_mult=None, scope=''):
    """
    PointNet Set Abstraction Module - supports multi scale grouping (MSG)
    Input:
        data: (B, N, CC), Symbol
        feature: (B, N, C), Symbol
        shape: specify (B, N, C)
        sampling_method: 'fps' or 'random'
        fps_npoints: int32, #points sampled in farthest point sampling
        radius_list: float32, or list of float32, search radius
        rs_npoints_list: int32, or list of int32, #points sampled in radius search
        group_all: boolean, whether to keep all points in grouping (same as setting fps_npoints=1, radius=inf, rs_npoints=N)
        xyz_mlp: list of int32, MLP for relative data before concatenating with feature
        mlp_list: list of int32, or list of list of int32, output size of MLP
        bn_decay: decay parameter in batch normalization
        scope: str, name_scope
    Returns:
        centeroids: (B, fps_npoints, CC), Symbol
        new_feature: (B, fps_npoints, sum(mlp[-1])), Symbol
    """
    if lr_mult is None: attr = None
    else: attr = {'lr_mult': str(lr_mult)}
    if group_all:
        grouped_data = grouping_all(data, feature, xyz_mlp=xyz_mlp, scope=scope)
        if configs["agg"] == "max_pooling":
            pooled_data = mlp_and_pooling(grouped_data, mlp_list, 128, bn_decay=bn_decay, attr=attr, scope=scope)
        elif configs["agg"] == "max":
            pooled_data = mlp_and_pool(grouped_data, mlp_list, bn_decay=bn_decay, attr=attr, scope=scope)
        else:
            raise NotImplementedError
        return None, pooled_data
    else:
        # sampling
        centeroids = sampling(data, shape, method=sampling_method, fps_npoints=fps_npoints, scope=scope)
        # grouping
        pooled_data = []
        if not isinstance(radius_list, list):
            radius_list, rs_npoints_list, mlp_list = [radius_list], [rs_npoints_list], [mlp_list]
        for i, (radius, rs_npoints, mlp_list_) in enumerate(zip(radius_list, rs_npoints_list, mlp_list)):
            _, grouped_data = grouping(data, feature, centeroids, shape, fps_npoints,
                rs_npoints, radius=radius, xyz_mlp=xyz_mlp, scope='{}/{}'.format(scope, i+1))

            if configs["agg"] == "max_pooling":
                _pooled_data = mlp_and_pooling(grouped_data, mlp_list_, rs_npoints, use_bn=True,
                    bn_decay=bn_decay, attr=attr, scope='{}/{}'.format(scope, i+1))
            elif configs["agg"] == "max":
                _pooled_data = mlp_and_pool(grouped_data, mlp_list_,
                    bn_decay=bn_decay, attr=attr, scope='{}/{}'.format(scope, i+1))
            else:
                raise NotImplementedError
            pooled_data.append(_pooled_data)
        if len(pooled_data) == 1:
            pooled_data = pooled_data[0]
        else:
            pooled_data = mx.symbol.concat(*pooled_data, dim=2)
        return centeroids, pooled_data


def pointnet_fp_module(data, feature, query_data, query_feature, shape, mlp_list, bn_decay=0.9, lr_mult=None, scope=''):
    """
    PointNet Feature Propagation Module
    Input:
        data: (B, N, 3), Symbol
        feature: (B, N, C1), Symbol
        query_data: (B, M, 3), Symbol
        query_feature: (B, M, C2), Symbol, or None
        shape: specify (B, M, N, C1)
        mlp_list: list of int32, output size of MLP
        bn_decay: decay parameter in batch normalization
        scope: sre, name scope
    Returns:
        new_feature: (B, M, mlp[-1]), Symbol
    """
    B, M, N, C1 = shape
    if lr_mult is None: attr = None
    else: attr = {'lr_mult': str(lr_mult)}
    interp_feature = interpolate(query_data, data, feature, shape=(B, M, N, C1), scope=scope)  # (B, M, C1)
    if query_feature is not None:
        interp_feature = mx.symbol.concat(interp_feature, query_feature, dim=2)  # (B, M, C1+C2)
    new_feature = mlp1d(interp_feature, mlp_list, use_bn=True, bn_decay=bn_decay, attr=attr, scope=scope)  # (B, M, mlp[-1])
    return new_feature


