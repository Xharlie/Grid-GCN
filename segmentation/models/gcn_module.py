import mxnet as mx
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.ops import mlp1d, mlp2d, conv1d, conv2d, depthwise_conv2d, separable_conv2d, batch_take, fully_connected
from configs.configs import configs
import pnt2_module

# def connect_edge_layer(centers, edge_type = "ball", edge_properties = None, edge_num = None, edge_num_limit = 10000):
#     '''
#     :param centers: the center xyz or featuer space of the center
#     :param edge_type: use ball for ball query, ball_knn for ball query + knn, knn, or grid of squarish grid query
#     :param edge_properties: edge properties to provide in verts pair calculation.
#     :param edge_num: a fixed edge num to be built. if not enough, padding zeros, if exceed the amount, sample it.
#     :param edge_num_limit: if edge_num is not provide, the maximum allowed edge number.
#     :return:
#     '''
#     pass

def update_func(centers, center_feats, outDim=[64, 256], scope="center_update"):
    '''
    :param centers: B * max_o_grid * 3
    :param center_feats: B * max_o_grid * C
    :param outDim: [128, 256,...]
    :return: center_feats  B * max_o_grid * outDim[-1]
    '''
    if configs["relu"]:
        center_feats = mx.symbol.relu(center_feats, name=scope+'/pre_relu')
    if len(outDim) != 0:
        center_feats = mlp1d(center_feats, outDim, bn_decay=0.9, use_bn=True,
                             attr=None, scope=scope + '/cov1d')
    return center_feats

def aggregation_func(masked_pair_feats, neighbor_masks, geo_dist, max_o_grid, max_p_grid,
                     pool_type = "max", mlp_lst = None, scope='center_aggregation'):
    '''
    :param masked_pair_feats: B * max_o_grid * max_p_grid * C
    :param mlp: true of false
    :return:
    '''
    if pool_type == "max_pooling":
        masked_pair_feats = mx.symbol.transpose(masked_pair_feats, axes=(0,3,1,2)) # change to B C H W
        aggregated_feats = mx.sym.Pooling(name=scope+"/max_pooling", data=masked_pair_feats,
            kernel=(1, max_p_grid), pool_type="max")
        aggregated_feats = mx.sym.squeeze(mx.symbol.transpose(aggregated_feats, axes=(0,2,3,1)), axis=2)
    elif pool_type == "max":
        aggregated_feats = mx.symbol.max(masked_pair_feats, axis=2, name=scope+'/max')
    elif pool_type == "sum":
        aggregated_feats = mx.symbol.sum(masked_pair_feats, axis=2, name=scope + '/sum')
    elif pool_type == "inter_sum":
        aggregated_feats = interpolate_all(geo_dist, masked_pair_feats, neighbor_masks, scope=scope+'/interp')  # (B, M, C1)
    else:
        raise NotImplementedError
    if mlp_lst is not None:
        aggregated_feats = mlp1d(aggregated_feats, mlp_lst, bn_decay=0.9, use_bn=True, attr=None,
                                           scope=scope + '/agg_mlp/cov1d')
    return aggregated_feats

def interpolate_all(dist, masked_pair_feats, neighbor_masks, scope=''):
    """
    perform distance based interpolation
    Input:
        geo_dist: (B, O, P), Symbol, locations to interpolate
        masked_pair_feats: (B, O, P, C), Symbol, locations of existing points
        neighbor_masks: (B, O, P, 1), Symbol, feature of existing points
        shape: specify (B, M, N, C)
    Output:
        interp_feature: (B, M, C), Symbol, feature of interpolated points
    """
    weight = 1. / mx.symbol.maximum(dist**2, 1e-10) # (B, O, P)
    weight = mx.sym.elemwise_mul(weight, neighbor_masks)  # (B, O, P)
    norm = mx.symbol.sum(weight, axis=2, keepdims=True)   # (B, O, 1)
    weight = mx.symbol.broadcast_div(weight, norm).expand_dims(axis=3)  # (B, O, P, 1)
    interp_feature = mx.symbol.broadcast_mul(masked_pair_feats, weight)  # (B, O, P, C)
    interp_feature = mx.symbol.sum(interp_feature, axis=2)  # (B, O, C)
    return interp_feature

# def onenn(dist, masked_pair_feats, neighbor_masks, scope=''):
#     min_dist = mx.sym.min(dist, axis=2, keepdims=True)
#     weight = mx.sym.elemwise_mul(mx.sym.broadcast_equal(dist, min_dist), neighbor_masks).expand_dims(axis=3)
#     interp_feature = mx.symbol.broadcast_mul(masked_pair_feats, weight)  # (B, O, P, C)
#     interp_feature = mx.symbol.sum(interp_feature, axis=2)  # (B, O, C)
#     return interp_feature

def neighbors_fetch(neighbors, neighbor_masks, centers, shape, k=3, scope=''):
    """
    k nearest neighbor
    neighbors: symbol, shape=(B, O, P, 4+C)
    neighbor_masks: symbol, shape=(B, O, P)
    centers: symbol, shape=(B, O, 4)
    shape: specify (M, N)
    scope: name scope for this operator
    returns:
        dist, shape=(B, M, k), squared distances to the k nearest points
        ids, shape=(B, M, k), dtype=int32, indices of the k nearest points
    """
    B, O, P, C = shape
    neighbor_locs = mx.sym.slice_axis(neighbors, axis=3, begin=0, end=3)  # shape=(B, O, P, 3)
    centers_expand_xyz = mx.sym.slice_axis(mx.sym.expand_dims(centers, axis=2), axis=3, begin=0, end=3) # shape=(B, O, 1, 3)
    diff = mx.symbol.broadcast_sub(neighbor_locs, centers_expand_xyz)  # (B, O, P, 3)
    all_dist = mx.symbol.sum(diff*diff, axis=3) + (1.0 - neighbor_masks) # (B, O, P)
    ids = mx.symbol.topk(all_dist, axis=2, k=k, ret_typ='indices',
        is_ascend=True, name='{}_top{}'.format(scope, k))   # (B, O, k)
    ids = ids.astype('int32')
    neighbors_k = neigh_batch_take(neighbors, ids, shape=(B, O, P, C, k), scope=scope)  # (B, M, k, C+4)
    neighbor_masks_k = mx.sym.squeeze(neigh_batch_take(neighbor_masks, ids, shape=(B, O, P, 1, k), scope=scope))  # (B, M, k, C+4)
    return neighbors_k, neighbor_masks_k

def neigh_batch_take(data, index, shape, scope=''):
    """
    Per-batch take operator. Perform on axis 1
    data:  symbol, shape=(B, O, P, C+4)
    index: symbol, shape=(B, ...)
    shape: specify shape (B, ..., N, C+4)
    scope: name scope for this operator
    returns: symbol, shape=(B, ..., C+4)
    """
    B, O, P, C, K = shape[0], shape[1], shape[2], shape[3], shape[4]
    data = mx.symbol.reshape(data, shape=(B*O*P, C), name=scope+'_pretake_reshape')
    index_transformer = mx.symbol.arange(B*O, repeat=K, dtype=np.int32).reshape((B, O, K)) * P
    index = index + index_transformer
    outputs = mx.symbol.take(data, index, name=scope+'_take')
    return outputs

def verts_pair_func(centers, neighbor_feats, att_vec, B, max_o_grid, max_p_grid, N, C, pt_mlp_lst= None, scope="pair_relation"):
    '''
    :param centers:  B * max_o_grid * 3
    :param neighbor_feats:  B * max_o_grid * max_p_grid * C
    :param neighbor_locs:  B * max_o_grid * max_p_grid * 3
    :param C:
    :param max_p_grid:
    :param scope:
    :return: geo_feat: B * max_o_grid * max_p_grid * C
    '''

    if pt_mlp_lst is not None and len(pt_mlp_lst) > 0:
        C = pt_mlp_lst[-1]
        print("pt_mlp_lst:", pt_mlp_lst)
        neighbor_feats = mlp2d(neighbor_feats, pt_mlp_lst, use_bn=True, bn_decay=configs["bn_decay"], scope=scope)
    # centers=centers.reshape((B, max_o_grid, 3), name=scope+"/centerreshape")
    if configs["attfdim"] > 0:
        att_vec = fully_connected(att_vec, C, bn_decay=0.9, dropout_ratio=0, flatten=False, use_bn=True, use_relu=True,
                        bn_axis=1, attr=None, scope=scope+'/fc1', flip=True)
        att_vec = fully_connected(att_vec, C, bn_decay=0.9, dropout_ratio=0, flatten=False, use_bn=True, use_relu=True,
                                   bn_axis=1, attr=None, scope=scope + '/fc2', flip=True)
        att_vec = fully_connected(att_vec, C, bn_decay=0.9, dropout_ratio=0, flatten=False, use_bn=True, use_relu=True,
                                   bn_axis=1, attr=None, scope=scope + '/fc3', flip=True)

        pair_feats = att_vec * neighbor_feats
    else:
        pair_feats = neighbor_feats
    return pair_feats

def sub_g_update(centers, neighbors, has_feats, center_masks, neighbor_masks, attfdim,
        center_ori_feats = None, pt_mlp_lst = None, outDim= [64, 256], shape = [64, 800000, 100, 6], scope='layer',
        no_mask=False, aggtype="gcn", pool_type="max_pooling", neighbor_num = 0, att_full=False):
    '''
    RELU( MAX_POOLING ((geoRelation() * f_neib)) )
    :param centers:  B * max_o_grid * 3
    :param neighbor_feats: B * max_o_grid * max_p_grid * 4+C
    :param center_masks: B * max_o_grid
    :param neighbor_masks: B * max_o_grid * max_p_grid
    :param scope:
    :return:
    '''
    #
    B, max_o_grid, max_p_grid, N, C = shape
    if neighbor_num != 0:
        neighbors, neighbor_masks = neighbors_fetch(neighbors, neighbor_masks, centers,
            shape=(B, max_o_grid, max_p_grid, C), k=neighbor_num, scope=scope+"/neighfetch")
        max_p_grid = neighbor_num
    neighbor_feats = mx.sym.slice_axis(neighbors, axis=3, begin=4, end=None) if has_feats else None

    centers_expand = mx.sym.tile(mx.sym.expand_dims(centers, axis=2), reps=(1, 1, max_p_grid, 1))
    centers_expand_xyz = mx.sym.slice_axis(centers_expand, axis=3, begin=0, end=3)
    neighbor_locs_xyz = mx.sym.slice_axis(neighbors, axis=3, begin=0, end=3)
    geo_vec = neighbor_locs_xyz - centers_expand_xyz # B * max_o_grid * max_p_grid * 3
    geo_dist = mx.sym.sqrt(mx.sym.sum(mx.sym.square(geo_vec), axis=3)) # B * max_o_grid * max_p_grid

    if max(attfdim, configs["localfdim"]) in [4,5,11,12]:
        geo_num = mx.sym.slice_axis(neighbors, axis=3, begin=3, end=4)
        geo_den_sum = mx.sym.sum(geo_num, axis=3, keepdims=True) # B * max_o_grid * 1
        num_points = mx.sym.ones_like(geo_den_sum, name = scope+"/oneslike") * configs["num_points"]
        geo_density = mx.sym.elemwise_div(geo_num, geo_den_sum, name=scope+"/den_ele_div")
        geo_global_density = mx.sym.elemwise_div(geo_num, num_points, name=scope+"/den_glo_ele_div")
    if attfdim <= 3:
        att_vec = geo_vec
    elif attfdim == 4:
        att_vec =  mx.sym.concat(geo_vec, geo_density, dim=3)
    elif attfdim == 5:
        att_vec = mx.sym.concat(geo_vec, geo_density, geo_global_density, dim=3)
    elif attfdim == 10:
        att_vec = mx.sym.concat(geo_dist.expand_dims(axis=3), geo_vec, centers_expand_xyz, neighbor_locs_xyz, dim=3)
    elif attfdim == 11:
        att_vec = mx.sym.concat(geo_dist.expand_dims(axis=3), geo_vec, centers_expand_xyz, neighbor_locs_xyz, geo_density, dim=3)
    elif attfdim == 12:
        att_vec = mx.sym.concat(geo_dist.expand_dims(axis=3), geo_vec, centers_expand_xyz, neighbor_locs_xyz, geo_density, geo_global_density, dim=3)
    else:
        raise NotImplementedError

    if configs["localfdim"] <= 3:
        geo_feats = geo_vec
    elif configs["localfdim"] == 4:
        geo_feats =  mx.sym.concat(geo_vec, geo_density, dim=3)
    elif configs["localfdim"] == 5:
        geo_feats = mx.sym.concat(geo_vec, geo_density, geo_global_density, dim=3)
    elif configs["localfdim"] == 10:
        geo_feats = mx.sym.concat(geo_dist.expand_dims(axis=3), geo_vec, centers_expand_xyz, neighbor_locs_xyz, dim=3)
    elif configs["localfdim"] == 11:
        geo_feats = mx.sym.concat(geo_dist.expand_dims(axis=3), geo_vec, centers_expand_xyz, neighbor_locs_xyz, geo_density, dim=3)
    elif configs["localfdim"] == 12:
        geo_feats = mx.sym.concat(geo_dist.expand_dims(axis=3), geo_vec, centers_expand_xyz, neighbor_locs_xyz, geo_density, geo_global_density, dim=3)
    else:
        raise NotImplementedError

    if neighbor_feats is None:
        neighbor_feats = geo_feats
    else:
        neighbor_feats = mx.symbol.concat(geo_feats, neighbor_feats, dim=3)
    neighbor_masks_expand = mx.sym.expand_dims(neighbor_masks, axis=3)

    if aggtype == "gcn":
        if center_ori_feats is not None:
            center_ori_feats = mx.sym.tile(mx.sym.expand_dims(center_ori_feats, axis=2), reps=(1, 1, max_p_grid, 1))
            neighbor_feats = mx.symbol.concat(neighbor_feats, center_ori_feats, dim=3)
        if att_full:
            att_vec = mx.sym.concat(att_vec, neighbor_feats, dim=3)
        pair_feats = verts_pair_func(centers, neighbor_feats, att_vec,
            B, max_o_grid, max_p_grid, N, C, pt_mlp_lst = pt_mlp_lst, scope=scope) # B * max_o_grid * max_p_grid * C
        # pair_feats = pair_feats.reshape((B, max_o_grid, max_p_grid, C), name=scope+"/pair_feats_reshape")
        if not no_mask and pool_type!="max" and pool_type!="max_pooling":
            pair_feats = mx.sym.broadcast_mul(pair_feats, neighbor_masks_expand, name="pairmask") # B * max_o_grid * max_p_grid * C
        center_feats = aggregation_func(pair_feats, neighbor_masks, geo_dist, max_o_grid, max_p_grid, pool_type=pool_type, scope=scope+"/center_aggregation") # B * max_o_grid * 1 * outDim[-1]
        center_feats = update_func(centers, center_feats, outDim=outDim, scope=scope+"/center_update") # B * max_o_grid * outDim[-1]
        # center_feats = center_feats.reshape((B, max_o_grid, pt_mlp_lst[-1]), name=scope+"/center_feats_reshape")
        if not no_mask and center_masks is not None:
            center_feats = mx.sym.broadcast_mul(center_feats, mx.sym.expand_dims(center_masks,axis=2), name="centermask") # B * max_o_grid * outDim[-1]
    elif aggtype == "agg_gcn":
        agg_neigh_feats = aggregation_func(neighbor_feats, neighbor_masks, geo_dist, max_o_grid, max_p_grid, pool_type=pool_type,
                                        mlp_lst = pt_mlp_lst, scope=scope + "/neigh_aggregation")
        agg_feats = mx.symbol.concat(agg_neigh_feats, center_ori_feats, dim=2)
        center_feats = update_func(centers, agg_feats, outDim=outDim, scope=scope + "/center_update")  # B * max_o_grid * outDim[-1]
        if not no_mask and center_masks is not None:
            center_feats = mx.sym.broadcast_mul(center_feats, mx.sym.expand_dims(center_masks, axis=2), name="centermask")  # B * max_o_grid * outDim[-1]
    elif aggtype == "pnt":
        # (B, fps_npoints, rs_npoints, CC+C)
        # return (B, fps_npoints, sum(mlp[-1]))
        center_feats = mlp_mask_and_pool(neighbor_feats, pt_mlp_lst, max_p_grid, pool_type=pool_type,
            use_bn=True, mask= neighbor_masks_expand, scope='{}/{}'.format(scope, "mlp_mask_and_pool"))
        if not no_mask:
            center_feats = mx.sym.broadcast_mul(center_feats, mx.sym.expand_dims(center_masks,axis=2), name="centermask") # B * max_o_grid * outDim[-1]
    else:
        raise NotImplementedError("aggtype %s is not implemented" % aggtype)
    return center_feats

def mlp_mask_and_pool(data, mlp_list, max_p_grid, pool_type="max", use_bn=True, bn_decay=0.9, mask=None, scope=''):
    """
    mlp and max-pooling
    Input:
        data: (B, fps_npoints, rs_npoints, C), Symbol
        mlp: list of int32, output size of MLP
        bn_decay: decay parameter in batch normalization
    Output:
        pooled_data: (B, fps_npoints, mlp[-1]), Symbol
    """
    # mlp
    data = mlp2d(data, mlp_list, use_bn=use_bn, bn_decay=bn_decay, scope=scope)
    # pooling
    if mask is not None and pool_type!="max" and pool_type!="max_pooling":
        data = mx.sym.broadcast_mul(data, mask, axis=3, name="pairmask")  # B * max_o_grid * max_p_grid * C
    # data = mx.symbol.max(data, axis=2, name=scope+'/pool')
    if pool_type == "max_pooling":
        data = mx.symbol.transpose(data, axes=(0,3,1,2)) # change to B C H W
        data = mx.sym.Pooling(name=scope + "/max_pooling", data=data,
                       kernel=(1, max_p_grid), pool_type="max")
        return mx.sym.squeeze(mx.symbol.transpose(data, axes=(0,2,3,1)), axis=2)
    elif pool_type == "max":
        data = mx.symbol.max(data, axis=2, name=scope + '/max')
        return data
    else:
        raise NotImplementedError("agg %s is no implemented" % pool_type)
