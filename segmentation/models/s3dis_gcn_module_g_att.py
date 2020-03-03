import mxnet as mx
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.ops import mlp1d_c, mlp2d_c, fully_connected_mlp_withbn, fully_connected_mlp, fully_connected
from s3dis_configs.configs import configs

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
BN = False if (configs["use_bn"] == 'f' or configs["use_bn"] == 'p') else True
C_dim = 1 if BN else 3
P_dim = 3 if BN else 2
def update_func(center_feats, outDim=[64, 256], scope="center_update", bn_decay=0.9):
    '''
    :param centers: B * max_o_grid * 3
    :param center_feats: B * max_o_grid * C
    :param outDim: [128, 256,...]
    :return: center_feats  B * max_o_grid * outDim[-1]
    '''
    if configs["relu"]:
        center_feats = mx.symbol.relu(center_feats, name=scope+'/pre_relu')
    if len(outDim) != 0:
        if BN:
            center_feats = mlp1d_c(center_feats, outDim, bn_decay=bn_decay, use_bn=BN,
                                   attr=None, scope=scope + '/cov1d')
        elif configs["use_bn"] == 'p':
            center_feats = fully_connected_mlp_withbn(center_feats, outDim, dim = 3, bn_decay=bn_decay, dropout_ratio=0,
                flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope + '/cov1dfc', flip=False)
        else:
            center_feats = fully_connected_mlp(center_feats, outDim, dim = 3, bn_decay=bn_decay, dropout_ratio=0,
                flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope + '/cov1dfc', flip=False)
    return center_feats

def aggregation_func(masked_pair_feats, neighbor_masks_expand, geo_dist, max_o_grid, max_p_grid,
                     pool_type = "max", mlp_lst = None, scope='center_aggregation', bn_decay=0.9):
    '''
    :param masked_pair_feats: B * max_o_grid * max_p_grid * C
    :param mlp: true of false
    :return:
    '''
    if pool_type == "max_pooling":
        # aggregated_feats = mx.sym.Pooling(name=scope+"/max_pooling", data=masked_pair_feats,
        #     kernel=(1, max_p_grid), pool_type="max", layout="NCHW" if BN else "NHWC")
        # aggregated_feats = mx.sym.squeeze(aggregated_feats, axis=P_dim)

        aggregated_feats = mx.sym.Pooling(name=scope + "/max_pooling", data=masked_pair_feats,
                                          kernel=(1, max_p_grid) if BN else (max_p_grid, 1), pool_type="max", layout="NCHW")
        aggregated_feats = mx.sym.squeeze(aggregated_feats, axis=P_dim)

    elif pool_type == "max":
        aggregated_feats = mx.symbol.max(masked_pair_feats, axis=P_dim, name=scope + '/max')
    elif pool_type == "sum":
        aggregated_feats = mx.symbol.sum(masked_pair_feats, axis=P_dim, name=scope + '/sum')
    elif pool_type == "inter_sum":
        aggregated_feats = interpolate_all(geo_dist, masked_pair_feats, neighbor_masks_expand, scope=scope+'/interp')  # (B, M, C1)
    else:
        raise NotImplementedError
    if mlp_lst is not None:
        if BN:
            aggregated_feats = mlp1d_c(aggregated_feats, mlp_lst, bn_decay=bn_decay, use_bn=BN, attr=None,
                                           scope=scope + '/agg_mlp/cov1d')
        elif configs["use_bn"] == 'p':
            aggregated_feats = fully_connected_mlp_withbn(aggregated_feats, mlp_lst, dim=3, bn_decay=bn_decay, dropout_ratio=0,
                flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope + '/agg_fc', flip=False)
        else:
            aggregated_feats = fully_connected_mlp(aggregated_feats, mlp_lst, dim = 3, bn_decay=bn_decay, dropout_ratio=0, flatten=False, use_bn=BN,
                            use_relu=True, bn_axis=1, attr=None, scope=scope+'/agg_fc', flip=False)
    return aggregated_feats

def interpolate_all(dist, masked_pair_feats, neighbor_masks_expand, scope=''):
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
    if  neighbor_masks_expand is not None:
        weight = mx.sym.elemwise_mul(weight, neighbor_masks_expand)  # (B, O, P)
    norm = mx.symbol.sum(weight, axis=P_dim, keepdims=True)   # (B, O, 1)
    weight = mx.symbol.broadcast_div(weight, norm, keepdims=True)  # (B, 1, O, P)
    interp_feature = mx.symbol.broadcast_mul(masked_pair_feats, weight)  # (B, C, O, P)
    interp_feature = mx.symbol.sum(interp_feature, axis=P_dim)  # (B, O, C)

    return interp_feature

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
    if BN:
        data = mx.symbol.transpose(data, axes=(0, 2, 3, 1))
    data = mx.symbol.reshape(data, shape=(B*O*P, C), name=scope+'_pretake_reshape')
    index_transformer = mx.symbol.arange(B*O, repeat=K, dtype=np.int32).reshape((B, O, K)) * P
    index = index + index_transformer
    outputs = mx.symbol.take(data, index, name=scope+'_take')
    if BN:
        outputs = mx.symbol.transpose(outputs, axes=(0, 3, 1, 2))
    return outputs

def verts_pair_func(ori_neighbor_feats, att_vec, B, max_o_grid, max_p_grid, N, C, att_full = "", pt_mlp_lst= None, contextvec = None,scope="pair_relation", bn_decay=0.9, neighbor_masks_expand=None, att_norm=None):
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
        if BN:
            neighbor_feats = mlp2d_c(ori_neighbor_feats, pt_mlp_lst, use_bn=BN, bn_decay=bn_decay, scope=scope)
        else:
            neighbor_feats = fully_connected_mlp(ori_neighbor_feats, pt_mlp_lst, dim = 4, bn_decay=bn_decay, dropout_ratio=0, flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope + '/cov2fc', flip=False)
    # centers=centers.reshape((B, max_o_grid, 3), name=scope+"/centerreshape")
    if configs["attfdim"] > 0:
        if BN:
            att_vec = mlp2d_c(att_vec, [C//4], use_bn=BN, bn_decay=bn_decay, attr=None,
                              scope=scope + "/update_att_mlp2d_frst")
            if att_full == "last":
                att_vec = mx.sym.concat(att_vec, ori_neighbor_feats, dim=C_dim, name=scope + "/att_full_concat")
                print("has attfull last")
            elif att_full == "next":
                att_vec = mx.sym.concat(att_vec, neighbor_feats, dim=C_dim, name=scope + "/att_full_concat")
                print("has attfull next")
            if contextvec is not None:
                att_vec = mx.sym.concat(att_vec, contextvec, dim=C_dim, name=scope + "/att_contextvec_concat")
                print("att_vec has contextvec")
            att_vec = mlp2d_c(att_vec, [C, C] if contextvec is not None else [C], use_bn=BN, bn_decay=bn_decay, attr=None, scope=scope + "/update_att_mlp2d_scnd", relutyp="leaky" if configs["attrelu"] == "leaky" and att_norm is not None else "relu")
        else:
            att_vec = fully_connected_mlp(att_vec, [C//4], dim=4, bn_decay=bn_decay, dropout_ratio=0, flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope + '/update_att_mlp2dfc_frst', flip=False)
            if att_full == "last":
                att_vec = mx.sym.concat(att_vec, ori_neighbor_feats, dim=C_dim, name=scope + "/att_full_concat")
                print("has attfull last")
            elif att_full == "next":
                att_vec = mx.sym.concat(att_vec, neighbor_feats, dim=C_dim, name=scope + "/att_full_concat")
                print("has attfull next")
            if contextvec is not None:
                att_vec = mx.sym.concat(att_vec, contextvec, dim=C_dim, name=scope + "/att_contextvec_concat")
                print("att_vec has contextvec")
            att_vec = fully_connected_mlp(att_vec, [C, C] if contextvec is not None else [C], dim=4, bn_decay=bn_decay, dropout_ratio=0, flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope + '/update_att_mlp2dfc_scnd', flip=False)
        if att_norm is not None:
            if att_norm== "softmax":
                att_vec_max = mx.sym.max(att_vec, axis=P_dim, keepdims=True)
                att_vec = mx.sym.broadcast_add(att_vec, - att_vec_max)
                att_vec = mx.sym.exp(att_vec) 
            elif att_norm== "l2":
                att_vec = mx.sym.square(att_vec)
            elif att_norm== "l1":
                att_vec = mx.sym.abs(att_vec)
            if  neighbor_masks_expand is not None:
                att_vec = mx.symbol.broadcast_mul(att_vec, neighbor_masks_expand)
            att_sum = mx.symbol.maximum(mx.sym.sum(att_vec, axis = P_dim, keepdims=True), 0.001)
            att_vec = mx.symbol.broadcast_div(att_vec, att_sum) 
        pair_feats = att_vec * neighbor_feats
    else:
        pair_feats = neighbor_feats
    return pair_feats

def sub_g_update(centers_xyz, center_den, neighbors, notFirstLayer, center_masks, neighbor_masks, attfdim,
        center_ori_feats = None, pt_mlp_lst = None, outDim= [64, 256], cntxt_mlp = None, shape = [64, 800000, 100, 6], scope='layer', aggtype="gcn", pool_type="max_pooling", att_full="", center_dim=[], recalden=False, bn_decay=0.9, att_norm=None):
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
    neighbor_masks_expand = None
    if neighbor_masks is not None:
        neighbor_masks_expand = mx.sym.expand_dims(neighbor_masks, axis=C_dim, name=scope+"/concatmask")
    neighbor_feats = mx.sym.slice_axis(neighbors, axis=C_dim, begin=4, end=None) if notFirstLayer or configs["indim"]>0 else None
    # todo expand centers in Gridify
    centers_expand_xyz = mx.sym.tile(mx.sym.expand_dims(centers_xyz, axis=P_dim, name=scope + "/centerxyzexpand"), reps=(1, 1, 1, max_p_grid) if BN else (1, 1, max_p_grid, 1))
    neighbor_locs_xyz = mx.sym.slice_axis(neighbors, axis=C_dim, begin=0, end=3)
    geo_vec = neighbor_locs_xyz - centers_expand_xyz # B * 3 * max_o_grid * max_p_grid
    geo_dist = mx.sym.sqrt(mx.sym.sum(mx.sym.square(geo_vec), axis=C_dim, keepdims=True)) # B * max_o_grid * max_p_grid

    if max(attfdim, configs["localfdim"]) in [5,11,12]:
        geo_num = mx.sym.slice_axis(neighbors, axis=C_dim, begin=3, end=4) # B * 1 * O * P or B * O * P * 1
        num_points = mx.sym.ones_like(geo_num, name = scope+"/oneslike") * configs["num_points"]
        geo_global_density = mx.sym.elemwise_div(geo_num, num_points, name=scope+"/den_glo_ele_div")
        if max(attfdim, configs["localfdim"]) in [5,12]:
            if recalden:
                geo_num_masked = geo_num
                if neighbor_masks is not None:
                    geo_num_masked = mx.sym.elemwise_mul(neighbor_masks_expand, geo_num)
                geo_den_sum = mx.sym.sum(geo_num_masked, axis=P_dim, keepdims=True)  # B * 1 * O * 1 or B * O * 1 * 1
                geo_den_sum = mx.sym.clip(geo_den_sum, 1, 1000000) # make sure it's bigger than zero
            else:
                geo_den_sum = mx.sym.tile(mx.sym.expand_dims(center_den, axis=P_dim,
                    name=scope + "/centerdenexpand"), reps=(1, 1, 1, max_p_grid) if BN else (1, 1, max_p_grid, 1))
            geo_density = mx.sym.broadcast_div(geo_num, geo_den_sum, name=scope+"/den_ele_div")
    if attfdim <= 3:
        att_vec = geo_vec
    elif attfdim == 4:
        att_vec = mx.sym.concat(geo_dist,geo_vec, dim=C_dim, name=scope+"/attconcat")
    elif attfdim == 5:
        att_vec = mx.sym.concat(geo_dist,geo_vec, geo_global_density, dim=C_dim, name=scope+"/attconcat")
    elif attfdim == 10:
        att_vec = mx.sym.concat(geo_dist, geo_vec, centers_expand_xyz, neighbor_locs_xyz, dim=C_dim, name=scope+"/attconcat")
    elif attfdim == 11:
        att_vec = mx.sym.concat(geo_dist, geo_vec, centers_expand_xyz, neighbor_locs_xyz, geo_global_density, dim=C_dim, name=scope+"/attconcat")
    elif attfdim == 12:
        att_vec = mx.sym.concat(geo_dist, geo_vec, centers_expand_xyz, neighbor_locs_xyz, geo_density, geo_global_density, dim=C_dim, name=scope+"/attconcat")
    else:
        raise NotImplementedError
    #

    if configs["localfdim"] <= 3:
        geo_feats = geo_vec
    elif configs["localfdim"] == 4:
        geo_feats =  mx.sym.concat(geo_dist, geo_vec, dim=C_dim, name=scope+"/geo_concat")
    elif configs["localfdim"] == 5:
        geo_feats = mx.sym.concat(geo_dist, geo_vec, geo_global_density, dim=C_dim, name=scope+"/geo_concat")
    elif configs["localfdim"] == 10:
        geo_feats = mx.sym.concat(geo_dist, geo_vec, centers_expand_xyz, neighbor_locs_xyz, dim=C_dim, name=scope+"/geo_concat")
    elif configs["localfdim"] == 11:
        geo_feats = mx.sym.concat(geo_dist, geo_vec, centers_expand_xyz, neighbor_locs_xyz, geo_global_density, dim=C_dim, name=scope+"/geo_concat")
    elif configs["localfdim"] == 12:
        geo_feats = mx.sym.concat(geo_dist, geo_vec, centers_expand_xyz, neighbor_locs_xyz, geo_density, geo_global_density, dim=C_dim, name=scope+"/geo_concat")
    else:
        raise NotImplementedError

    if neighbor_feats is None:
        neighbor_feats = geo_feats
    elif configs["localfdim"] != 0:
        neighbor_feats = mx.symbol.concat(geo_feats, neighbor_feats, dim=C_dim, name=scope+"/neighbor_feats_concat")
    if not notFirstLayer and len(configs["elevation"]) > 0:
        if BN:
            neighbor_feats = mlp2d_c(neighbor_feats, configs["elevation"], use_bn=BN, bn_decay=bn_decay,
                                     scope=scope + "/elevation")
        else:
            neighbor_feats = fully_connected_mlp(neighbor_feats, configs["elevation"], dim=4, bn_decay=bn_decay,
                dropout_ratio=0, flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope + '/elevation/cov2fc', flip=False)

    contextvec = None
    if cntxt_mlp is not None:
        contextvec = contextvec_func(neighbor_feats, max_p_grid, cntxt_mlp = cntxt_mlp, bn_decay=bn_decay, scope=scope)

    if aggtype == "gcn":
        pair_feats = verts_pair_func(neighbor_feats, att_vec, B, max_o_grid, max_p_grid, N, C, contextvec=contextvec, att_full=att_full, pt_mlp_lst=pt_mlp_lst, scope=scope, bn_decay=bn_decay, neighbor_masks_expand=neighbor_masks_expand, att_norm=att_norm)  # B * max_o_grid * max_p_grid * C
        if neighbor_masks is not None and pool_type!="max" and pool_type!="max_pooling":
            pair_feats = mx.sym.broadcast_mul(pair_feats, neighbor_masks_expand, name="pairmask") # B * max_o_grid * max_p_grid * C
        agg_neigh_feats = aggregation_func(pair_feats, neighbor_masks_expand, geo_dist, max_o_grid, max_p_grid, pool_type=pool_type, scope=scope+"/center_aggregation", bn_decay=bn_decay) # B * max_o_grid * 1 * outDim[-1]
    elif aggtype == "agg_gcn":
        agg_neigh_feats = aggregation_func(neighbor_feats, neighbor_masks_expand, geo_dist, max_o_grid, max_p_grid,
            pool_type=pool_type, mlp_lst=pt_mlp_lst, scope=scope + "/neigh_aggregation", bn_decay=bn_decay)
    else:
        raise NotImplementedError("aggtype %s is not implemented" % aggtype)
    if center_ori_feats is not None:
        if len(center_dim) > 0:
            if BN:
                center_feats = mlp1d_c(center_ori_feats, center_dim, bn_decay=bn_decay, use_bn=BN, attr=None, scope=scope + '/centerfeats_conv1d')
            elif configs["use_bn"] == 'p':
                center_feats = fully_connected_mlp_withbn(center_ori_feats, center_dim, dim=3, bn_decay=bn_decay, dropout_ratio=0, flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope + '/centerfeats_conv1d1d', flip=False)
            else:
                center_feats = fully_connected_mlp(center_ori_feats, center_dim, dim = 3, bn_decay=bn_decay, dropout_ratio=0, flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope + '/centerfeats_conv1d1d', flip=False)
        else:
            center_feats = center_ori_feats
        if configs["up_center_inte"] == "add":
            agg_neigh_feats = center_feats + agg_neigh_feats
        elif configs["up_center_inte"] == "concat":
            agg_neigh_feats = mx.symbol.concat(center_feats, agg_neigh_feats, dim=1 if BN else 2, name=scope + "/agg_feats_concat")
    agg_feats = update_func(agg_neigh_feats, outDim=outDim, scope=scope + "/center_update", bn_decay=bn_decay)  # B * max_o_grid * outDim[-1]
    if center_masks is not None:
        agg_feats = mx.sym.broadcast_mul(agg_feats, mx.sym.expand_dims(center_masks, axis=1 if BN else 2, name=scope+"/maskexpand"), name=scope+"/centermask")  # B * outDim[-1] * max_o_grid
    return agg_feats


def contextvec_func(cntxt, max_p_grid, cntxt_mlp = [64,64], bn_decay=0.9, scope=""):
    print("cntxt_mlp:", cntxt_mlp)
    if len(cntxt_mlp)>0:
        if BN:
            cntxt = mlp2d_c(cntxt, cntxt_mlp, use_bn=BN, bn_decay=bn_decay, scope=scope+"/cnt")
        else:
            cntxt = fully_connected_mlp(cntxt, cntxt_mlp, dim = 4, bn_decay=bn_decay, dropout_ratio=0, flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope + '/cnt/cov2fc', flip=False)

    aggregated_cntxt = mx.sym.Pooling(name=scope + "/cnt/max_pooling", data=cntxt,
                                      kernel=(1, max_p_grid) if BN else (max_p_grid, 1), pool_type="max", layout="NCHW")
    aggregated_cntxt = mx.sym.tile(aggregated_cntxt, reps=(1, 1, 1, max_p_grid) if BN else (1, 1, max_p_grid, 1))
    return aggregated_cntxt


def mlp_mask_and_pool(data, mlp_list, max_p_grid, pool_type="max", use_bn=BN, bn_decay=0.9, mask=None, scope=''):
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
    if BN:
        data = mlp2d_c(data, mlp_list, use_bn=use_bn, bn_decay=bn_decay, scope=scope)
    else:
        data = fully_connected_mlp(data, mlp_list, dim = 4, bn_decay=configs["bn_decay"], dropout_ratio=0,
            flatten=False, use_bn=BN, use_relu=True, bn_axis=1, attr=None, scope=scope, flip=False)
    # pooling
    if mask is not None and pool_type!="max" and pool_type!="max_pooling":
        data = mx.sym.broadcast_mul(data, mask, axis=1, name="pairmask")  # B * C * max_o_grid * max_p_grid
    # data = mx.symbol.max(data, axis=2, name=scope+'/pool')
    if pool_type == "max_pooling":
        data = mx.sym.Pooling(name=scope + "/max_pooling", data=data,
                       kernel=(1, max_p_grid), pool_type="max")
        return mx.sym.squeeze(data,  axis=3)
    elif pool_type == "max":
        data = mx.symbol.max(data, axis=3, name=scope + '/max')
        return data
    else:
        raise NotImplementedError("agg %s is no implemented" % pool_type)
