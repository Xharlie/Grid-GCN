"""
PointNet++ model for ModelNet40 classification and ScanNet semantic labelling
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
# import ctypes
# _ = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../gridifyop/additional.so'))
# print(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../gridifyop/additional.so'))
import mxnet as mx
from configs.configs import configs
from models.gcn_module_g_att import sub_g_update
from utils.ops import batch_take_g, batch_take_c, conv1d, batch_take
import custom_op.weighted_gradient
from mxnet.symbol import Gridify
from mxnet.symbol import GridifyUp


BN = False if (configs["use_bn"] == 'f' or configs["use_bn"] == 'p') else True
C_dim = 1 if BN else 2

def get_seg_inputs():
    data = mx.symbol.Variable(name='data')
    actual_centnum = mx.symbol.Variable(name='actual_centnum')
    label = mx.symbol.Variable(name='label')
    return data, actual_centnum, label

def get_seg_head(features, label, dropout_ratio=0.5, bn_decay=0.9, weights=None):
    """
    Get symbol for ScanNet semantic labelling
    """
    net = conv1d(features, num_filter=128, kernel=(1,), stride=(1,), bn_decay=bn_decay, layout='NCW', scope='fc1')
    if dropout_ratio > 0:
        net = mx.symbol.Dropout(net, p=dropout_ratio, name='fc1/dropout')
    net = conv1d(net, num_filter=21, kernel=(1,), stride=(1,), layout='NCW', use_bn=False, use_relu=False, scope='fc2')
    if weights is not None:
        print("weights is not None")
        net = mx.symbol.Custom(data=net, weight=weights, input_dim=3, name='fc2/weighted', op_type='weighted_gradient')
    net = mx.symbol.SoftmaxOutput(data=net, label=label, use_ignore=True, ignore_label=0, multi_output=True, normalization='valid', name='pred')
    # return net
    return net

def debug_get_seg_head(features, label, dropout_ratio=0.5, bn_decay=0.9, weights=None):
    """
    Get symbol for ScanNet semantic labelling
    """
    net = conv1d(features, num_filter=128, kernel=(1,), stride=(1,), bn_decay=bn_decay, layout='NCW', scope='fc1')
    if dropout_ratio > 0:
        net = mx.symbol.Dropout(net, p=dropout_ratio, name='fc1/dropout')
    num_net = conv1d(net, num_filter=21, kernel=(1,), stride=(1,), layout='NCW', use_bn=False, use_relu=False, scope='fc2')
    if weights is not None:
        print("weights is not None")
        num_net = mx.symbol.Custom(data=num_net, weight=weights, input_dim=3, name='fc2/weighted', op_type='weighted_gradient')
    net = mx.symbol.SoftmaxOutput(data=num_net, label=label, use_ignore=True, ignore_label=0, multi_output=True, normalization='valid', name='pred')
    # return net
    return mx.symbol.Group([net, num_net, features])

def neighbors_fetch(downdata, downnum, updata, upnum, shape, k=3, scope='', radius=1):
    """
    k nearest neighbor
    downdata: symbol, shape=(B, N, 3)
    downnum: symbol, shape=(B, 1)
    centers: symbol, shape=(B, O, 3)
    upnum: symbol, shape=(B, 1)
    shape: specify (M, N)
    scope: name scope for this operator
    returns:
        dist, shape=(B, M, k), squared distances to the k nearest points
        ids, shape=(B, M, k), dtype=int32, indices of the k nearest points
    """

    if "real_knn" in configs.keys() and configs["real_knn"] :
        print("global knn!!!!!")

        if k == 3:
            # query: (B, M, 3), Symbol, locations to interpolate
            # data: (B, N, 3), Symbol, locations of existing points

            _, nebidx = mx.sym.contrib.ThreeNN(updata, downdata)  # (B, M, k)
        else:
            nebidx = mx.sym.contrib.KNN(updata, downdata, downnum, upnum, k = k)
    else:
        nebidx = mx.sym.contrib.BallKNN(updata, downdata, downnum, upnum, k = k, radius=radius)
        # nebidx = mx.sym.contrib.KNN(updata, downdata, downnum, upnum, k = k)  # (B, M, k)

    return nebidx, None

def knn(query, data, shape, k=3, scope=''):
    """
    k nearest neighbor
    query: symbol, shape=(B, M, 3), query points
    data: symbol, shape=(B, N, 3), all points
    shape: specify (M, N)
    scope: name scope for this operator
    returns:
        dist, shape=(B, M, k), squared distances to the k nearest points
        ids, shape=(B, M, k), dtype=int32, indices of the k nearest points
    """
    M, N = shape
    query = query.expand_dims(axis=2)  # (B, M, 1, 3)
    data = data.expand_dims(axis=1)  # (B, 1, N, 3)
    diff = mx.symbol.broadcast_sub(query, data)  # (B, M, N, 3)
    all_dist = mx.symbol.sum(diff*diff, axis=3)  # (B, M, N)
    ids = mx.symbol.topk(all_dist, axis=2, k=k, ret_typ='indices', is_ascend=True, name='{}_top{}'.format(scope, k))   # (B, M, k)
    return ids.astype('int32')


def get_symbol_seg_ggcn(B, N, take_shapes, take_up_shapes, bn_decay=0.9, weights=None):
    """
    Get symbol for GGCN ModelNet40 classification
    B: batch_size
    N: #points in each batch
    weights: weights for weighted_gradient
    The model has inputs:
      data: (B, N, 3), point clouds
      'neighbors_arr' + str(i), (self.batch_size, self.max_o_grid[i], self.max_p_grid[i])
      'neighbors_mask' + str(i), (self.batch_size, self.max_o_grid[i], self.max_p_grid[i])
      'centers_arr' + str(i), (self.batch_size, self.max_o_grid[i], 3)),
      'centers_mask' + str(i), (self.batch_size, self.max_o_grid[i]
      label: (B,), ground truth labels
    returns: symbol, (B, 40)
    """
    # inputs
    data, actual_centnum, label = get_seg_inputs()
    # get features and sub GCN
    center_locnfeat_alllayers = []
    up_center_locnfeat_alllayers = []
    pt_mlp_lsts = configs["pt_ele_dim"]
    max_o_grid_lst = configs["max_o_grid_lst"]
    up_max_p_grid_lst = configs["up_max_p_grid_lst"]
    centers_xyz_reverse_lst = [data]
    if BN:
        centers_xyz_bn_reverse_lst = [data.transpose(axes=(0,2,1), name="dataxyz_tpose")]
        centers_den_bn_reverse_lst = [mx.sym.ones((B, 1, N), dtype="float32")]
        data = mx.sym.concat(data, mx.sym.ones((B, N, 1),dtype="float32"), dim = 2)
    else:
        centers_xyz_bn_reverse_lst = [data]
        centers_den_bn_reverse_lst = [mx.sym.ones((B, N, 1),dtype="float32")]
        data = mx.sym.concat(data, centers_den_bn_reverse_lst[0], dim=2)
    data_layer = data
    centers_reverse_lst = [data]
    centers_mask_lst = []
    centnum_lst = [actual_centnum]
    if BN:
        data_trans = data.transpose(axes=(0, 2, 1), name="up_center_tpose")
        center_locnfeat_alllayers.append(data_trans)
    else:
        center_locnfeat_alllayers.append(data)
    data_loc = data
    for i in range(len(max_o_grid_lst)):
        take_shapes[i][-1]+=4
        nebidx, nebidxmsk, centers, centmsk, actual_centnum \
            = Gridify(data_loc, actual_centnum, max_o_grid=max_o_grid_lst[i],
                max_p_grid=configs["max_p_grid_lst"][i],
                kernel_size=configs["kernel_size_lst"][i], stride=configs['stride_lst'][i],
                coord_shift=configs["lidar_coord"], voxel_size=configs['voxel_size_lst'][i],
                grid_size=configs['grid_size_lst'][i], loc=1 if configs["loc_within"] else 0)
        data_loc = centers
        centers_xyz = mx.sym.slice_axis(centers, axis=2, begin=0, end=3)
        centers_den = mx.sym.slice_axis(centers, axis=2, begin=3, end=4)
        centers_xyz_reverse_lst.append(centers_xyz)
        centers_reverse_lst.append(data_loc)
        centers_mask_lst.append(centmsk)
        centnum_lst.append(actual_centnum)

        if BN:
            if i > 0:
                data_layer = data_layer.transpose(axes=(0, 2, 1),
                                      name= "/down_batch_take_g_pre_tpose{}".format(i))
            neighbors = batch_take_g(data_layer, nebidx,
                            take_shapes[i], scope='down_neigh_batch_take_'+str(i))
            neighbors = neighbors.transpose(axes=(0, 3, 1, 2),
                            name="/down_batch_take_g_post_tpose{}".format(i))
            centers = centers.transpose(axes=(0,2,1), name="center_tpose"+str(i))
            centers_xyz = centers_xyz.transpose(axes=(0,2,1), name="center_xyz_tpose"+str(i))
            centers_den = centers_den.transpose(axes=(0,2,1), name="center_den_tpose"+str(i))
        else:
            neighbors = batch_take_g(data_layer, nebidx, take_shapes[i], scope='down_neigh_batch_take_' + str(i))
        centers_xyz_bn_reverse_lst.append(centers_xyz)
        centers_den_bn_reverse_lst.append(centers_den)

        pt_mlp_lst = pt_mlp_lsts[i] if pt_mlp_lsts is not None else None
        center_feats = sub_g_update(centers_xyz, centers_den, neighbors, i>0, centmsk, nebidxmsk, configs["attfdim"], pt_mlp_lst=pt_mlp_lst, outDim=configs['gcn_outDim'][i], cntxt_mlp= configs["cntxt_mlp_lst"][i] if configs["cntxt_mlp_lst"] is not None else None, shape = take_shapes[i], scope='sub_g_'+str(i), aggtype=configs["aggtype"], pool_type=configs["agg"], att_full= configs["att_full"], recalden = False, bn_decay=bn_decay)
        data_layer = mx.sym.concat(centers, center_feats, dim=C_dim, name="down_concat_datalayer"+str(i))
        center_locnfeat_alllayers.append(data_layer)

    # up sampling part
    f_last_layer = center_locnfeat_alllayers[-1]
    for i in range(len(up_max_p_grid_lst)):
        take_up_shapes[i][-1] = take_up_shapes[i][-1] + 4
        downdata = centers_reverse_lst[-i-1]
        downdata_xyz = centers_xyz_reverse_lst[-i-1]
        centers = centers_reverse_lst[-i-2]
        updata_xyz = centers_xyz_reverse_lst[-i-2]
        centers_xyz = centers_xyz_bn_reverse_lst[-i-2]
        centers_den = centers_den_bn_reverse_lst[-i-2]
        downnum = centnum_lst[-i-1]
        upnum = centnum_lst[-i-2]
        if configs["up_neigh_fetch"]:
            nebidx, nebidxmsk = \
                neighbors_fetch(downdata_xyz, downnum, updata_xyz, upnum,
                    (configs["up_max_o_grid_lst"][i], configs["max_o_grid_lst"][-i-1]), k=configs["up_max_p_grid_lst"][i], scope='up_{}/up_neigh_fetch'.format(i), radius=configs["up_voxel_size_lst"][i][0] * configs["up_kernel_size_lst"][i] * 1.7 / 2)
        else:
            nebidx, nebidxmsk = \
                GridifyUp(downdata, centers, downnum, upnum, max_p_grid=configs["up_max_p_grid_lst"][i],
                max_o_grid=configs["up_max_o_grid_lst"][i], kernel_size=configs["up_kernel_size_lst"][i],
                coord_shift=configs["lidar_coord"], voxel_size=configs['up_voxel_size_lst'][i],
                grid_size=configs['up_grid_size_lst'][i])

        f_this_layer = center_locnfeat_alllayers[-i - 2]
        if BN:
            f_last_layer = f_last_layer.transpose(axes=(0, 2, 1),
                                  name="/up_batch_take_g_pre_tpose{}".format(i))

            neighbors_up = batch_take_g(f_last_layer, nebidx, take_up_shapes[i],
                                        scope='up_neigh_up_batch_take_g_' + str(i))

            neighbors_up = neighbors_up.transpose(axes=(0, 3, 1, 2),
                            name="/up_batch_take_g_post_tpose{}".format(i))
        else:
            neighbors_up = batch_take_g(f_last_layer, nebidx, take_up_shapes[i], scope='up_neigh_up_batch_take_g' + str(i))
        center_masks = centers_mask_lst[-i-2] if i != len(configs['up_gcn_outDim'])-1 else None
        up_pt_mlp_lst = configs["up_pt_ele_dim"][i] if configs["up_pt_ele_dim"] is not None else None
        print("uplayer {}, take_up_shapes: {}".format(i, take_up_shapes[i]))
        if BN:
            centers = centers.transpose(axes=(0, 2, 1), name="center_tpose" + str(i))
        center_feats_up = sub_g_update(centers_xyz, centers_den, neighbors_up, True, center_masks, nebidxmsk, configs["up_attfdim"], center_ori_feats = f_this_layer, pt_mlp_lst=up_pt_mlp_lst, outDim=configs['up_gcn_outDim'][i], cntxt_mlp= configs["up_cntxt_mlp_lst"][i] if configs["up_cntxt_mlp_lst"] is not None else None, shape = take_up_shapes[i], scope='up_sub_g_'+str(i), aggtype=configs["up_aggtype"], pool_type=configs["up_agg"],att_full= configs["up_att_full"], center_dim=configs["up_center_dim"][i], recalden = True, bn_decay=bn_decay)
        up_center_locnfeat_alllayers.append(center_feats_up)
        f_last_layer = mx.sym.concat(centers, center_feats_up, dim=C_dim, name="up_concat_f_last_layer"+str(i))
    # segmentation head
    f_last_layer = up_center_locnfeat_alllayers[-1] # noloc
    if not BN:
        f_last_layer = f_last_layer.transpose(axes=(0, 2, 1), name="last_f_transpose")
    net = get_seg_head(f_last_layer, label, bn_decay=bn_decay, weights=weights)
    return net

if __name__ == "__main__":
    '''
    #net = get_symbol_cls_ssg(B=64, N=1024)
    net = get_symbol_seg(B=64, N=1024)
    t = mx.viz.plot_network(net)
    t.render()
    '''

