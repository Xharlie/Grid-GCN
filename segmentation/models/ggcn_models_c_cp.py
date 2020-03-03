"""
PointNet++ model for ModelNet40 classification and ScanNet semantic labelling
"""

import mxnet as mx
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from configs.configs import configs
from models.gcn_module_c import sub_g_update
from utils.ops import batch_take_halfc, batch_take_c, conv1d, batch_take
from models.pnt2_module import pointnet_sa_module, grouping, sampling, pointnet_fp_module
import custom_op.weighted_gradient

BN = False if (configs["use_bn"] == 'f' or configs["use_bn"] == 'p') else True
C_dim = 1 if BN else 2

def get_seg_inputs(layer_num=None):
    data = mx.symbol.Variable(name='data')
    label = mx.symbol.Variable(name='label')
    if layer_num is not None and not configs["fps"]:
        neighbors_arr_lst = [None for i in range(layer_num)]
        neighbors_mask_lst = [None for i in range(layer_num)]
        neighbors_arr_up_lst = [None for i in range(layer_num)]
        neighbors_mask_up_lst = [None for i in range(layer_num)]
        centers_arr_lst = [None for i in range(layer_num)]
        centers_mask_lst = [None for i in range(layer_num)]
        for i in range(layer_num):
            neighbors_arr_lst[i]= mx.symbol.Variable(name='neighbors_arr' + str(i))

            if configs["agg"] not in ["max","max_pooling"] or configs["neigh_fetch"] > 0:
                neighbors_mask_lst[i]=mx.symbol.Variable(name='neighbors_mask' + str(i))

            if configs["up_type"] != "inter":
                neighbors_arr_up_lst[i] = mx.symbol.Variable(name='neighbors_arr_up' + str(i))
                if configs["up_agg"] not in ["max", "max_pooling"] or configs["up_neigh_fetch"] > 0:
                    neighbors_mask_up_lst[i]= mx.symbol.Variable(name='neighbors_mask_up' + str(i))

            centers_arr_lst[i]=mx.symbol.Variable(name='centers_arr' + str(i))
            centers_mask_lst[i]=mx.symbol.Variable(name='centers_mask' + str(i))
        return data, label, neighbors_arr_lst, neighbors_mask_lst,  neighbors_arr_up_lst, neighbors_mask_up_lst, centers_arr_lst, centers_mask_lst
    return data, label, None, None, None, None

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
    data, label, neighbors_arr_lst, neighbors_mask_lst, neighbors_arr_up_lst, neighbors_mask_up_lst, \
        centers_arr_lst, centers_mask_lst = get_seg_inputs(layer_num=len(configs['gcn_outDim']))
    # get features and sub GCN
    center_locnfeat_alllayers = []
    up_center_locnfeat_alllayers = []
    pt_mlp_lsts = configs["pt_ele_dim"]
    max_o_grid_lst = configs["max_o_grid_lst"]
    data = mx.sym.concat(data, mx.sym.ones((B, N, 1),dtype="float32"), dim=2)
    data_layer = data
    centers_reverse_lst = []
    for i in range(len(neighbors_arr_lst)):
        take_shapes[i][-1]+=4
        if BN:
            neighbors = batch_take_halfc(data_layer, neighbors_arr_lst[i], take_shapes[i], scope='neigh_batch_take_'+str(i)) if i == 0 \
                        else batch_take_c(data_layer, neighbors_arr_lst[i], take_shapes[i], scope='neigh_batch_take_c_'+str(i))
        else:
            neighbors = batch_take(data_layer, neighbors_arr_lst[i], take_shapes[i], scope='neigh_batch_take_' + str(i))
        neighbor_masks = neighbors_mask_lst[i]
        centers = centers_arr_lst[i]
        if BN:
            centers = centers.transpose(axes=(0,2,1), name="center_tpose"+str(i))
        centers_reverse_lst.append(centers)
        center_masks = centers_mask_lst[i]
        pt_mlp_lst = pt_mlp_lsts[i] if pt_mlp_lsts is not None else None
        center_feats = sub_g_update(centers, neighbors, i>0, center_masks, neighbor_masks, configs["attfdim"],
            pt_mlp_lst=pt_mlp_lst, outDim=configs['gcn_outDim'][i], shape = take_shapes[i], scope='sub_g_'+str(i),
            aggtype=configs["aggtype"], pool_type=configs["agg"], neighbor_num=configs["neigh_fetch"],
            att_full= configs["att_full"])
        data_layer = mx.sym.concat(centers, center_feats, dim=C_dim, name="down_concat_datalayer"+str(i))
        center_locnfeat_alllayers.append(data_layer)
    # up sampling part
    up_pt_mlp_lsts = configs["up_pt_ele_dim"]
    up_inputDim = configs["up_inputDim"]
    up_max_o_grid_lst = configs["up_max_o_grid_lst"]

    if configs["up_type"] == "inter":
        f_last_layer = mx.sym.slice_axis(center_locnfeat_alllayers[-1], axis=C_dim, begin=4, end=None)
        for i in range(len(configs['up_gcn_outDim'])):
            f_this_layer = mx.sym.slice_axis(center_locnfeat_alllayers[-i - 2], axis=C_dim, begin=4, end=None) if i != len(configs['up_gcn_outDim'])-1 else None
            centeroids_last = mx.sym.slice_axis(centers_reverse_lst[-i-1], axis=C_dim, begin=0, end=3)
            centeroids_this = mx.sym.slice_axis(centers_reverse_lst[-i-2], axis=C_dim, begin=0, end=3)\
                if i != len(configs['up_gcn_outDim'])-1 else mx.sym.slice_axis(data, axis=C_dim, begin=0, end=3)
            f_last_layer = pointnet_fp_module(centeroids_last, f_last_layer, centeroids_this, f_this_layer,
                shape=(B, up_max_o_grid_lst[i], up_max_o_grid_lst[i-1] if i != 0 else max_o_grid_lst[-1],
                up_inputDim[i]), mlp_list=up_pt_mlp_lsts[i], bn_decay=bn_decay, scope='uplayer_'+str(i))
            up_center_locnfeat_alllayers.append(f_last_layer)
    elif configs["up_type"] == "grid_full":
        centers = data.transpose(axes=(0, 2, 1), name="up_center_tpose")
        center_masks = None
        for i in range(len(neighbors_arr_up_lst)):
            take_up_shapes[i][-1] = take_up_shapes[i][-1] + 4
            f_last_layer = center_locnfeat_alllayers[-1 - i]
            if BN:
                neighbors_up = batch_take_c(f_last_layer, neighbors_arr_up_lst[i], take_up_shapes[i],
                                      scope='neigh_up_batch_c_take_' + str(i))
            else:
                neighbors_up = batch_take(f_last_layer, neighbors_arr_lst[i], take_up_shapes[i], scope='neigh_up_batch_take_' + str(i))
            neighbor_masks_up = neighbors_mask_up_lst[i]
            up_pt_mlp_lst = up_pt_mlp_lsts[i] if up_pt_mlp_lsts is not None else None
            print("uplayer {}, take_up_shapes: {}".format(i, take_up_shapes[i]))
            center_feats_up = sub_g_update(centers, neighbors_up, True, center_masks,
                neighbor_masks_up, configs["up_attfdim"], center_ori_feats=None, pt_mlp_lst=up_pt_mlp_lst,
                outDim=configs['up_gcn_outDim'][i], shape=take_up_shapes[i], scope='up_sub_g_' + str(i),
                aggtype=configs["up_aggtype"], pool_type=configs["up_agg"], neighbor_num=configs["up_neigh_fetch"],
                att_full= configs["up_att_full"])
            up_center_locnfeat_alllayers.append(center_feats_up)
        f_last_layer = mx.sym.concat(up_center_locnfeat_alllayers[0], up_center_locnfeat_alllayers[1],
                                     up_center_locnfeat_alllayers[2], up_center_locnfeat_alllayers[3], dim=C_dim)
    else:
        f_last_layer = center_locnfeat_alllayers[-1]
        if BN:
            data = data.transpose(axes=(0, 2, 1), name="up_center_tpose")
        for i in range(len(neighbors_arr_up_lst)):
            take_up_shapes[i][-1] = take_up_shapes[i][-1] + 4
            f_this_layer = center_locnfeat_alllayers[-i - 2] if i != len(configs['up_gcn_outDim']) - 1 else data
            if BN:
                neighbors_up = batch_take_c(f_last_layer, neighbors_arr_up_lst[i], take_up_shapes[i],
                                            scope='neigh_up_batch_take_c_' + str(i))
            else:
                neighbors_up = batch_take(f_last_layer, neighbors_arr_up_lst[i], take_up_shapes[i], scope='neigh_up_batch_take_' + str(i))
            neighbor_masks_up = neighbors_mask_up_lst[i]
            centers = centers_reverse_lst[-i-2] if i != len(configs['up_gcn_outDim'])-1 else data
            center_masks = centers_mask_lst[-i-2] if i != len(configs['up_gcn_outDim'])-1 else None
            up_pt_mlp_lst = up_pt_mlp_lsts[i] if up_pt_mlp_lsts is not None else None
            print("uplayer {}, take_up_shapes: {}".format(i, take_up_shapes[i]))
            center_feats_up = sub_g_update(centers, neighbors_up, True, center_masks, neighbor_masks_up,
                configs["up_attfdim"], center_ori_feats = f_this_layer, pt_mlp_lst=up_pt_mlp_lst, outDim=configs['up_gcn_outDim'][i],
                shape = take_up_shapes[i], scope='up_sub_g_'+str(i), aggtype=configs["up_aggtype"], pool_type=configs["up_agg"],
                neighbor_num=configs["up_neigh_fetch"], att_full= configs["up_att_full"], center_dim=configs["up_center_dim"][i])
            up_center_locnfeat_alllayers.append(center_feats_up)
            f_last_layer = mx.sym.concat(centers, center_feats_up, dim=C_dim, name="up_concat_f_last_layer"+str(i))
    # segmentation head
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

