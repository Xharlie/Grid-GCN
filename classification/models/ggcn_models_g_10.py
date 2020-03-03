"""
PointNet++ model for ModelNet40 classification and ScanNet semantic labelling
"""

import mxnet as mx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from configs.configs_10 import configs
from models.gcn_module_g_10 import sub_g_update
from utils.ops import fully_connected, batch_take, batch_take_g
from mxnet.symbol import Gridify

BN = False if (configs["use_bn"] == 'f' or configs["use_bn"] == 'p') else True
C_dim = 1 if BN else 2

def get_cls_inputs():
    data = mx.symbol.Variable(name='data')
    actual_centnum = mx.symbol.Variable(name='actual_centnum')
    label = mx.symbol.Variable(name='label')
    return data, actual_centnum, label

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

def get_symbol_cls_ggcn(B, N, take_shapes, gcn_outDim=[[64, 128, 256], [256, 512, 1024]], bn_decay=0.9, weights=None):
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

    data, actual_centnum, label = get_cls_inputs()
    data = mx.sym.concat(data, mx.sym.ones((B, N, 1), dtype="float32"), dim=2)
    # get features and sub GCN
    # center_locnfeat_alllayers = [data]
    # if BN:
    #     data_trans = data.transpose(axes=(0, 2, 1), name="up_center_tpose")
    #     center_locnfeat_alllayers.append(data_trans)
    # else:
    #     center_locnfeat_alllayers.append(data)

    data_loc = data
    for i in range(len(configs["voxel_size_lst"])):
        take_shapes[i][-1] += 4
        # data_loc = data_loc.reshape((B, configs["num_points"], 4))
        if configs["group_all"] and i == len(configs["voxel_size_lst"])-1:
            if BN:
                centers_xyz = mx.sym.zeros((B, 3, 1), dtype="float32")
                centers_den = mx.sym.sum(centers_den, axis=2, keepdims=True)
                neighbors = data.expand_dims(axis=2)
            else:
                centers_xyz = mx.sym.zeros((B, 1, 3), dtype="float32")
                centers_den = mx.sym.sum(centers_den, axis=1, keepdims=True)
                neighbors = data.expand_dims(axis=1)
            nebidxmsk = centmsk.expand_dims(axis=1)
            centmsk =  mx.sym.ones((B, 1), dtype="float32")
        else:
            nebidx, nebidxmsk, centers, centmsk, actual_centnum \
                = Gridify(data_loc, actual_centnum, max_o_grid=configs["max_o_grid_lst"][i],
                          max_p_grid=configs["max_p_grid_lst"][i],
                          kernel_size=configs["kernel_size_lst"][i], stride=configs['stride_lst'][i],
                          coord_shift=configs["lidar_coord"], voxel_size=configs['voxel_size_lst'][i],
                          grid_size=configs['grid_size_lst'][i], loc=1 if configs["loc_within"] else 0)
            data_loc = centers
            centers_xyz = mx.sym.slice_axis(centers, axis=2, begin=0, end=3)
            centers_den = mx.sym.slice_axis(centers, axis=2, begin=3, end=4)
            if BN:
                if i > 0:
                    data = data.transpose(axes=(0, 2, 1),
                                          name= "/down_batch_take_g_pre_tpose{}".format(i))
                neighbors = batch_take_g(data, nebidx, take_shapes[i], scope='down_neigh_batch_take_'+str(i))
                neighbors = neighbors.transpose(axes=(0, 3, 1, 2),
                                name="/down_batch_take_g_post_tpose{}".format(i))
                centers = centers.transpose(axes=(0,2,1), name="center_tpose"+str(i))
                centers_xyz = centers_xyz.transpose(axes=(0,2,1), name="center_xyz_tpose"+str(i))
                centers_den = centers_den.transpose(axes=(0,2,1), name="center_den_tpose"+str(i))
            else:
                neighbors = batch_take_g(data, nebidx, take_shapes[i], scope='down_neigh_batch_take_' + str(i))
        pt_mlp_lst = configs["pt_ele_dim"][i] if configs["pt_ele_dim"] is not None else None
        att_ele_lst = configs["att_ele_dim"][i] if configs["att_ele_dim"] is not None else None
        # print("gcn_outDim i", gcn_outDim[i])
        center_feats = sub_g_update(centers_xyz, centers_den, neighbors, i > 0, centmsk, nebidxmsk, configs["attfdim"], pt_mlp_lst=pt_mlp_lst, att_ele_lst=att_ele_lst, outDim=gcn_outDim[i], cntxt_mlp= None if configs["cntxt_mlp_lst"] is None else configs["cntxt_mlp_lst"][i] , shape = take_shapes[i], scope='sub_g_'+str(i), aggtype=configs["aggtype"], pool_type=configs["agg"], att_full=configs["att_full"], recalden=False, bn_decay=bn_decay)

        data = mx.sym.concat(centers, center_feats, dim=C_dim, name="down_concat_datalayer" + str(i))
        # center_locnfeat_alllayers.append(data)

    # classification
    net = get_cls_head(center_feats, label, bn_decay=bn_decay, weights=weights)
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

