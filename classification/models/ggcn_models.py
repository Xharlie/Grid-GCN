"""
PointNet++ model for ModelNet40 classification and ScanNet semantic labelling
"""

import mxnet as mx
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from configs.configs import configs
from models.gcn_module import sub_g_update
from utils.ops import fully_connected, batch_take
from models.pnt2_module import pointnet_sa_module, grouping, sampling


def get_cls_inputs(layer_num=None):
    data = mx.symbol.Variable(name='data')
    label = mx.symbol.Variable(name='label')
    if layer_num is not None and not configs["fps"]:
        neighbors_arr_lst = []
        neighbors_mask_lst = []
        centers_arr_lst = []
        centers_mask_lst = []
        for i in range(layer_num):
            neighbors_arr_lst.append(mx.symbol.Variable(name='neighbors_arr' + str(i)))
            if configs["agg"] in ["max","max_pooling"]:
                neighbors_mask_lst.append(None)
            else:
                neighbors_mask_lst.append(mx.symbol.Variable(name='neighbors_mask' + str(i)))
            centers_arr_lst.append(mx.symbol.Variable(name='centers_arr' + str(i)))
            centers_mask_lst.append(mx.symbol.Variable(name='centers_mask' + str(i)))
        return data, label, neighbors_arr_lst, neighbors_mask_lst, centers_arr_lst, centers_mask_lst
    return data, label, None, None, None, None

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
    data, label, neighbors_arr_lst, neighbors_mask_lst, centers_arr_lst, centers_mask_lst \
        = get_cls_inputs(layer_num=len(gcn_outDim))
    # get features and sub GCN
    center_locnfeat_alllayers = [data]
    pt_mlp_lsts = configs["pt_ele_dim"]
    data = mx.sym.concat(data, mx.sym.ones((B, N, 1),dtype="float32"), dim=2)
    for i in range(len(neighbors_arr_lst)):
        take_shapes[i][-1]+=4
        neighbors = batch_take(data, neighbors_arr_lst[i], take_shapes[i], scope='neigh_batch_take_'+str(i))
        # neighbor_locs = mx.sym.slice_axis(neighbors, axis=3, begin=0, end=4)  # get xyz of data points
        # neighbor_feats = mx.sym.slice_axis(neighbors, axis=3, begin=4, end=None) if i>0 else None
        neighbor_masks = neighbors_mask_lst[i]
        centers = centers_arr_lst[i]
        center_masks = centers_mask_lst[i]
        pt_mlp_lst = pt_mlp_lsts[i] if pt_mlp_lsts is not None else None
        # print("gcn_outDim i", gcn_outDim[i])
        # center_feats = sub_g_update(centers, neighbor_feats, neighbor_locs, center_masks, neighbor_masks,
        #     pt_mlp_lst=pt_mlp_lst, outDim=gcn_outDim[i], shape = take_shapes[i], scope='sub_g_'+str(i), aggtype=configs["aggtype"])

        centers_xyz = mx.sym.slice_axis(centers, axis=2, begin=0, end=3)
        centers_den = mx.sym.slice_axis(centers, axis=2, begin=3, end=4)
        center_feats = sub_g_update(centers_xyz, centers_den, neighbors, i > 0, center_masks, neighbor_masks, configs["attfdim"],
                                    pt_mlp_lst=pt_mlp_lst, outDim=gcn_outDim[i], shape=take_shapes[i],
                                    scope='sub_g_' + str(i), aggtype=configs["aggtype"], pool_type=configs["agg"],
                                    att_full=configs["att_full"], recalden=False)
        center_locnfeat_alllayers.append(center_feats)
        data = mx.sym.concat(centers, center_feats, dim=2)
    # classification
    net = get_cls_head(center_locnfeat_alllayers[-1], label, bn_decay=bn_decay, weights=weights)
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
    # PointNet++ set abstractionF
    centeroids, features = pointnet_sa_module(data, None, shape=(B, N, 0), fps_npoints=512, radius_list=0.2, rs_npoints_list=rs_npoints[0], xyz_mlp=[32,32], mlp_list=[64,64,128], bn_decay=bn_decay, scope='layer1')
    centeroids, features = pointnet_sa_module(centeroids, features, shape=(B, 512, 128), fps_npoints=128, radius_list=0.4, rs_npoints_list=rs_npoints[1], xyz_mlp=[32,32], mlp_list=[128,128,256], bn_decay=bn_decay, scope='layer2')
    return centeroids, features

def get_symbol_cls_fps_gcn(B, N, shapes, gcn_outDim=[[64, 128, 256], [256, 512, 1024]], bn_decay=0.9, weights=None):
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
    data, label, neighbors_arr_lst, neighbors_mask_lst, centers_arr_lst, centers_mask_lst \
        = get_cls_inputs(layer_num=len(gcn_outDim))
    # get features and sub GCN
    center_locnfeat_alllayers = [data]

    rs_npoints = [32, 64, 128]
    fps_npoints_lst = [512, 128, 1]
    pt_num = [N, fps_npoints_lst[0], fps_npoints_lst[1]]
    radius_list_lst = [0.2, 0.4, 2.0]
    neighbor_locs = mx.sym.slice_axis(data, axis=2, begin=0, end=3)
    neighbor_feats = None
    take_shapes = [[B, fps_npoints_lst[0], rs_npoints[0], pt_num[0], shapes[0][-1]],
                   [B, fps_npoints_lst[1], rs_npoints[1], pt_num[1], shapes[1][-1]],
                   [B, fps_npoints_lst[2], rs_npoints[2], pt_num[2], shapes[2][-1]]]  # B, max_o_grid, max_p_grid, N, C
    pt_mlp_lsts = configs["pt_ele_dim"]
    for i in range(len(fps_npoints_lst)):
        if i == len(fps_npoints_lst)-1:
            centers = mx.sym.zeros((B, fps_npoints_lst[i], 3))
        else:
            centers = sampling(neighbor_locs, (B, pt_num[i], shapes[i][-1]), method='fps', fps_npoints=fps_npoints_lst[i], scope='sample_layer'+str(i+1))
        # centers = centers.reshape((B, fps_npoints_lst[i], 3), name="centerreshape_"+str(i))
        neighbor_locs, neighbor_feats = grouping(neighbor_locs, neighbor_feats, centers, (B, pt_num[i], shapes[i][-1]), fps_npoints_lst[i], rs_npoints[i],
                                   radius=radius_list_lst[i], xyz_mlp=[], scope='{}/{}'.format('group_layer', i+1), conxyz=False)
        # neighbor_locs = neighbor_locs.reshape((B, fps_npoints_lst[i], rs_npoints[i], 3), name="neighbor_locs_"+str(i))
        # neighbor_feats = neighbor_feats.reshape((B, fps_npoints_lst[i], rs_npoints[i], take_shapes[i][-1]), name="neighbor_feats_"+str(i))
        appendix = "/1" if i < 2 else ""
        center_feats = sub_g_update(centers, neighbor_feats, neighbor_locs, None, None, pt_mlp_lst = pt_mlp_lsts[i],
            outDim=gcn_outDim[i], shape = take_shapes[i], scope='layer'+str(i+1)+appendix, no_mask=True)
        # if i != len(fps_npoints_lst)-1:
        #     center_feats = center_feats.reshape((B, fps_npoints_lst[i], shapes[i+1][-1]), name="center_feats_"+str(i))
        neighbor_locs, neighbor_feats = centers, center_feats
        data = mx.symbol.concat(neighbor_locs, neighbor_feats, dim=2)  # (B, fps_npoints, CC+C)
        center_locnfeat_alllayers.append(neighbor_feats)
    # classification
    net = get_cls_head(center_locnfeat_alllayers[-1], label, bn_decay=bn_decay, weights=weights)
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
    for i in xrange(3):
        for j in xrange(3):
            for k in xrange(3):
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
    print mod.get_outputs()[0].asnumpy()  # [[[4/3]], [[12/17]]]

