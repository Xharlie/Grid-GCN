#!/usr/bin/env python
# -*- coding:UTF-8 -*-

import sys
import os
sys.path.append("../../data/")
sys.path.append("../../configs/")
sys.path.append("../../models/")
sys.path.append("./segmentation/")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if (sys.version_info < (3, 0)):
    import cPickle as pickle
else:
    import pickle
# sys.path.append("../../library/")
import numpy as np
from joblib import Parallel, delayed
import random
import utils.utils as utils
from configs.configs import configs

def gridify_pointcloud(points, voxel_size=[0.2, 0.2, 0.4], grid_size=[200, 300, 130],
            lidar_coord=[20, 0, 17], max_p_grid=100, max_o_grid=780000, allow_sub=True):
    # Input:
    #   (N, 4)
    # Output:
    #    gridify_dict = {'pindex2ogrid': pindex2ogrid,  [(gx1,gy1,gz1),(),...]
    #                     'ogrid2pindex': ogrid2pindex} {(gx1,gy1,gz1):[12,10,45]}
    #    points: [[px1,py1,pz1,f11,f12..], ....]

    voxel_size = np.array(voxel_size, dtype=np.float32)
    grid_size = np.array(grid_size, dtype=np.int32)
    lidar_coord = np.array(lidar_coord, dtype=np.float32)

    #
    shifted_coord = points[:, :3] + lidar_coord
    # reverse the point cloud coordinate (X, Y, Z) -> (Z, Y, X)
    pindex2ogrid = np.floor(
        shifted_coord[:, :] / voxel_size).astype(np.int32)

    # [K, 3] coordinate buffer as described in the paper, e.g. coordinate_buffer[18] = (175, 293, 7)
    coordinate_buffer = np.unique(pindex2ogrid, axis=0)
    #
    K = len(coordinate_buffer)
    if allow_sub:
        if K > max_o_grid:
            choice = random.sample(range(K), max_o_grid)
            coordinate_buffer = [coordinate_buffer[i] for i in choice]
            K = max_o_grid
    else:
        assert K <= max_o_grid , " K > max_o_grid: %d > %d" % (K, max_o_grid)

    number_dict = {}
    # build a reverse index for coordinate buffer e.g. index_buffer[(175, 293, 7)] = 18
    ogrid2pindex = {}
    for i in range(K):
        ogrid2pindex[tuple(coordinate_buffer[i])] = []
        number_dict[tuple(coordinate_buffer[i])] = 0
    for index in range(pindex2ogrid.shape[0]):
        grid_index = pindex2ogrid[index,...]
        grid_index = tuple(grid_index)
        if grid_index in number_dict:
            number = number_dict[grid_index]
            if number < max_p_grid:
                number_dict[grid_index] += 1
                ogrid2pindex[grid_index].append(index)
            # no limit to number of point for gridify
            # elif not firstlayer:
            #     raise Exception('lose point exception: max_p_grid:{}'.format(max_p_grid))
    # relative coordinates
    gridify_dict = {'pindex2ogrid': pindex2ogrid,
                    'ogrid2pindex': ogrid2pindex}
    return gridify_dict

def queryNeighborCalCenters(gridify_dict, points, max_p_grid, max_o_grid, kernel_size=3, stride=2, \
        single_padding=None, reverse_index=False, allow_sub=True, loc_weight=False, loc_within=False):
    '''
    :param gridify_dict:
    :param points:
    :param kernel_size:
    :param stride:
    :param single_padding:
    :param reverse_index:
    :return: neighbors_hash: {(gx1,gy1,gz1):[ind1, ind150, ind1008]}
                centers_hash: {(gx1,gy1,gz1):np[cx1, cy1, cz1]}
    '''
    assert kernel_size%2==1, " kernel_size: %d is not odd" % kernel_size
    pindex2ogrid = gridify_dict["pindex2ogrid"]
    ogrid2pindex = gridify_dict["ogrid2pindex"]
    assert points.shape[0] == pindex2ogrid.shape[0]
    # if single_padding == None:
    #     single_padding = (kernel_size-1) // 2
    kernel_stretch = (kernel_size-1) // 2
    axis_index_shift = np.tile(np.arange(-kernel_stretch, kernel_stretch+1), (kernel_size, kernel_size, 1))
    kernel_index_shift = np.zeros((kernel_size, kernel_size, kernel_size, 3), dtype=np.int32)
    kernel_index_shift[:, :, :, 2] = axis_index_shift
    kernel_index_shift[:, :, :, 1] = np.swapaxes(axis_index_shift, 2, 1)
    kernel_index_shift[:, :, :, 0] = np.swapaxes(axis_index_shift, 0, 2)
    # print("axis_index_shift0", axis_index_shift)
    # print("axis_index_shifty", np.swapaxes(axis_index_shift, 2, 1))
    # print("axis_index_shiftx", np.swapaxes(axis_index_shift, 0, 2))
    # print(kernel_index_shift)
    neighbors_hash = {}
    centers_hash = {}
    centers = []
    neighbors_arr = np.zeros((max_o_grid, max_p_grid), dtype=np.int32)
    neighbors_mask_arr = np.zeros((max_o_grid, max_p_grid), dtype=np.float32)
    centers_arr = np.zeros((max_o_grid, 4), dtype=np.float32)
    centers_mask_arr = np.zeros((max_o_grid), dtype=np.float32)
    grid_lst = ogrid2pindex.keys()
    for ind, grid in enumerate(grid_lst):
        if sum(dim_idx % stride for dim_idx in grid) == 0:
            kernel_index = np.tile(np.asarray(grid), (kernel_size, kernel_size,1)) + kernel_index_shift
            # print(kernel_index)
            kernel_index = np.reshape(kernel_index, (-1, 3))
            ptidx_in_kernel = []
            center_grid_idx = []
            for i in range(kernel_index.shape[0]):
                grid_key = tuple(kernel_index[i,...])
                if grid_key in ogrid2pindex:
                    ptidx_in_kernel = ptidx_in_kernel + ogrid2pindex[grid_key]
                    if loc_within and i == kernel_index.shape[0] // 2:
                        center_grid_idx = ogrid2pindex[grid_key]
                    if reverse_index:
                        # TODO build projection from point index to grid
                        pass
            if not allow_sub:
                assert ptidx_in_kernel <= max_p_grid, \
                    "num of points in one grid %d is more than allowed %d" % (len(ptidx_in_kernel), max_p_grid)
            elif len(ptidx_in_kernel) > max_p_grid:
                # print("subsampled points in one grid %d is more than allowed %d" % (len(ptidx_in_kernel), max_p_grid))
                choice = random.sample(range(len(ptidx_in_kernel)), max_p_grid)
                ptidx_in_kernel = [ptidx_in_kernel[i] for i in choice]
            if not loc_within:
                center_grid_idx = ptidx_in_kernel
            neighbors_hash[grid] = ptidx_in_kernel
            points_inconv = np.take(points[:, :4], center_grid_idx, axis=0)
            # print(points_inconv)
            if loc_weight:
                centers_hash[grid] = np.zeros(4)
                centers_hash[grid][:3] = np.average(points_inconv[:,:3], axis=0, weights = points_inconv[:,3])
                centers_hash[grid][3] = np.sum(points_inconv[:,3])
            else:
                centers_hash[grid] = np.mean(points_inconv, axis=0)
            centers.append(centers_hash[grid])
            neighbors_arr[ind, :len(ptidx_in_kernel)] = np.asarray(neighbors_hash[grid], dtype = np.int32)
            # pad first
            if len(ptidx_in_kernel) < max_p_grid:
                neighbors_arr[ind, len(ptidx_in_kernel):] = np.full(max_p_grid - len(ptidx_in_kernel), neighbors_arr[ind,0], dtype=np.int32)
            neighbors_mask_arr[ind, :len(ptidx_in_kernel)].fill(1)
            centers_arr[ind] = centers_hash[grid]
            centers_mask_arr[ind] = 1
    return neighbors_hash, centers_hash, neighbors_arr, neighbors_mask_arr,\
           centers_arr, centers_mask_arr, np.asarray(centers, dtype=np.float32)

def queryNeighborWCenters(gridify_dict_down, gridify_dict_up, points_down, points_up,
        max_p_grid, max_o_grid, kernel_size=3, stride=2, single_padding=None, allow_sub=True):

    assert kernel_size % 2 == 1, " kernel_size: %d is not odd" % kernel_size
    pindex2ogrid_down = gridify_dict_down["pindex2ogrid"]
    ogrid2pindex_down = gridify_dict_down["ogrid2pindex"]
    pindex2ogrid_up = gridify_dict_up["pindex2ogrid"]
    ogrid2pindex_up = gridify_dict_up["ogrid2pindex"]

    assert points_down.shape[0] == pindex2ogrid_down.shape[0]
    assert points_up.shape[0] == pindex2ogrid_up.shape[0]
    # if single_padding == None:
    #     single_padding = (kernel_size-1) // 2
    kernel_stretch = (kernel_size - 1) // 2
    axis_index_shift = np.tile(np.arange(-kernel_stretch, kernel_stretch + 1), (kernel_size, kernel_size, 1))
    kernel_index_shift = np.zeros((kernel_size, kernel_size, kernel_size, 3), dtype=np.int32)
    kernel_index_shift[:, :, :, 2] = axis_index_shift
    kernel_index_shift[:, :, :, 1] = np.swapaxes(axis_index_shift, 2, 1)
    kernel_index_shift[:, :, :, 0] = np.swapaxes(axis_index_shift, 0, 2)
    # print("axis_index_shift0", axis_index_shift)
    # print("axis_index_shifty", np.swapaxes(axis_index_shift, 2, 1))
    # print("axis_index_shiftx", np.swapaxes(axis_index_shift, 0, 2))
    # print(kernel_index_shift)
    neighbors_hash = {}
    neighbors_arr = np.zeros((max_o_grid, max_p_grid), dtype=np.int32)
    neighbors_mask_arr = np.zeros((max_o_grid, max_p_grid), dtype=np.float32)
    for ind in range(pindex2ogrid_up.shape[0]):
        grid = tuple(pindex2ogrid_up[ind])
        if sum(dim_idx % stride for dim_idx in grid) == 0:
            kernel_index = np.tile(np.asarray(grid), (kernel_size, kernel_size, 1)) + kernel_index_shift
            # print(kernel_index)
            kernel_index = np.reshape(kernel_index, (-1, 3))
            ptidx_in_kernel = []
            for i in range(kernel_index.shape[0]):
                grid_key = tuple(kernel_index[i, ...])
                if grid_key in ogrid2pindex_down:
                    ptidx_in_kernel = ptidx_in_kernel + ogrid2pindex_down[grid_key]
            if not allow_sub:
                assert ptidx_in_kernel <= max_p_grid, \
                    "num of points in one grid %d is more than allowed %d" % (len(ptidx_in_kernel), max_p_grid)
            elif len(ptidx_in_kernel) > max_p_grid:
                # print("subsampled points in one grid %d is more than allowed %d" % (len(ptidx_in_kernel), max_p_grid))
                choice = random.sample(range(len(ptidx_in_kernel)), max_p_grid)
                ptidx_in_kernel = [ptidx_in_kernel[i] for i in choice]
            neighbors_hash[grid] = ptidx_in_kernel
            neighbors_arr[ind, :len(ptidx_in_kernel)] = np.asarray(neighbors_hash[grid], dtype=np.int32)
            # pad first
            if len(ptidx_in_kernel) < max_p_grid:
                neighbors_arr[ind, len(ptidx_in_kernel):] = np.full(max_p_grid - len(ptidx_in_kernel),
                                                                    neighbors_arr[ind, 0], dtype=np.int32)
            neighbors_mask_arr[ind, :len(ptidx_in_kernel)].fill(1)
    return neighbors_hash, neighbors_arr, neighbors_mask_arr


def assemble_up_layer_tensor(centers, points_batch, voxel_size = [0.2, 0.2, 0.4], grid_size = [200, 300, 130],
        lidar_coord = [20, 0, 17], max_p_grid = 100, max_o_grid = 780000, kernel_size=3, stride=2,
        single_padding=None, para=True, allow_sub=True):

    batch_size = len(centers)
    batch_neighbors_arr = np.zeros((batch_size, max_o_grid, max_p_grid), dtype=np.int32)
    batch_neighbors_mask_arr = np.zeros((batch_size, max_o_grid, max_p_grid), dtype=np.float32)

    if para:
        repeat = batch_size
        indices = list(range(repeat))
        voxel_size_lst = [voxel_size for i in range(repeat)]
        grid_size_lst = [grid_size for i in range(repeat)]
        lidar_coord_lst = [lidar_coord for i in range(repeat)]
        max_p_grid_lst = [max_p_grid for i in range(repeat)]
        max_o_grid_lst = [max_o_grid for i in range(repeat)]
        kernel_size_lst = [kernel_size for i in range(repeat)]
        stride_lst = [stride for i in range(repeat)]
        single_padding_lst = [single_padding for i in range(repeat)]
        allow_sub_lst = [allow_sub for i in range(repeat)]
        with Parallel(n_jobs=min(batch_size, 32)) as parallel:
            result_lst \
                = parallel(delayed(get_up_hash_singlePC)
                    (ind, points_down, points_up, voxel_size, grid_size, lidar_coord, max_p_grid, max_o_grid,
                    kernel_size, stride, single_padding, allow_sub)
                    for ind, points_down, points_up, voxel_size, grid_size, lidar_coord, max_p_grid, max_o_grid,
                        kernel_size, stride, single_padding, allow_sub in
                    zip(indices, centers, points_batch, voxel_size_lst, grid_size_lst, lidar_coord_lst,
                        max_p_grid_lst, max_o_grid_lst, kernel_size_lst, stride_lst,
                        single_padding_lst, allow_sub_lst))
            for i in range(len(result_lst)):
                index, neighbors_hash, neighbors_hash_arr, neighbors_mask_arr = result_lst[i]
                batch_neighbors_arr[index, ...] = neighbors_hash_arr
                batch_neighbors_mask_arr[index, ...] = neighbors_mask_arr
    else:
        for ind, points_up in enumerate(points_batch):
            points_down = centers[ind]
            index, neighbors_hash, neighbors_hash_arr, neighbors_mask_arr \
                = get_up_hash_singlePC(ind, points_down, points_up, voxel_size, grid_size, lidar_coord, max_p_grid, max_o_grid,
                                    kernel_size, stride, single_padding, allow_sub)

            batch_neighbors_arr[index, ...] = neighbors_hash_arr
            batch_neighbors_mask_arr[index, ...] = neighbors_mask_arr
    return batch_neighbors_arr, batch_neighbors_mask_arr


def assemble_down_layer_tensor(points_batch, voxel_size = [0.2, 0.2, 0.4], grid_size = [200, 300, 130],
        lidar_coord = [20, 0, 17], max_p_grid = 100, max_o_grid = 780000, kernel_size=3, stride=2,
        single_padding=None, reverse_index=False, para=True, allow_sub=True,
        loc_weight=False, loc_within=False):
    '''
    :param points_batch:
    :param voxel_size:
    :param grid_size:
    :param lidar_coord:
    :param max_p_grid:
    :param max_o_grid:
    :param kernel_size:
    :param stride:
    :param single_padding:
    :param reverse_index:
    :return: batch_neighbors_hash_arr : batch_size, max_o_grid, max_p_grid
             batch_neighbors_mask_arr : batch_size, max_o_grid, max_p_grid
             batch_centers_arr:         batch_size, max_o_grid, 3
             batch_centers_mask_arr:    batch_size, max_o_grid
    '''
    batch_size = len(points_batch)
    batch_neighbors_arr = np.zeros((batch_size, max_o_grid, max_p_grid), dtype=np.int32)
    batch_neighbors_mask_arr = np.zeros((batch_size, max_o_grid, max_p_grid), dtype=np.float32)
    batch_centers_arr = np.zeros((batch_size, max_o_grid, 4), dtype=np.float32)
    batch_centers_mask_arr = np.zeros((batch_size, max_o_grid), dtype=np.float32)
    batch_centers = []

    if para:
        repeat = len(points_batch)
        indices = list(range(repeat))
        voxel_size_lst = [voxel_size for i in range(repeat)]
        grid_size_lst = [grid_size for i in range(repeat)]
        lidar_coord_lst = [lidar_coord for i in range(repeat)]
        max_p_grid_lst = [max_p_grid for i in range(repeat)]
        max_o_grid_lst = [max_o_grid for i in range(repeat)]
        kernel_size_lst = [kernel_size for i in range(repeat)]
        stride_lst = [stride for i in range(repeat)]
        single_padding_lst = [single_padding for i in range(repeat)]
        reverse_index_lst = [reverse_index for i in range(repeat)]
        allow_sub_lst = [allow_sub for i in range(repeat)]
        loc_weight_lst = [loc_weight for i in range(repeat)]
        loc_within_lst = [loc_within for i in range(repeat)]
        with Parallel(n_jobs=min(batch_size,32)) as parallel:
            result_lst \
                = parallel(delayed(get_down_hash_singlePC)
                     (ind, points_ori, voxel_size, grid_size, lidar_coord, max_p_grid, max_o_grid,
                      kernel_size, stride, single_padding, reverse_index, allow_sub, loc_weight, loc_within)
                     for ind, points_ori, voxel_size, grid_size, lidar_coord, max_p_grid, max_o_grid,
                         kernel_size, stride, single_padding, reverse_index, allow_sub, loc_weight, loc_within in
                     zip(indices, points_batch, voxel_size_lst, grid_size_lst, lidar_coord_lst,
                         max_p_grid_lst, max_o_grid_lst, kernel_size_lst, stride_lst,
                         single_padding_lst, reverse_index_lst, allow_sub_lst, loc_weight_lst, loc_within_lst))
            for i in range(len(result_lst)):
                index, neighbors_hash, centers_hash, neighbors_hash_arr,\
                neighbors_mask_arr, centers_arr, centers_mask_arr, centers = result_lst[i]
                batch_neighbors_arr[index, ...] = neighbors_hash_arr
                batch_neighbors_mask_arr[index, ...] = neighbors_mask_arr
                batch_centers_arr[index, ...] = centers_arr
                batch_centers_mask_arr[index, ...] = centers_mask_arr
                batch_centers.append(centers)
    else:
        for ind, points_ori in enumerate(points_batch):
            index, neighbors_hash, centers_hash, neighbors_hash_arr, neighbors_mask_arr, centers_arr, centers_mask_arr, centers \
                = get_down_hash_singlePC(ind, points_ori, voxel_size, grid_size, lidar_coord, max_p_grid, max_o_grid,
                    kernel_size, stride, single_padding, reverse_index, allow_sub, loc_weight, loc_within)

            batch_neighbors_arr[index, ...] = neighbors_hash_arr
            batch_neighbors_mask_arr[index, ...] = neighbors_mask_arr
            batch_centers_arr[index, ...] = centers_arr
            batch_centers_mask_arr[index, ...] = centers_mask_arr
            batch_centers.append(centers)
    return batch_neighbors_arr, batch_neighbors_mask_arr, batch_centers_arr, batch_centers_mask_arr, batch_centers

def get_up_hash_singlePC(ind, points_down, points_up, voxel_size, grid_size, lidar_coord, max_p_grid,
        max_o_grid, kernel_size, stride, single_padding, allow_sub):

    gridify_dict_down = gridify_pointcloud(points_down, voxel_size=voxel_size, grid_size=grid_size,
        lidar_coord=lidar_coord, max_p_grid=max_p_grid, max_o_grid=max_o_grid)

    gridify_dict_up = gridify_pointcloud(points_up, voxel_size=voxel_size, grid_size=grid_size,
        lidar_coord=lidar_coord, max_p_grid=max_p_grid, max_o_grid=max_o_grid)

    neighbors_hash, neighbors_hash_arr, neighbors_mask_arr \
        = queryNeighborWCenters(gridify_dict_down, gridify_dict_up, points_down, points_up, max_p_grid, max_o_grid,
            kernel_size=kernel_size, stride=stride, single_padding=single_padding,allow_sub=allow_sub)
    return ind, neighbors_hash, neighbors_hash_arr, neighbors_mask_arr


def get_down_hash_singlePC(ind, points, voxel_size, grid_size, lidar_coord, max_p_grid,
        max_o_grid, kernel_size, stride, single_padding, reverse_index, allow_sub, loc_weight, loc_within):

    gridify_dict = gridify_pointcloud(points, voxel_size=voxel_size, grid_size=grid_size,
        lidar_coord=lidar_coord, max_p_grid=max_p_grid, max_o_grid=max_o_grid, allow_sub=allow_sub)

    neighbors_hash, centers_hash, neighbors_hash_arr, neighbors_mask_arr, centers_arr, centers_mask_arr, centers \
        = queryNeighborCalCenters(gridify_dict, points, max_p_grid, max_o_grid,
            kernel_size=kernel_size, stride=stride, single_padding=single_padding,
            reverse_index=reverse_index, allow_sub=allow_sub, loc_weight=loc_weight, loc_within=loc_within)
    return ind, neighbors_hash, centers_hash, neighbors_hash_arr, neighbors_mask_arr, centers_arr, centers_mask_arr, centers

def assemble_tensor(points_batch, voxel_size_lst = [[0.2, 0.2, 0.4]], grid_size_lst = [[40, 40, 40], [8, 8, 8], [1, 1, 1]],
        lidar_coord = [1.0, 1.0, 1.0], max_p_grid_lst = [64, 64, 128], max_o_grid_lst = [1024, 128, 1], kernel_size_lst=[7, 3, 1],
        stride_lst=[1,1,1], single_padding_lst=[None,None,None], reverse_index=False, para=True, allow_sub=True,
        loc_weight=False, loc_within=False, up_voxel_size_lst = [],  up_max_p_grid_lst = [], up_max_o_grid_lst= [],
        up_grid_size_lst=[], up_kernel_size_lst=[], up_stride_lst=[]):
    num_dn_layers = len(voxel_size_lst)
    num_up_layers = len(up_voxel_size_lst)
    if num_up_layers !=0:
        assert num_up_layers==num_dn_layers, "num_up_layers != num_dn_layers !!!"
    assert num_dn_layers == len(grid_size_lst) == len(max_p_grid_lst) == len(max_o_grid_lst) == len(kernel_size_lst) \
           == len(stride_lst) == len(single_padding_lst) \
           , " input of assemble_tensor should has same layer num %d" % num_dn_layers
    neighbors_arr_lst = []
    neighbors_mask_lst = []
    neighbors_arr_up_lst = []
    neighbors_mask_up_lst = []
    centers_arr_lst = []
    centers_mask_lst = []
    centers_lst = []
    B,N,C = points_batch.shape
    points_weight = np.ones((B, N, 1), dtype="float32")
    if C == 6:
        points_batch = points_batch[:, :, :3]
    elif C != 3:
        raise NotImplementedError
    points_batch_w = np.concatenate((points_batch, points_weight), axis=2)

    for i in range(num_dn_layers):
        batch_neighbors_arr, batch_neighbors_mask_arr, batch_centers_arr, batch_centers_mask_arr, points_batch_w = \
            assemble_down_layer_tensor(points_batch_w, voxel_size=voxel_size_lst[i], grid_size=grid_size_lst[i],
            lidar_coord=lidar_coord, max_p_grid=max_p_grid_lst[i], max_o_grid=max_o_grid_lst[i],
            kernel_size=kernel_size_lst[i], stride=stride_lst[i], single_padding=single_padding_lst[i],
            reverse_index=reverse_index, para=para, allow_sub=allow_sub,
            loc_weight=loc_weight, loc_within=loc_within)
        neighbors_arr_lst.append(batch_neighbors_arr)
        neighbors_mask_lst.append(batch_neighbors_mask_arr)
        centers_arr_lst.append(batch_centers_arr)
        centers_mask_lst.append(batch_centers_mask_arr)
        centers_lst.append(points_batch_w)
    if "up_type" in configs.keys():
        if not configs["up_neigh_fetch"]:
            if configs["up_type"] == "grid":
                for i in range(num_up_layers):
                    points_down = centers_lst[num_up_layers-i-1]
                    if i == num_up_layers-1:
                        points_up = points_batch
                    else:
                        points_up = centers_lst[num_up_layers-i-2]

                    batch_neighbors_arr, batch_neighbors_mask_arr = \
                        assemble_up_layer_tensor(points_down, points_up, voxel_size=up_voxel_size_lst[i],
                            grid_size=up_grid_size_lst[i], lidar_coord=lidar_coord, max_p_grid=up_max_p_grid_lst[i],
                            max_o_grid=up_max_o_grid_lst[i], kernel_size=up_kernel_size_lst[i], stride=up_stride_lst[i],
                            single_padding=single_padding_lst[i], para=para, allow_sub=allow_sub)
                    neighbors_arr_up_lst.append(batch_neighbors_arr)
                    neighbors_mask_up_lst.append(batch_neighbors_mask_arr)
            elif configs["up_type"] == "grid_full":
                for i in range(num_up_layers):
                    points_down = centers_lst[num_up_layers - i - 1]
                    points_up = points_batch
                    batch_neighbors_arr, batch_neighbors_mask_arr = \
                        assemble_up_layer_tensor(points_down, points_up, voxel_size=up_voxel_size_lst[i],
                             grid_size=up_grid_size_lst[i], lidar_coord=lidar_coord,
                             max_p_grid=up_max_p_grid_lst[i], max_o_grid=up_max_o_grid_lst[i], kernel_size=up_kernel_size_lst[i],
                             stride=up_stride_lst[i], single_padding=single_padding_lst[i], para=para, allow_sub=allow_sub)
                    neighbors_arr_up_lst.append(batch_neighbors_arr)
                    neighbors_mask_up_lst.append(batch_neighbors_mask_arr)
            elif configs["up_type"] != "inter":
                raise NotImplementedError
    return neighbors_arr_lst, neighbors_mask_lst, neighbors_arr_up_lst, neighbors_mask_up_lst,\
           centers_arr_lst, centers_mask_lst, centers_lst

def read_dataset():
    dataset = []
    with open(os.path.join("../../data/ApolloScape", '{}.lst'.format("train"))) as f:
        for line in f:
            dataset.append(os.path.join("../../data/ApolloScape", line.strip()))
    return dataset

def cal_stats():
    # 'max_num_point, min_num_point, mean_num_point,    max_x, min_x, mean_x,    max_y, min_y, mean_y,    max_z, min_z, mean_z',
    # 0, 0, 181045, 19.99999957950906, -19.999999214392805, 0.5265633702711263, 59.985, 0, 14.233127554956258,
    # 34.53871342733551, -16.82669667553399, -1.6792175217866303
    dataset = read_dataset()
    max_num_point, min_num_point, sum_num_point = 0, 1000000, 0
    max_x, min_x, sum_x = 0, 0, 0
    max_y, min_y, sum_y = 0, 0, 0
    max_z, min_z, sum_z = 0, 0, 0
    count = 0
    for index in range(len(dataset)):
        npz = np.load(dataset[index])
        data, label = npz['pc'], npz['label']
        # keep only points within abs(x) < 20, abs(y) < 60
        distance_mask = (np.abs(data[:, 0]) < 20) & (np.abs(data[:, 1]) < 60)
        data = data[distance_mask]
        label = label[distance_mask]
        # in case all points are too far away, randomly choose another sample to replace it
        if data.shape[0] > 0:
            max_num_point = max(max_num_point, data.shape[0])
            min_num_point = min(min_num_point, data.shape[0])
            sum_num_point += data.shape[0]
            sum_x += np.sum(data[:, 0])
            sum_y += np.sum(data[:, 1])
            sum_z += np.sum(data[:, 2])
            max_x = max(max_x, np.max(data[:, 0]))
            max_y = max(max_y, np.max(data[:, 1]))
            max_z = max(max_z, np.max(data[:, 2]))
            min_x = min(min_x, np.min(data[:, 0]))
            min_y = min(min_y, np.min(data[:, 1]))
            min_z = min(min_z, np.min(data[:, 2]))
            count+=1
            print(index, max_x,max_y,max_z,min_x,min_y,min_z)
        else:
            print("size is zero:", dataset[index])
    return count, max_num_point, min_num_point, sum_num_point/count,\
    max_x, min_x, sum_x/sum_num_point,\
    max_y, min_y, sum_y/sum_num_point,\
    max_z, min_z, sum_z/sum_num_point

def get_one_pointcloud(file):
    npz = np.load(file)
    data, label = npz['pc'], npz['label']
    return data, label


if __name__ == "__main__":

# get stats of entire dataset
    # number_pt, \
    # max_num_point, min_num_point, mean_num_point,\
    # max_x, min_x, mean_x,\
    # max_y, min_y, mean_y,\
    # max_z, min_z, mean_z = cal_stats()
    # print("max_num_point, min_num_point, mean_num_point,\
    # max_x, min_x, mean_x,\
    # max_y, min_y, mean_y,\
    # max_z, min_z, mean_z",max_num_point, min_num_point, mean_num_point,\
    # max_x, min_x, mean_x,\
    # max_y, min_y, mean_y,\
    # max_z, min_z, mean_z)

# # check gridify_pointcloud
# #     point_cloud = get_one_pointcloud("../../data/ApolloScape/road01_ins/Record016/170908_062206637_Camera_6.npz")[0]
#     points = [[0.5, 0.5, 0.5], [0.1, 1.8, 0.5], [0.3, 2.9, 0.5], [0.15, 3.8, 0.5],
#               [1.5, 0.5, 0.5], [1.4, 1.3, 0.5],
#               [2.3, 0.7, 0.5], [2.4, 1.55, 0.5],
#               [3.5, 0.3, 0.5], [3.4, 1.15, 0.5], [3.4, 2.3, 0.5]]
#     points=np.asarray(points)
#     max_p_grid = 100
#     max_o_grid = 780000
#     gridify_dict, points_index, points = gridify_pointcloud(points, voxel_size=[1.0, 1.0, 1.0],
#         grid_size=[4, 4, 4], lidar_coord=[0, 0, 0], max_p_grid=max_p_grid, max_o_grid=max_o_grid, firstlayer=False)
#     # print(gridify_dict)
#
# # check queryNeighborCalCenters
#     neighbors_hash, centers_hash, neighbors_arr, neighbors_mask_arr, centers_arr, centers_mask_arr, centers \
#         = queryNeighborCalCenters(gridify_dict, points, points_index, max_p_grid, max_o_grid,
#             kernel_size=3, stride=1, single_padding=None, reverse_index=False)
#     # print(neighbors_hash)
#     # print(centers_hash)
#     print(neighbors_arr[:16,:9,...])
#     print(neighbors_mask_arr[:16,:9,...])
#     print(centers_arr[:16,...])
#     print(centers_mask_arr[:16,...])

# check all pipeline:
#     points1 = [[0.5, 0.5, 0.5], [0.1, 1.8, 0.5], [0.3, 2.9, 0.5], [0.15, 3.8, 0.5],
#               [1.5, 0.5, 0.5], [1.4, 1.3, 0.5],
#               [2.3, 0.7, 0.5], [2.4, 1.55, 0.5],
#               [3.5, 0.3, 0.5], [3.4, 1.15, 0.5], [3.4, 2.3, 0.5]]
#     points1 = np.asarray(points1)
#     points2 = np.concatenate([points1[:,:2], points1[:,[2]] + 2], axis = 1)
#     points_batch = [points1, points2]
#     neighbors_arr_lst, neighbors_mask_lst, centers_arr_lst, centers_mask_lst, centers_lst = \
#         assemble_tensor(points_batch, voxel_size_lst = [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], grid_size_lst = [[4,4,4], [4,4,4]],
#         lidar_coord = [0, 0, 0], max_p_grid_lst = [16, 16], max_o_grid_lst = [16, 16], kernel_size_lst=[3, 3], stride_lst=[1, 2],
#         single_padding_lst=[None, None], reverse_index=False)
#
#     print(neighbors_arr_lst)
#     print(neighbors_mask_lst)
#     print(centers_arr_lst)
#     print(centers_mask_lst)
#     print(centers_lst[0])





# To vis
# print all layers
#     root = "/mnt/truenas/scratch/qiangeng/dev/GGCN/classification/data/modelnet40_normal_resampled"
#     shape_ids = [line.rstrip() for line in open(os.path.join(root, 'modelnet40_{}.txt'.format("train")))]
#     shape_names = ['_'.join(idx.split('_')[0:-1]) for idx in shape_ids]
#     # list of (shape_name, shape_txt_file_path) tuple
#     datapath = [(shape_names[i], os.path.join(root, shape_names[i], idx+'.txt')) for i, idx in enumerate(shape_ids)]
#     vis_root = "/mnt/truenas/scratch/qiangeng/dev/GGCN/classification/vis/"
#     import mxnet as mx
#     from classification.models import sampling
#
#     data_batch = mx.symbol.Variable('data_batch')
#     layer0 = sampling(data_batch, (1, 1024, 0), method='fps', fps_npoints=512, scope="l0")
#     layer1 = sampling(layer0, (1, 512, 0), method='fps', fps_npoints=128, scope="l1")
#     layer2 = sampling(layer1, (1, 128, 0), method='fps', fps_npoints=8, scope="l2")
#     layer2 = layer2.get_internals()
#     context = mx.cpu()
#     mod = mx.mod.Module(layer2, data_names=['data_batch'], label_names=[], context=context)
#     mod.bind(data_shapes=[('data_batch', (1, 1024, 3))])
#     mod.init_params()
#     print layer2.list_outputs()
#
#     for index in range(len(datapath)):
#         cls_name, shape_txt_filename = datapath[index]
#         point_set = np.loadtxt(shape_txt_filename, delimiter=',')[:1024, ...]
#         point_set[:, 0:3] = utils.normalize_point_cloud(point_set[:, 0:3])
#         points_batch = np.asarray([point_set]).astype(np.float32)
#         shape_dir = shape_txt_filename.split("/")[-1].split("_")[1].split(".")[0]
#         fdir = os.path.join(vis_root, cls_name, shape_dir)
#         if not os.path.exists(fdir):
#             os.makedirs(fdir)
#         print("write to fdir: ",fdir)

        #
        # neighbors_arr_lst, neighbors_mask_lst, centers_arr_lst, centers_mask_lst, centers_lst = \
        #     assemble_tensor(points_batch,
        #                     voxel_size_lst=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [2.0, 2.0, 2.0]],
        #                     grid_size_lst=[[20, 20, 20], [10, 10, 10], [5, 5, 5], [1, 1, 1]],
        #                     lidar_coord=[1.0, 1.0, 1.0],
        #                     max_p_grid_lst=[513, 514, 515, 125],
        #                     max_o_grid_lst=[1024, 1000, 125, 1],
        #                     kernel_size_lst=[3, 3, 3, 1],
        #                     stride_lst=[1, 1, 1, 1],
        #                     single_padding_lst=[None, None, None, None], reverse_index=False, para=False,
        #                     allow_sub=True)
        # fname = "raw" + ".txt"
        # fname = os.path.join(fdir, fname)
        # np.savetxt(fname, points_batch[0], delimiter=";")
        # print("saved:", fname)
        # for layer in range(len(centers_lst)):
        #     centers = centers_lst[layer][0]
        #     print(centers.shape)
        #     fname = "ly_" + str(layer) + ".txt"
        #     fname = os.path.join(fdir, fname)
        #     np.savetxt(fname, centers, delimiter=";")
        #     print("saved:", fname)
        #
        # neighbors_arr_lst, neighbors_mask_lst, centers_arr_lst, centers_mask_lst, centers_lst = \
        #     assemble_tensor(points_batch,
        #                     voxel_size_lst=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [2.0, 2.0, 2.0]],
        #                     grid_size_lst=[[20, 20, 20], [10, 10, 10], [5, 5, 5], [1, 1, 1]],
        #                     lidar_coord=[1.0, 1.0, 1.0],
        #                     max_p_grid_lst=[513, 514, 515, 125],
        #                     max_o_grid_lst=[1024, 1000, 125, 1],
        #                     kernel_size_lst=[3, 3, 3, 1],
        #                     stride_lst=[1, 1, 1, 1],
        #                     single_padding_lst=[None, None, None, None], reverse_index=False, para=False,
        #                     allow_sub=True, loc_weight=True, loc_within=False)
        # # fname = "raw" + ".txt"
        # # fname = os.path.join(fdir, fname)
        # # np.savetxt(fname, points_batch[0], delimiter=";")
        # # print("saved:", fname)
        # for layer in range(len(centers_lst)):
        #     centers = centers_lst[layer][0]
        #     print(centers.shape)
        #     fname = "ly_locweight" + str(layer) + ".txt"
        #     fname = os.path.join(fdir, fname)
        #     np.savetxt(fname, centers, delimiter=";")
        #     print("saved:", fname)
        #
        # neighbors_arr_lst, neighbors_mask_lst, centers_arr_lst, centers_mask_lst, centers_lst = \
        #     assemble_tensor(points_batch,
        #                     voxel_size_lst=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [2.0, 2.0, 2.0]],
        #                     grid_size_lst=[[20, 20, 20], [10, 10, 10], [5, 5, 5], [1, 1, 1]],
        #                     lidar_coord=[1.0, 1.0, 1.0],
        #                     max_p_grid_lst=[513, 514, 515, 125],
        #                     max_o_grid_lst=[1024, 1000, 125, 1],
        #                     kernel_size_lst=[3, 3, 3, 1],
        #                     stride_lst=[1, 1, 1, 1],
        #                     single_padding_lst=[None, None, None, None], reverse_index=False, para=False,
        #                     allow_sub=True, loc_weight=False, loc_within=True)
        # # fname = "raw" + ".txt"
        # # fname = os.path.join(fdir, fname)
        # # np.savetxt(fname, points_batch[0], delimiter=";")
        # # print("saved:", fname)
        # for layer in range(len(centers_lst)):
        #     centers = centers_lst[layer][0]
        #     print(centers.shape)
        #     fname = "ly_locin" + str(layer) + ".txt"
        #     fname = os.path.join(fdir, fname)
        #     np.savetxt(fname, centers, delimiter=";")
        #     print("saved:", fname)
        #
        # neighbors_arr_lst, neighbors_mask_lst, centers_arr_lst, centers_mask_lst, centers_lst = \
        #     assemble_tensor(points_batch,
        #                     voxel_size_lst=[[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [2.0, 2.0, 2.0]],
        #                     grid_size_lst=[[20, 20, 20], [10, 10, 10], [5, 5, 5], [1, 1, 1]],
        #                     lidar_coord=[1.0, 1.0, 1.0],
        #                     max_p_grid_lst=[513, 514, 515, 125],
        #                     max_o_grid_lst=[1024, 1000, 125, 1],
        #                     kernel_size_lst=[3, 3, 3, 1],
        #                     stride_lst=[1, 1, 1, 1],
        #                     single_padding_lst=[None, None, None, None], reverse_index=False, para=False,
        #                     allow_sub=True, loc_weight=True, loc_within=True)
        # # fname = "raw" + ".txt"
        # # fname = os.path.join(fdir, fname)
        # # np.savetxt(fname, points_batch[0], delimiter=";")
        # # print("saved:", fname)
        # for layer in range(len(centers_lst)):
        #     centers = centers_lst[layer][0]
        #     print(centers.shape)
        #     fname = "ly_weight_locin" + str(layer) + ".txt"
        #     fname = os.path.join(fdir, fname)
        #     np.savetxt(fname, centers, delimiter=";")
        #     print("saved:", fname)

        # inputs_points_batch = mx.ndarray.array(points_batch[:,:,:3])
        # mod.forward(mx.io.DataBatch([inputs_points_batch]))
        # layer0_result = mod.get_outputs()[7].asnumpy()
        # layer1_result = mod.get_outputs()[14].asnumpy()
        # layer2_result = mod.get_outputs()[21].asnumpy()
        # centers_lst = []
        # centers_lst.append(layer0_result)
        # centers_lst.append(layer1_result)
        # centers_lst.append(layer2_result)
        # for layer in range(len(centers_lst)):
        #     centers = centers_lst[layer][0]
        #     print(centers.shape)
        #     fname = "ly_fps_" + str(layer) + ".txt"
        #     fname = os.path.join(fdir, fname)
        #     np.savetxt(fname, centers, delimiter=";")
        #     print("saved:", fname)

# To single vis
# single file
#     vis_root = "/mnt/truenas/scratch/qiangeng/dev/GGCN/single_vis/"
#
#     single_path = '/mnt/truenas/scratch/qiangeng/dev/GGCN/data/modelnet40_normal_resampled/radio/radio_0013.txt'
#     db_name, cls_name, shape_txt_filename = "modelnet40", "radio", single_path
#
#     point_set = np.loadtxt(shape_txt_filename, delimiter=',')[:1024, ...]
#     point_set = utils.normalize_point_cloud(point_set[:, 0:3])
#     points_batch = np.asarray([point_set]).astype(np.float32)
#     shape_dir = shape_txt_filename.split("/")[-1].split("_")[1].split(".")[0]
#     fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
#     if not os.path.exists(fdir):
#         os.makedirs(fdir)
#     print("write to fdir: ",fdir)
#
#     neighbors_arr_lst, neighbors_mask_lst, neighbors_arr_up_lst, neighbors_mask_up_lst, \
#     centers_arr_lst, centers_mask_lst, centers_lst = \
#         assemble_tensor(points_batch,
#             voxel_size_lst=[[0.05, 0.05, 0.05], [0.25, 0.25, 0.25], [2.0, 2.0, 2.0]],
#             grid_size_lst=[[40, 40, 40], [8, 8, 8], [1, 1, 1]],
#             lidar_coord=[1.0, 1.0, 1.0],
#             max_p_grid_lst=[64, 64, 32],
#             max_o_grid_lst=[256, 32, 1],
#             kernel_size_lst=[7,3,1],
#             stride_lst=[1, 1, 1],
#             single_padding_lst=[None, None, None], reverse_index=False, para=False,
#             allow_sub=True,
#             loc_weight=True,
#             loc_within=True,
#             up_voxel_size_lst = [[2.0, 2.0, 2.0], [0.25, 0.25, 0.25], [0.05, 0.05, 0.05]],
#             up_max_p_grid_lst = [1, 64, 64],
#             up_max_o_grid_lst= [32, 256, 1024],  up_grid_size_lst=[[1, 1, 1], [8, 8, 8], [40, 40, 40]],
#             up_kernel_size_lst=[1,3,7], up_stride_lst=[1,1,1])
#     fname = "raw" + ".txt"
#     fname = os.path.join(fdir, fname)
#     np.savetxt(fname, points_batch[0], delimiter=";")
#     print("saved:", fname)
#     print("neighbors_arr_up.shape", [neighbors_arr_up[0].shape for neighbors_arr_up in neighbors_arr_up_lst])
#
#     for layer in range(len(centers_lst)):
#         centers = centers_lst[layer][0]
#         last_centers = centers_lst[layer-1][0] if layer!=0 else points_batch[0]
#         neighbors_arr_up = neighbors_arr_up_lst[len(neighbors_arr_up_lst) - layer - 1][0]
#         points_down = centers
#         points_up = last_centers
#         print("points_down.shape", points_down.shape)
#         print("points_up.shape", points_up.shape)
#         print(centers.shape)
#         fname = "ly_" + str(layer) + ".txt"
#         fname = os.path.join(fdir, fname)
#         np.savetxt(fname, centers, delimiter=";")
#         print("saved:", fname)
#
#         neighbors_arr = neighbors_arr_lst[layer][0]
#         centers_mask = centers_mask_lst[layer][0]
#
#         if layer != 0:
#             points_up_mask = centers_mask_lst[layer - 1][0]
#
#         neighbors_mask = neighbors_mask_lst[layer][0]
#         neighbors_mask_up = neighbors_mask_up_lst[len(neighbors_mask_up_lst) - layer - 1][0]
#         for cl_ind in range(neighbors_arr.shape[0]):
#             associated_points = []
#             if centers_mask[cl_ind] == 0:
#                 break
#             else:
#                 for pt_ind in range(neighbors_arr.shape[1]):
#                     if neighbors_mask[cl_ind, pt_ind] == 0:
#                         break
#                     else:
#                         associated_points.append(last_centers[neighbors_arr[cl_ind, pt_ind, ...]])
#             associated_points = np.asarray(associated_points, dtype="float32")
#             print("finish asso pts for {}, shape: {}".format(cl_ind, associated_points.shape))
#             fname = "ass_lyr" + str(layer) + "_pt" + str(cl_ind) + ".txt"
#             fname = os.path.join(fdir, fname)
#             np.savetxt(fname, associated_points, delimiter=";")
#             target_point = centers[cl_ind]
#             fpname = "ass_lyr" + str(layer) + "_pts" + str(cl_ind) + "_target.txt"
#             fpname = os.path.join(fdir, fpname)
#             np.savetxt(fpname, target_point, delimiter=";")
#
#         print('neighbors_mask_up.shape', neighbors_mask_up.shape)
#         for cl_ind in range(neighbors_arr_up.shape[0]):
#             associated_points = []
#             if layer != 0 and points_up_mask[cl_ind] == 0:
#                 break
#             else:
#                 for pt_ind in range(neighbors_arr_up.shape[1]):
#                     if neighbors_mask_up[cl_ind, pt_ind] == 0:
#                         break
#                     else:
#                         print("len(points_down), cl_ind, pt_ind", len(points_down), cl_ind, pt_ind)
#                         associated_points.append(points_down[neighbors_arr_up[cl_ind, pt_ind, ...]])
#             associated_points = np.asarray(associated_points, dtype="float32")
#             target_point = np.asarray([points_up[cl_ind,:3]], dtype="float32")
#             print("finish up asso  for layer {} 's pts {}, shape: {}".format(layer, cl_ind, associated_points.shape))
#             fname = "up_ass_lyr" + str(layer) + "_pts" + str(cl_ind) + ".txt"
#             fpname = "up_ass_lyr" + str(layer) + "_pts" + str(cl_ind) + "_target.txt"
#             fname = os.path.join(fdir, fname)
#             fpname = os.path.join(fdir, fpname)
#             np.savetxt(fname, associated_points, delimiter=";")
#             np.savetxt(fpname, target_point, delimiter=";")


# To single vis
# single file of scannet
    vis_root = "/home/xharlie/dev/GGCN-corp/single_vis/"
    CLASS_NAMES = {0:'unknown', 1: 'wall', 2: 'floor', 3: 'chair', 4: 'table', 5: 'desk', 6: 'bed', 7: 'bookshelf', 8: 'sofa',
                   9: 'sink', 10: 'bathtub', 11: 'toilet', 12: 'curtain', 13: 'counter', 14: 'door', 15: 'window',
                   16: 'shower_curtain', 17: 'refrigerator', 18: 'picture', 19: 'cabinet', 20: 'other'}
    CLASS_COLORS = np.asarray([(0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
                               (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
                               (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
                               (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]).astype("float32")

    single_path = '/home/xharlie/dev/GGCN-corp/data/scannet/scannet_test.pickle'
    db_name, cls_name, shape_txt_filename = "scannet", "10", single_path
    index = int(cls_name)
    with open(single_path, "rb") as f:
        data = pickle.load(f, encoding="bytes")[index]
        label = pickle.load(f, encoding="bytes")[index]
    zmax, zmin = data.max(axis=0)[2], data.min(axis=0)[2]
    for ind in range(10):
        center_idx = random.randint(0, data.shape[0]-1)  # randomly select a crop center, then check if it is a valid choice
        center = data[center_idx]
        print("center", center)
        crop_min = np.array([center[0]-0.75, center[1]-0.75, zmin])
        crop_max = np.array([center[0]+0.75, center[1]+0.75, zmax])
        crop_ids = np.sum((data>=(crop_min-0.2)) * (data<=(crop_max+0.2)), axis=1) == 3
        if crop_ids.size == 0: continue
        crop_data, crop_label = data[crop_ids], label[crop_ids]
        if np.sum(crop_label>0)/crop_label.size < 0.7 and ind < 9:
            continue
        mask = np.sum((crop_data>=(crop_min-0.01)) * (crop_data<=(crop_max+0.01)), axis=1) == 3
        vidx = np.ceil((crop_data[mask]-crop_min) / (crop_max-crop_min) * [31,31,62])
        vidx = np.unique(vidx[:,0]*31*62 + vidx[:,1]*62 + vidx[:,2])
        # check if large portion of points are annotated, and the points occupy enough spaces
        if vidx.size*1./31/31/62 >= 0.02:
             break
    ids = np.random.choice(crop_label.size, 8192, replace=True)
    data_cropped = crop_data[ids]
    label_cropped = crop_label[ids]
    mask = mask[ids]
    points_batch, centroid, m = utils.normalize_point_cloud_param(data_cropped)
    points_batch = np.asarray([points_batch]).astype(np.float32)
    norm_all_data = (data - centroid) / m
    norm_all_data = np.asarray(norm_all_data).astype(np.float32)


    shape_dir = shape_txt_filename.split("/")[-1].split("_")[1].split(".")[0]
    fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
    if not os.path.exists(fdir):
        os.makedirs(fdir)
    print("write to fdir: ",fdir)
    neighbors_arr_lst, neighbors_mask_lst, neighbors_arr_up_lst, neighbors_mask_up_lst, \
    centers_arr_lst, centers_mask_lst, centers_lst = \
        assemble_tensor(points_batch,
            voxel_size_lst=[[0.04, 0.04, 0.04], [0.1, 0.1, 0.1], [0.25, 0.25, 0.25], [0.5,0.5,0.5]],
            grid_size_lst=[[50, 50, 50], [20, 20, 20], [8, 8, 8], [4,4,4]],
            lidar_coord=[1.0, 1.0, 1.0], max_p_grid_lst=[32, 32, 32, 32],
            max_o_grid_lst=[1024, 256, 64, 16], kernel_size_lst=[3, 3, 5, 3],
            stride_lst=[1, 1, 1, 1], single_padding_lst=[None, None, None, None],
            para=False, allow_sub=True, loc_weight=True, loc_within=True,
            up_voxel_size_lst=[[0.5,0.5,0.5], [0.25, 0.25, 0.25], [0.1, 0.1, 0.1], [0.04, 0.04, 0.04]],
            up_max_p_grid_lst=[4, 4, 4, 8], up_max_o_grid_lst=[8192, 8192, 8192, 8192],
            up_grid_size_lst=[[4, 4, 4], [8, 8, 8], [20, 20, 20], [50, 50, 50]],
            up_kernel_size_lst=[3, 3, 5, 3], up_stride_lst=[1, 1, 1, 1])
    fname = "raw" + ".txt"
    fname = os.path.join(fdir, fname)
    np.savetxt(fname, points_batch[0], delimiter=";")
    print("saved:", fname)

    fname = "norm_all" + ".txt"
    fname = os.path.join(fdir, fname)
    np.savetxt(fname, norm_all_data, delimiter=";")
    print("saved all points at :", fname)

    for i in range(21):
        cat_ind = np.where(label == i)
        if cat_ind[0].shape[0] > 0:
            print("cat_ind[0]",cat_ind[0].shape[0])
            fname = "cat{}_{}.txt".format(i,CLASS_NAMES[i])
            fname = os.path.join(fdir, fname)
            cat_pnts = norm_all_data[cat_ind]
            colors = np.tile(CLASS_COLORS[[i],...], (cat_pnts.shape[0],1))
            print("colors.shape", colors.shape)
            cat_pnts_colors = np.concatenate((cat_pnts, colors), axis=1)
            np.savetxt(fname, cat_pnts_colors, delimiter=";")
            print("saved cat{}-{} points at :{}".format(i,CLASS_NAMES[i],fname))

    for layer in range(len(centers_lst)):
        centers = centers_lst[layer][0]
        last_centers = centers_lst[layer-1][0] if layer!=0 else points_batch[0]
        print(centers.shape)
        fname = "ly_" + str(layer) + ".txt"
        fname = os.path.join(fdir, fname)
        np.savetxt(fname, centers, delimiter=";")
        print("saved:", fname)

        neighbors_arr = neighbors_arr_lst[layer][0]
        centers_mask = centers_mask_lst[layer][0]
        neighbors_mask = neighbors_mask_lst[layer][0]
        for cl_ind in range(neighbors_arr.shape[0]):
            associated_points = []
            if centers_mask[cl_ind]  == 0:
                break
            else:
                for pt_ind in range(neighbors_arr.shape[1]):
                    if neighbors_mask[cl_ind, pt_ind] == 0:
                        break
                    else:
                        associated_points.append(last_centers[neighbors_arr[cl_ind, pt_ind, ...]])
            associated_points = np.asarray(associated_points, dtype="float32")
            print("finish asso pts for {}, shape: {}".format(cl_ind, associated_points.shape))
            fname = "ass_lyr" + str(layer) + "_pt" + str(cl_ind) + ".txt"
            fname = os.path.join(fdir, fname)
            np.savetxt(fname, associated_points, delimiter=";")




