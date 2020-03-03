import ctypes
_ = ctypes.CDLL('../gridifyop/additional.so')
import sys
import os
import random
sys.path.append("../../data/")
sys.path.append("../../configs/")
sys.path.append("../../models/")
sys.path.append("../segmentation/")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import utils.utils as utils
from utils.ops import knn
import utils.data_utils as data_utils
 
import mxnet as mx
import mxnet.profiler as profiler
import time
import pickle
from mxnet.symbol import Gridify
from mxnet.symbol import GridifyUp
from utils.ops import batch_take
from gridify import assemble_tensor
import h5py 

ctx = [mx.gpu(0)]
npoints=2048
max_o_grid=256
max_p_grid=32
batch_size = 1
os.environ["MXNET_EXEC_BULK_EXEC_INFERENCE"] = "0"
profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    filename='gpu_profile_output_{}.json'.format(time.time()))

def get_seg_inputs():
    data = mx.symbol.Variable(name='data')
    actnum = mx.symbol.Variable(name='actnum')
    return data, actnum

def get_seg_up_inputs():
    downdata = mx.symbol.Variable(name='downdata')
    updata = mx.symbol.Variable(name='updata')
    downnum = mx.symbol.Variable(name='downnum')
    upnum = mx.symbol.Variable(name='upnum')
    return downdata, updata, downnum, upnum

def get_pnt_inputs():
    data = mx.symbol.Variable(name='data')
    return data

def get_pnt_up_inputs():
    downdata = mx.symbol.Variable(name='data')
    updata = mx.symbol.Variable(name='query')
    return downdata, updata


def get_symbol(max_p_grid=max_p_grid, max_o_grid=max_o_grid, kernel_size=3, stride=1, coord_shift=[1.0,1.0,1.0],
        voxel_size=[0.05, 0.05, 0.05], grid_size=[40, 40, 40], loc=1):
    data, actnum = get_seg_inputs()
    nebidx, nebidxmsk, cent, centmsk, actual_centnum = \
        Gridify(data, actnum, max_p_grid=max_p_grid, max_o_grid=max_o_grid, kernel_size=3, stride=1, coord_shift=[1.0,1.0,1.0],
        voxel_size=[0.05, 0.05, 0.05], grid_size=[40, 40, 40], loc=1)
        # voxel_size=[0.025, 0.025, 0.025], grid_size=[80, 80, 80])
    group = mx.symbol.Group([nebidx, nebidxmsk, cent, centmsk, actual_centnum])
    return group


def get_symbol_multi(max_p_grid_lst=[], max_o_grid_lst=[], kernel_size_lst=[], stride_lst=[], coord_shift=[1.00, 1.00, 1.00],
        voxel_size_lst=[], grid_size_lst=[], loc=1):
    data, actnum = get_seg_inputs()
    nebidx_lst = []
    nebidxmsk_lst = []
    centers_lst = []
    centmsk_lst = []
    for i in range(len(max_p_grid_lst)):
        nebidx, nebidxmsk, cent, centmsk, actual_centnum = \
        Gridify(data, actnum, max_p_grid=max_p_grid_lst[i], max_o_grid=max_o_grid_lst[i], kernel_size=kernel_size_lst[i], stride=stride_lst[i], coord_shift=coord_shift,
        voxel_size=voxel_size_lst[i], grid_size=grid_size_lst[i], loc=loc)
        data = cent
        actnum = actual_centnum
        nebidx_lst.append(nebidx)
        nebidxmsk_lst.append(nebidxmsk)
        centers_lst.append(cent)
        centmsk_lst.append(centmsk)
    group = mx.symbol.Group(nebidx_lst + nebidxmsk_lst + centers_lst + centmsk_lst)
    return group

def get_up_symbol(max_p_grid=max_p_grid, max_o_grid=npoints, kernel_size=3, coord_shift=[1.0, 1.0, 1.0],
         voxel_size=[0.04, 0.04, 0.04], grid_size=[50, 50, 50]):
    downdata, updata, downnum, upnum = get_seg_up_inputs()
    nebidx, nebidxmsk = \
        GridifyUp(downdata, updata, downnum, upnum, max_p_grid=max_p_grid, max_o_grid=npoints, kernel_size=3,
            coord_shift=[1.0, 1.0, 1.0],
         voxel_size=[0.04, 0.04, 0.04], grid_size=[50, 50, 50])
        # voxel_size=[0.025, 0.025, 0.025], grid_size=[80, 80, 80])
    group = mx.symbol.Group([nebidx, nebidxmsk])
    return group

def get_pnt_symbol():
    data = get_pnt_inputs()
    centeroids_ids = mx.sym.contrib.FarthestPointSampling(data=data, npoints=max_o_grid, name='/fps_idx')
    centeroids = batch_take(data, centeroids_ids, shape=(1, max_o_grid, npoints, 3), scope='/fps_batch_take')
    grouped_ids = mx.sym.contrib.BallQuery(data, centeroids, radius=0.05, nsample=max_p_grid)
    return grouped_ids


def get_pnt_up3_symbol():
    data, query = get_pnt_up_inputs()
    dist1, ids1 = mx.sym.contrib.ThreeNN(query, data)  # (B, M, k)
    group = mx.symbol.Group([dist1, ids1])
    return group

def get_pnt_up_symbol():
    data, query = get_pnt_up_inputs()
    dist2, ids2 = knn(query, data, (npoints, max_o_grid), max_p_grid)  # (B, M, k)
    group = mx.symbol.Group([dist2, ids2])
    return group

if __name__ == "__main__":

    # vis_root = "../single_vis/"
    # single_path = '../data/modelnet40_normal_resampled/radio/radio_0013.txt'
    # db_name, cls_name, shape_txt_filename = "modelnet40", "radio", single_path
    # point_set = np.loadtxt(shape_txt_filename, delimiter=',')[:min(8192,npoints), ...]
    # # if npoints > 8192:
    # #     point_set = np.concatenate([point_set, point_set,
    # #                                 point_set, point_set,
    # #                                 point_set, point_set,
    # #                                 point_set, point_set,
    # #                                 point_set, point_set], axis=0)
    # #     point_set = point_set[np.random.shuffle(np.arange(8192*10)),...]
    # #     point_set = point_set[0, :npoints, ...]
    # #     db_name, cls_name, shape_txt_filename = "modelnet40", "radio_mix", '../data/modelnet40_normal_resampled/radio/radio_repeat{}.txt'.format(npoints)

    # point_set = utils.normalize_point_cloud(point_set[:, 0:3])
    # points_batch = np.asarray([point_set for i in range(batch_size)]).astype(np.float32)
    # points_w_batch = np.concatenate([points_batch, np.ones([batch_size, npoints, 1], dtype=np.float32)], axis = 2)
    # shape_dir = shape_txt_filename.split("/")[-1].split("_")[1].split(".")[0]
    # fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
    # if not os.path.exists(fdir):
    #     os.makedirs(fdir)
    # print("write to fdir: ",fdir)
    # ctx = [mx.gpu(0)]
    # module = mx.mod.Module(get_symbol(), context=ctx, data_names=['data', 'actnum'], label_names=None)
    # module.bind(data_shapes=[('data', (batch_size, npoints, 4)), ('actnum', (batch_size, 1), "int32")], label_shapes=None)
    # module.init_params(initializer=mx.init.Xavier())

    # up_module = mx.mod.Module(get_up_symbol(), context=ctx, data_names=['downdata', 'updata', 'downnum', 'upnum'], label_names=None)
    # up_module.bind(data_shapes=[('downdata', (batch_size, max_o_grid, 4)), ('updata', (batch_size, npoints, 4)),
    #                          ('downnum', (batch_size, 1), "int32"), ('upnum', (batch_size, 1), "int32")], label_shapes=None)
    # up_module.init_params(initializer=mx.init.Xavier())
    # #
    # #
    module_com = mx.mod.Module(get_pnt_symbol(), context=ctx, data_names=['data'], label_names=None)
    module_com.bind(data_shapes=[('data', (1, npoints, 3))], label_shapes=None)
    module_com.init_params(initializer=mx.init.Xavier())
    # # #
    # # up_module_com = mx.mod.Module(get_pnt_up_symbol(), context=ctx, data_names=['data', 'query'], label_names=None)
    # # up_module_com.bind(data_shapes=[('data', (1, max_o_grid, 3)), ('query', (1, npoints, 3))], label_shapes=None)
    # # up_module_com.init_params(initializer=mx.init.Xavier())
    # #
    # # up_module3_com = mx.mod.Module(get_pnt_up3_symbol(), context=ctx, data_names=['data', 'query'], label_names=None)
    # # up_module3_com.bind(data_shapes=[('data', (1, max_o_grid, 3)), ('query', (1, npoints, 3))], label_shapes=None)
    # # up_module3_com.init_params(initializer=mx.init.Xavier())
    # sum_time = 0
    # count_infe = 0
    # for i in range(1):
    #     print("down,i:", i)
    #     batch = points_w_batch
    #     input = mx.io.DataBatch(data=[mx.ndarray.array(batch), mx.ndarray.ones((batch_size,1), dtype="int32")*npoints])
    #     tic = time.time()
    #     module.forward(input, is_train=False)
    #     mx.nd.waitall()
    #     if i > 2:
    #         sum_time += time.time() - tic
    #         count_infe += 1
    #     neighbors_arr = module.get_outputs()[0].asnumpy()
    #     neighbors_mask = module.get_outputs()[1].asnumpy()
    #     centers_arr = module.get_outputs()[2].asnumpy()
    #     centers_mask = module.get_outputs()[3].asnumpy()
    #     actual_centnum = module.get_outputs()[4].asnumpy()
    #     #
    #     print("neighbors_arr", neighbors_arr.shape, neighbors_arr)
    #     print("neighbors_mask", neighbors_mask.shape, neighbors_mask)
    #     print("centers_arr", centers_arr.shape, centers_arr)
    #     print("centers_mask", centers_mask.shape, centers_mask)
    #     print("actual_centnum", actual_centnum.shape, actual_centnum)
    #     for b in range(batch_size):
    #         fdir_b = os.path.join(fdir, str(b))
    #         if not os.path.exists(fdir_b):
    #             os.makedirs(fdir_b)
    #         fname = "raw" + ".txt"
    #         fname = os.path.join(fdir_b, fname)
    #         np.savetxt(fname, batch[b], delimiter=";")
    #         print("saved:", fname)
    #         lidx = 1
    #         last_centers = batch[b]
    #         fname = "ly_" + str(lidx) + ".txt"
    #         fname = os.path.join(fdir_b, fname)
    #         centers = []
    #         for j in range(centers_arr.shape[1]):
    #             if centers_mask[b,j] < 1:
    #                 break
    #             centers.append(centers_arr[b,j,...])
    #         centers = np.asarray(centers, dtype="float32")
    #         np.savetxt(fname, centers, delimiter=";")
    #         print("saved:", fname)
        
    #         for cl_ind in range(neighbors_arr.shape[1]):
    #             associated_points = []
    #             if centers_mask[b, cl_ind] == 0:
    #                 break
    #             else:
    #                 for pt_ind in range(neighbors_arr.shape[2]):
    #                     if neighbors_mask[b, cl_ind, pt_ind] == 0:
    #                         break
    #                     else:
    #                         associated_points.append(last_centers[int(neighbors_arr[b, cl_ind, pt_ind, ...])])
    #             associated_points = np.asarray(associated_points, dtype="float32")
    #             # print("finish asso pts for {}, shape: {}".format(cl_ind, associated_points.shape))
    #             fname = "ass_lyr" + str(lidx) + "_pt" + str(cl_ind) + ".txt"
    #             fname = os.path.join(fdir_b, fname)
    #             np.savetxt(fname, associated_points, delimiter=";")
    #             target_point = centers[cl_ind]
    #             fpname = "ass_lyr" + str(lidx) + "_pts" + str(cl_ind) + "_target.txt"
    #             fpname = os.path.join(fdir_b, fpname)
    #             np.savetxt(fpname, [target_point], delimiter=";")



    #     #
    #     # input = mx.io.DataBatch(data=[mx.ndarray.array(points_batch)])
    #     # module_com.forward(input, is_train=False)
    #     # neighbors_arr_pnt = module_com.get_outputs()[0].asnumpy()
    #     # centers = np.take(points_batch, neighbors_arr_pnt)

    #     # input = mx.io.DataBatch(data=[mx.ndarray.array(centers[:,:,:3]), mx.ndarray.array(points_batch[:,:,:3])])
    #     # up_module_com.forward(input, is_train=False)
    #     # dist1, ids1 = \
    #     #     up_module_com.get_outputs()[0].asnumpy(), \
    #     #     up_module_com.get_outputs()[1].asnumpy()

    #     # input = mx.io.DataBatch(data=[mx.ndarray.array(centers[:,:,:3]), mx.ndarray.array(points_batch[:,:,:3])])
    #     # up_module3_com.forward(input, is_train=False)
    #     # dist2, ids2 = \
    #     #     up_module3_com.get_outputs()[0].asnumpy(), \
    #     #     up_module3_com.get_outputs()[1].asnumpy()

    # # print("gridify down inference time:", sum_time/count_infe)
    # # sum_time = 0
    # # count_infe = 0
    # # # profiler.set_state('run')
    # # for i in range(400):
    # #     print("up,i:", i)
    # #     input = mx.io.DataBatch(data=[mx.ndarray.array(centers_arr), mx.ndarray.array(batch),
    # #                                   mx.ndarray.array(actual_centnum, dtype="int32"), mx.ndarray.ones((batch_size,1), dtype="int32")*npoints])
    # #     tic = time.time()
    # #     up_module.forward(input, is_train=False)
    # #     mx.nd.waitall()
    # #     if i > 2:
    # #         sum_time += time.time() - tic
    # #         count_infe += 1
    # #     neighbors_up_arr = up_module.get_outputs()[0].asnumpy()
    # #     neighbors_up_mask = up_module.get_outputs()[1].asnumpy()

    #     # for b in range(batch_size):
    #     #     fdir_b = os.path.join(fdir, str(b))
    #     #     if not os.path.exists(fdir_b):
    #     #         os.makedirs(fdir_b)
    #     #     last_centers = centers_arr[b]
    #     #     for cl_ind in range(neighbors_up_arr.shape[1]):
    #     #         associated_points = []
    #     #         for pt_ind in range(neighbors_up_arr.shape[2]):
    #     #             if neighbors_up_mask[b, cl_ind, pt_ind] == 0:
    #     #                 break
    #     #             else:
    #     #                 associated_points.append(last_centers[int(neighbors_up_arr[b, cl_ind, pt_ind])])
    #     #         associated_points = np.asarray(associated_points, dtype="float32")
    #     #         # print("finish asso pts for {}, shape: {}".format(cl_ind, associated_points.shape))
    #     #         fname = "up_ass_lyr" + str(lidx) + "_pt" + str(cl_ind) + ".txt"
    #     #         fname = os.path.join(fdir_b, fname)
    #     #         np.savetxt(fname, associated_points, delimiter=";")
    #     #         target_point = batch[b, cl_ind]
    #     #         fpname = "up_ass_lyr" + str(lidx) + "_pts" + str(cl_ind) + "_target.txt"
    #     #         fpname = os.path.join(fdir_b, fpname)
    #     #         np.savetxt(fpname, [target_point], delimiter=";")

    #     # input = mx.io.DataBatch(data=[mx.ndarray.array(centers[:,:,:3]), mx.ndarray.array(points_batch[:,:,:3])])
    #     # up_module_com.forward(input, is_train=False)
    #     # dist1, ids1 = \
    #     #     up_module_com.get_outputs()[0].asnumpy(), \
    #     #     up_module_com.get_outputs()[1].asnumpy()
    #     #
    #     # input = mx.io.DataBatch(data=[mx.ndarray.array(centers[:,:,:3]), mx.ndarray.array(points_batch[:,:,:3])])
    #     # up_module3_com.forward(input, is_train=False)
    #     # dist2, ids2 = \
    #     #     up_module3_com.get_outputs()[0].asnumpy(), \
    #     #     up_module3_com.get_outputs()[1].asnumpy()

    # # profiler.set_state('stop')
    # # profiler.dump()

    # # print("gridify up inference time:", sum_time / count_infe)


###### single shape of shapenet part:

vis_root = "../single_vis/"
single_path = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/03467517/7eee3b79e053759143891ae68a82472e.txt'
db_name, cls_name, shape_txt_filename = "shapenet", "guitar", single_path
point_set = np.loadtxt(shape_txt_filename, delimiter=' ')[:min(8192,npoints), ...]
label = point_set[np.newaxis,:, [6]]
normal = point_set[np.newaxis,:, 3:6]
point_set = utils.normalize_point_cloud(point_set[:, 0:3])
points_batch = np.asarray([point_set for i in range(batch_size)]).astype(np.float32)
points_w_batch = np.concatenate([points_batch, np.ones([batch_size, npoints, 1], dtype=np.float32)], axis = 2)
shape_dir = shape_txt_filename.split("/")[-1].split(".")[0]
fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
if not os.path.exists(fdir):
    os.makedirs(fdir)
print("write to fdir: ",fdir)

CLASS_COLORS = np.asarray([(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48)])

voxel_size_lst = [[0.04, 0.04, 0.04], [0.4, 0.2, 0.2]]
grid_size_lst=[[50, 50, 50], [8, 10, 10]]
lidar_coord=[0.3, 1.0, 1.0]
max_p_grid_lst=[64, 32]
max_o_grid_lst=[256, 32]
kernel_size_lst=[5, 3]
stride_lst=[1, 1]
single_padding_lst=[None, None]
para=False
allow_sub=True
loc_weight=True
loc_within=True
up_voxel_size_lst=[[0.25, 0.25, 0.25], [0.04, 0.04, 0.04]]
up_max_p_grid_lst=[32, 32]
up_max_o_grid_lst=[256, 2048]
up_grid_size_lst=[[8, 8, 8], [50, 50, 50]]
up_kernel_size_lst=[3, 5]
up_stride_lst=[1, 1]
points_w_batch = np.concatenate([points_batch, np.ones([batch_size, npoints, 1], dtype=np.float32)], axis=2)

input = mx.io.DataBatch(data=[mx.ndarray.array(points_w_batch),
                              mx.ndarray.ones((batch_size, 1), dtype="int32") * npoints])
symbol = get_symbol_multi(max_p_grid_lst=max_p_grid_lst, max_o_grid_lst=max_o_grid_lst, kernel_size_lst=kernel_size_lst, stride_lst=stride_lst, coord_shift=lidar_coord,
                          voxel_size_lst=voxel_size_lst, grid_size_lst=grid_size_lst, loc=1 if True else 0)
module = mx.mod.Module(symbol, context=ctx, data_names=['data', 'actnum'], label_names=None)
module.bind(data_shapes=[('data', (batch_size, npoints, 4)), ('actnum', (batch_size, 1), "int32")],
            label_shapes=None)
module.init_params(initializer=mx.init.Xavier())

module.forward(input, is_train=False)
neighbors_arr_lst = [module.get_outputs()[i].asnumpy() for i in range(2)]
neighbors_mask_lst = [module.get_outputs()[i].asnumpy() for i in range(2,4)]
centers_lst = [module.get_outputs()[i].asnumpy() for i in range(4,6)]
centers_mask_lst = [module.get_outputs()[i].asnumpy() for i in range(6,8)]

sum_time = 0
count_infe = 0
print("label.shape", label.shape)
for b in range(batch_size):
    for i in range(19, 22): # 19, 20, 21
        cat_ind = np.where(label[b, :,0] == i)[0]
        if cat_ind.shape[0] > 0:
            print("cat_ind",cat_ind.shape)
            fname = "cat{}.txt".format(i)
            fname = os.path.join(fdir, fname)
            print("points_batch.shape", points_batch.shape)
            cat_pnts = points_batch[b, cat_ind, :]
            print("cat_pnts", cat_pnts.shape)
            colors = np.tile(CLASS_COLORS[[i-19],...], (cat_pnts.shape[0],1))
            print("colors.shape", colors.shape)
            cat_pnts_colors = np.concatenate((cat_pnts, colors), axis=1)
            np.savetxt(fname, cat_pnts_colors, delimiter=";")
            print("saved cat{} points at :{}".format(i,fname))

    fdir_b = fdir
    if not os.path.exists(fdir_b):
        os.makedirs(fdir_b)
    fname = "rawxyz" + ".txt"
    fname = os.path.join(fdir_b, fname)
    np.savetxt(fname, points_batch[b], delimiter=";")
    print("saved:", fname)

    # fname = "rawnormal" + ".txt"
    # fname = os.path.join(fdir_b, fname)
    # print("normal.shape",normal.shape)
    # color1 = (normal[b,:, 2,np.newaxis] + 1) * 255
    # color2 = np.ones_like(normal[b,:, 2,np.newaxis]) * 128 #(normal[b,:, 3,np.newaxis] + 1) * 255
    # color3 =  np.ones_like(normal[b,:, 2,np.newaxis]) * 128 # abs(normal[b,:, 2,np.newaxis]) * (255-128)
    # print("colors.shape", color1.shape,color2.shape)
    # cat_normal_colors = np.concatenate((points_batch[b], color1, color2,color3 ), axis=1)
    # np.savetxt(fname, cat_normal_colors, delimiter=";")
    # print("saved cat normal{} points at :{}".format(i,fname))

    fname = "rawnormal" + ".txt"
    fname = os.path.join(fdir_b, fname)
    print("normal.shape",normal.shape)
    x_norm = points_batch[b,:, 0,np.newaxis] - np.mean(points_batch[b,:, 0,np.newaxis])
    y_norm = points_batch[b,:, 1,np.newaxis] - np.mean(points_batch[b,:, 1,np.newaxis])
    z_norm = points_batch[b,:, 2,np.newaxis] - np.mean(points_batch[b,:, 2,np.newaxis]) 
    x_range = np.max(x_norm)
    y_range = np.max(y_norm)
    z_range = np.max(z_norm)
    color1 = (x_norm / x_range) * 16 + 64 # abs(normal[b,:, 2,np.newaxis]) * (255-128)
    color2 = (y_norm / y_range) * 64 + 128 #(normal[b,:, 3,np.newaxis] + 1) * 255
    color3 = (z_norm / z_range) * 128 + 128
    print("colors.shape", color1.shape,color2.shape)
    cat_normal_colors = np.concatenate((points_batch[b], color1, color2,color3 ), axis=1)
    np.savetxt(fname, cat_normal_colors, delimiter=";")
    print("saved cat normal{} points at :{}".format(i,fname))

    last_centers = points_batch[b]

    for lidx in range(len(grid_size_lst)):
        layer = lidx
        fname = "ly_" + str(lidx) + ".txt"
        fname = os.path.join(fdir_b, fname)
        centers_arr = centers_lst[lidx]

        neighbors_arr = neighbors_arr_lst[layer]
        centers_mask = centers_mask_lst[layer]
        neighbors_mask = neighbors_mask_lst[layer]

        centers = []
        for j in range(centers_arr.shape[1]):
            if centers_mask[b,j] < 1:
                break
            centers.append(centers_arr[b,j,...])
        centers = np.asarray(centers, dtype="float32")
        np.savetxt(fname, centers, delimiter=";")
        print("saved:", fname)


        for cl_ind in range(neighbors_arr.shape[1]):
            associated_points = []
            if centers_mask[b, cl_ind] == 0:
                break
            else:
                for pt_ind in range(neighbors_arr.shape[2]):
                    if neighbors_mask[b, cl_ind, pt_ind] == 0:
                        break
                    else:
                        associated_points.append(last_centers[int(neighbors_arr[b, cl_ind, pt_ind, ...])])
            associated_points = np.asarray(associated_points, dtype="float32")
            # print("finish asso pts for {}, shape: {}".format(cl_ind, associated_points.shape))
            fname = "ass_lyr" + str(lidx) + "_pt" + str(cl_ind) + ".txt"
            fname = os.path.join(fdir_b, fname)
            np.savetxt(fname, associated_points, delimiter=";")
            target_point = centers[cl_ind]
            fpname = "ass_lyr" + str(lidx) + "_pts" + str(cl_ind) + "_target.txt"
            fpname = os.path.join(fdir_b, fpname)
            np.savetxt(fpname, [target_point], delimiter=";")
        last_centers = centers



        #
        # input = mx.io.DataBatch(data=[mx.ndarray.array(points_batch)])
        # module_com.forward(input, is_train=False)
        # neighbors_arr_pnt = module_com.get_outputs()[0].asnumpy()
        # centers = np.take(points_batch, neighbors_arr_pnt)

        # input = mx.io.DataBatch(data=[mx.ndarray.array(centers[:,:,:3]), mx.ndarray.array(points_batch[:,:,:3])])
        # up_module_com.forward(input, is_train=False)
        # dist1, ids1 = \
        #     up_module_com.get_outputs()[0].asnumpy(), \
        #     up_module_com.get_outputs()[1].asnumpy()

        # input = mx.io.DataBatch(data=[mx.ndarray.array(centers[:,:,:3]), mx.ndarray.array(points_batch[:,:,:3])])
        # up_module3_com.forward(input, is_train=False)
        # dist2, ids2 = \
        #     up_module3_com.get_outputs()[0].asnumpy(), \
        #     up_module3_com.get_outputs()[1].asnumpy()





# # single file of scannet
#     vis_root = "/home/xharlie/dev/GGCN-corp/single_vis/"
#     CLASS_NAMES = {0:'unknown', 1: 'wall', 2: 'floor', 3: 'chair', 4: 'table', 5: 'desk', 6: 'bed', 7: 'bookshelf', 8: 'sofa',
#                    9: 'sink', 10: 'bathtub', 11: 'toilet', 12: 'curtain', 13: 'counter', 14: 'door', 15: 'window',
#                    16: 'shower_curtain', 17: 'refrigerator', 18: 'picture', 19: 'cabinet', 20: 'other'}
#     CLASS_COLORS = np.asarray([(0, 0, 0), (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
#                                (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190),
#                                (0, 128, 128), (230, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0),
#                                (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128)]).astype("float32")

#     single_path = '/home/xharlie/dev/GGCN-corp/data/scannet/scannet_test.pickle'
#     db_name, cls_name, shape_txt_filename = "scannet-gpu", "10", single_path
#     index = int(cls_name)
#     with open(single_path, "rb") as f:
#         data = pickle.load(f, encoding="bytes")[index]
#         label = pickle.load(f, encoding="bytes")[index]
#     zmax, zmin = data.max(axis=0)[2], data.min(axis=0)[2]
#     for ind in range(10):
#         center_idx = random.randint(0, data.shape[0]-1)  # randomly select a crop center, then check if it is a valid choice
#         center = data[center_idx]
#         print("center", center)
#         crop_min = np.array([center[0]-0.75, center[1]-0.75, zmin])
#         crop_max = np.array([center[0]+0.75, center[1]+0.75, zmax])
#         crop_ids = np.sum((data>=(crop_min-0.2)) * (data<=(crop_max+0.2)), axis=1) == 3
#         if crop_ids.size == 0: continue
#         crop_data, crop_label = data[crop_ids], label[crop_ids]
#         if np.sum(crop_label>0)/crop_label.size < 0.7 and ind < 9:
#             continue
#         mask = np.sum((crop_data>=(crop_min-0.01)) * (crop_data<=(crop_max+0.01)), axis=1) == 3
#         vidx = np.ceil((crop_data[mask]-crop_min) / (crop_max-crop_min) * [31,31,62])
#         vidx = np.unique(vidx[:,0]*31*62 + vidx[:,1]*62 + vidx[:,2])
#         # check if large portion of points are annotated, and the points occupy enough spaces
#         if vidx.size*1./31/31/62 >= 0.02:
#              break
#     ids = np.random.choice(crop_label.size, npoints, replace=True)
#     data_cropped = crop_data[ids]
#     label_cropped = crop_label[ids]
#     mask = mask[ids]
#     points_batch, centroid, m = utils.normalize_point_cloud_param(data_cropped)
#     points_batch = np.asarray([points_batch]).astype(np.float32)
#     norm_all_data = (data - centroid) / m
#     norm_all_data = np.asarray(norm_all_data).astype(np.float32)

# #   GPU VERSION
#     shape_dir = shape_txt_filename.split("/")[-1].split("_")[1].split(".")[0]
#     fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
#     if not os.path.exists(fdir):
#         os.makedirs(fdir)
#     print("write to fdir: ",fdir)
#     ctx = [mx.gpu(0)]

#     voxel_size_lst = [[0.04, 0.04, 0.04], [0.1, 0.1, 0.1], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5]]
#     grid_size_lst=[[50, 50, 50], [20, 20, 20], [8, 8, 8], [4,4,4]]
#     lidar_coord=[1.0, 1.0, 1.0]
#     max_p_grid_lst=[32, 32, 32, 32]
#     max_o_grid_lst=[1024, 256, 64, 16]
#     kernel_size_lst=[3, 3, 5, 3]
#     stride_lst=[1, 1, 1, 1]
#     single_padding_lst=[None, None, None, None]
#     para=False
#     allow_sub=True
#     loc_weight=True
#     loc_within=True
#     up_voxel_size_lst=[[0.5,0.5,0.5], [0.25, 0.25, 0.25], [0.1, 0.1, 0.1], [0.04, 0.04, 0.04]]
#     up_max_p_grid_lst=[4, 4, 4, 8]
#     up_max_o_grid_lst=[8192, 8192, 8192, 8192]
#     up_grid_size_lst=[[4, 4, 4], [8, 8, 8], [20, 20, 20], [50, 50, 50]]
#     up_kernel_size_lst=[3, 3, 5, 3]
#     up_stride_lst=[1, 1, 1, 1]
#     points_w_batch = np.concatenate([points_batch, np.ones([batch_size, npoints, 1], dtype=np.float32)], axis=2)

#     input = mx.io.DataBatch(data=[mx.ndarray.array(points_w_batch),
#                                   mx.ndarray.ones((batch_size, 1), dtype="int32") * npoints])
#     symbol = get_symbol_multi(max_p_grid_lst=max_p_grid_lst, max_o_grid_lst=max_o_grid_lst, kernel_size_lst=kernel_size_lst, stride_lst=stride_lst, coord_shift=lidar_coord,
#                               voxel_size_lst=voxel_size_lst, grid_size_lst=grid_size_lst, loc=1 if True else 0)
#     module = mx.mod.Module(symbol, context=ctx, data_names=['data', 'actnum'], label_names=None)
#     module.bind(data_shapes=[('data', (batch_size, npoints, 4)), ('actnum', (batch_size, 1), "int32")],
#                 label_shapes=None)
#     module.init_params(initializer=mx.init.Xavier())

#     module.forward(input, is_train=False)
#     neighbors_arr_lst = [module.get_outputs()[i].asnumpy() for i in range(4)]
#     neighbors_mask_lst = [module.get_outputs()[i].asnumpy() for i in range(4,8)]
#     centers_lst = [module.get_outputs()[i].asnumpy() for i in range(8,12)]
#     centers_mask_lst = [module.get_outputs()[i].asnumpy() for i in range(12,16)]



#         # up_symbol = get_up_symbol(max_p_grid=up_max_p_grid_lst[i], max_o_grid=up_max_o_grid_lst[i], kernel_size=3, coord_shift=[1.0, 1.0, 1.0],
#         #  voxel_size=[0.04, 0.04, 0.04], grid_size=[50, 50, 50])
#         # up_module = mx.mod.Module(up_symbol, context=ctx,
#         #             data_names=['downdata', 'updata', 'downnum', 'upnum'], label_names=None)
#         # up_module.bind(data_shapes=[('downdata', (batch_size, max_o_grid, 4)), ('updata', (batch_size, npoints, 4)),
#         #     ('downnum', (batch_size, 1), "int32"), ('upnum', (batch_size, 1), "int32")], label_shapes=None)
#         # up_module.init_params(initializer=mx.init.Xavier())
#         #
#         # input = mx.io.DataBatch(data=[mx.ndarray.array(centers_arr), mx.ndarray.array(points_w_batch),
#         #         mx.ndarray.array(actual_centnum, dtype="int32"), mx.ndarray.ones((batch_size,1), dtype="int32")*npoints])
#         # up_module.forward(input, is_train=False)
#         # neighbors_up_arr = up_module.get_outputs()[0].asnumpy()
#         # neighbors_up_mask = up_module.get_outputs()[1].asnumpy()

#     fname = "raw" + ".txt"
#     fname = os.path.join(fdir, fname)
#     np.savetxt(fname, points_batch[0], delimiter=";")
#     print("saved:", fname)

#     fname = "norm_all" + ".txt"
#     fname = os.path.join(fdir, fname)
#     np.savetxt(fname, norm_all_data, delimiter=";")
#     print("saved all points at :", fname)

#     for i in range(21):
#         cat_ind = np.where(label == i)
#         if cat_ind[0].shape[0] > 0:
#             print("cat_ind[0]",cat_ind[0].shape[0])
#             fname = "cat{}_{}.txt".format(i,CLASS_NAMES[i])
#             fname = os.path.join(fdir, fname)
#             cat_pnts = norm_all_data[cat_ind]
#             colors = np.tile(CLASS_COLORS[[i],...], (cat_pnts.shape[0],1))
#             print("colors.shape", colors.shape)
#             cat_pnts_colors = np.concatenate((cat_pnts, colors), axis=1)
#             np.savetxt(fname, cat_pnts_colors, delimiter=";")
#             print("saved cat{}-{} points at :{}".format(i,CLASS_NAMES[i],fname))

#     for layer in range(len(centers_lst)):
#         centers = centers_lst[layer][0]
#         last_centers = centers_lst[layer-1][0] if layer!=0 else points_batch[0]
#         print(centers.shape)
#         fname = "ly_" + str(layer) + ".txt"
#         fname = os.path.join(fdir, fname)
#         np.savetxt(fname, centers, delimiter=";")
#         print("saved:", fname)

#         neighbors_arr = neighbors_arr_lst[layer][0]
#         centers_mask = centers_mask_lst[layer][0]
#         neighbors_mask = neighbors_mask_lst[layer][0]
#         for cl_ind in range(neighbors_arr.shape[0]):
#             associated_points = []
#             if centers_mask[cl_ind]  == 0:
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

# # CPU Version
#     db_name, cls_name, shape_txt_filename = "scannet", "10", single_path
#     shape_dir = shape_txt_filename.split("/")[-1].split("_")[1].split(".")[0]
#     fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
#     if not os.path.exists(fdir):
#         os.makedirs(fdir)
#     print("write to fdir: ",fdir)
#     neighbors_arr_lst, neighbors_mask_lst, neighbors_arr_up_lst, neighbors_mask_up_lst, \
#     centers_arr_lst, centers_mask_lst, centers_lst = \
#         assemble_tensor(points_batch,
#             voxel_size_lst=[[0.04, 0.04, 0.04], [0.1, 0.1, 0.1], [0.25, 0.25, 0.25], [0.5,0.5,0.5]],
#             grid_size_lst=[[50, 50, 50], [20, 20, 20], [8, 8, 8], [4,4,4]],
#             lidar_coord=[1.0, 1.0, 1.0], max_p_grid_lst=[32, 32, 32, 32],
#             max_o_grid_lst=[1024, 256, 64, 16], kernel_size_lst=[3, 3, 5, 3],
#             stride_lst=[1, 1, 1, 1], single_padding_lst=[None, None, None, None],
#             para=False, allow_sub=True, loc_weight=True, loc_within=True,
#             up_voxel_size_lst=[[0.5,0.5,0.5], [0.25, 0.25, 0.25], [0.1, 0.1, 0.1], [0.04, 0.04, 0.04]],
#             up_max_p_grid_lst=[4, 4, 4, 8], up_max_o_grid_lst=[8192, 8192, 8192, 8192],
#             up_grid_size_lst=[[4, 4, 4], [8, 8, 8], [20, 20, 20], [50, 50, 50]],
#             up_kernel_size_lst=[3, 3, 5, 3], up_stride_lst=[1, 1, 1, 1])
#     fname = "raw" + ".txt"
#     fname = os.path.join(fdir, fname)
#     np.savetxt(fname, points_batch[0], delimiter=";")
#     print("saved:", fname)

#     fname = "norm_all" + ".txt"
#     fname = os.path.join(fdir, fname)
#     np.savetxt(fname, norm_all_data, delimiter=";")
#     print("saved all points at :", fname)

#     for i in range(21):
#         cat_ind = np.where(label == i)
#         if cat_ind[0].shape[0] > 0:
#             print("cat_ind[0]",cat_ind[0].shape[0])
#             fname = "cat{}_{}.txt".format(i,CLASS_NAMES[i])
#             fname = os.path.join(fdir, fname)
#             cat_pnts = norm_all_data[cat_ind]
#             colors = np.tile(CLASS_COLORS[[i],...], (cat_pnts.shape[0],1))
#             print("colors.shape", colors.shape)
#             cat_pnts_colors = np.concatenate((cat_pnts, colors), axis=1)
#             np.savetxt(fname, cat_pnts_colors, delimiter=";")
#             print("saved cat{}-{} points at :{}".format(i,CLASS_NAMES[i],fname))

#     for layer in range(len(centers_lst)):
#         centers = centers_lst[layer][0]
#         last_centers = centers_lst[layer-1][0] if layer!=0 else points_batch[0]
#         print(centers.shape)
#         fname = "ly_" + str(layer) + ".txt"
#         fname = os.path.join(fdir, fname)
#         np.savetxt(fname, centers, delimiter=";")
#         print("saved:", fname)

#         neighbors_arr = neighbors_arr_lst[layer][0]
#         centers_mask = centers_mask_lst[layer][0]
#         neighbors_mask = neighbors_mask_lst[layer][0]
#         for cl_ind in range(neighbors_arr.shape[0]):
#             associated_points = []
#             if centers_mask[cl_ind]  == 0:
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





# single file of s3dis
#     vis_root = "/home/xharlie/dev/GGCN-corp/single_vis/"
#     single_path = '/home/xharlie/dev/GGCN-corp/data/3DIS/prepare_label_rgb/Area_5/office_12/half_0.h5'
#     db_name, cls_name, shape_txt_filename = "s3disA5", "12", single_path
#     index = int(cls_name)
#     with h5py.File(single_path, 'r') as data_all:
#         data = data_all['data'][...].astype(np.float32)
#         point_nums= data_all['data_num'][...].astype(np.int32)
#         labels_seg =data_all['label_seg'][...].astype(np.int32)
     
#     dataxyz = data[0, :point_nums[0], :3]                
#     norm_all_data, _, _ = data_utils.filter_extre(dataxyz, None, None, 2/2)
            
#     norm_all_data =  utils.normalize_point_cloud(norm_all_data, double_axis=1)
#     pt = norm_all_data.shape[0]
#     batch_size = 1
#     points_batch = np.zeros((1, 8192, 3))
#     points_batch[0,:pt,:] = norm_all_data
# #   GPU VERSION
#     shape_dir = shape_txt_filename.split("/")[-1].split("_")[1].split(".")[0]
#     fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
#     if not os.path.exists(fdir):
#         os.makedirs(fdir)
#     print("write to fdir: ",fdir)
#     ctx = [mx.gpu(0)]

#     voxel_size_lst = [[0.05, 0.05, 0.05], [0.133333,0.133333,0.133333], [0.4, 0.4, 0.4]]
#     grid_size_lst = [[40, 80, 40], [15, 30, 15], [5, 10, 5]]
#     lidar_coord = [1.0, 2.0, 1.0]
#     max_p_grid_lst = [64, 32, 32]
#     max_o_grid_lst = [1024, 256, 24]
#     kernel_size_lst=[3, 3, 3]
#     stride_lst=[1, 1, 1]
#     single_padding_lst=[None, None, None]
#     para=False
#     allow_sub=True
#     loc_weight=True
#     loc_within=True

#     up_voxel_size_lst = [[0.4, 0.4, 0.4], [0.133333,0.133333,0.133333], [0.05, 0.05, 0.05]]
#     up_grid_size_lst = [[5, 10, 5], [15, 30, 15], [40, 80, 40]]
#     up_max_p_grid_lst= [5, 5, 5]
#     up_max_o_grid_lst= [256, 1024, 4096]
#     up_kernel_size_lst=[3, 3, 3]
#     up_stride_lst=[1, 1, 1]
#     points_w_batch = np.concatenate([points_batch, np.ones([batch_size, npoints, 1], dtype=np.float32)], axis=2)

#     input = mx.io.DataBatch(data=[mx.ndarray.array(points_w_batch),
#                                   mx.ndarray.ones((batch_size, 1), dtype="int32") * pt])
#     symbol = get_symbol_multi(max_p_grid_lst=max_p_grid_lst, max_o_grid_lst=max_o_grid_lst, kernel_size_lst=kernel_size_lst, stride_lst=stride_lst, coord_shift=lidar_coord,
#                               voxel_size_lst=voxel_size_lst, grid_size_lst=grid_size_lst, loc=1 if True else 0)
#     module = mx.mod.Module(symbol, context=ctx, data_names=['data', 'actnum'], label_names=None)
#     module.bind(data_shapes=[('data', (batch_size, npoints, 4)), ('actnum', (batch_size, 1), "int32")],
#                 label_shapes=None)
#     module.init_params(initializer=mx.init.Xavier())

#     module.forward(input, is_train=False)
#     print(len(module.get_outputs()))
#     neighbors_arr_lst = [module.get_outputs()[i].asnumpy() for i in range(3)]
#     neighbors_mask_lst = [module.get_outputs()[i].asnumpy() for i in range(3,6)]
#     centers_lst = [module.get_outputs()[i].asnumpy() for i in range(6,9)]
#     centers_mask_lst = [module.get_outputs()[i].asnumpy() for i in range(9,12)]



#         # up_symbol = get_up_symbol(max_p_grid=up_max_p_grid_lst[i], max_o_grid=up_max_o_grid_lst[i], kernel_size=3, coord_shift=[1.0, 1.0, 1.0],
#         #  voxel_size=[0.04, 0.04, 0.04], grid_size=[50, 50, 50])
#         # up_module = mx.mod.Module(up_symbol, context=ctx,
#         #             data_names=['downdata', 'updata', 'downnum', 'upnum'], label_names=None)
#         # up_module.bind(data_shapes=[('downdata', (batch_size, max_o_grid, 4)), ('updata', (batch_size, npoints, 4)),
#         #     ('downnum', (batch_size, 1), "int32"), ('upnum', (batch_size, 1), "int32")], label_shapes=None)
#         # up_module.init_params(initializer=mx.init.Xavier())
#         #
#         # input = mx.io.DataBatch(data=[mx.ndarray.array(centers_arr), mx.ndarray.array(points_w_batch),
#         #         mx.ndarray.array(actual_centnum, dtype="int32"), mx.ndarray.ones((batch_size,1), dtype="int32")*npoints])
#         # up_module.forward(input, is_train=False)
#         # neighbors_up_arr = up_module.get_outputs()[0].asnumpy()
#         # neighbors_up_mask = up_module.get_outputs()[1].asnumpy()

#     fname = "raw" + ".txt"
#     fname = os.path.join(fdir, fname)
#     np.savetxt(fname, points_batch[0], delimiter=";")
#     print("saved:", fname)

#     fname = "norm_all" + ".txt"
#     fname = os.path.join(fdir, fname)
#     np.savetxt(fname, norm_all_data, delimiter=";")
#     print("saved all points at :", fname)

#     # for i in range(13):
#     #     cat_ind = np.where(label == i)
#     #     if cat_ind[0].shape[0] > 0:
#     #         print("cat_ind[0]",cat_ind[0].shape[0])
#     #         fname = "cat{}_{}.txt".format(i,CLASS_NAMES[i])
#     #         fname = os.path.join(fdir, fname)
#     #         cat_pnts = norm_all_data[cat_ind]
#     #         colors = np.tile(CLASS_COLORS[[i],...], (cat_pnts.shape[0],1))
#     #         print("colors.shape", colors.shape)
#     #         cat_pnts_colors = np.concatenate((cat_pnts, colors), axis=1)
#     #         np.savetxt(fname, cat_pnts_colors, delimiter=";")
#     #         print("saved cat{}-{} points at :{}".format(i,CLASS_NAMES[i],fname))

#     for layer in range(len(centers_lst)):
#         centers = centers_lst[layer][0]
#         last_centers = centers_lst[layer-1][0] if layer!=0 else points_batch[0]
#         print(centers.shape)
#         fname = "ly_" + str(layer) + ".txt"
#         fname = os.path.join(fdir, fname)
#         np.savetxt(fname, centers, delimiter=";")
#         print("saved:", fname)

#         neighbors_arr = neighbors_arr_lst[layer][0]
#         centers_mask = centers_mask_lst[layer][0]
#         neighbors_mask = neighbors_mask_lst[layer][0]
#         for cl_ind in range(neighbors_arr.shape[0]):
#             associated_points = []
#             if centers_mask[cl_ind]  == 0:
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

# 