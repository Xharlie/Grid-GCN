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
# from mxnet.symbol import Gridify
# from mxnet.symbol import GridifyUp
from mxnet.symbol import Gridify
from mxnet.symbol import Gridify_occaware 
from mxnet.symbol import Gridify_occaware_s
from mxnet.symbol import GridifyKNN

from utils.ops import batch_take
from gridify import assemble_tensor
import h5py 

from data_loader.ggcn_gpu_modelnet_loader import ModelNet40Loader as loader


os.environ["MXNET_EXEC_BULK_EXEC_INFERENCE"] = "0"
profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    filename='gpu_profile_output_{}.json'.format(time.time()))

# max, min [[0.9212442 0.4067771 0.8582453 1.       ]] [[-0.9661901  -0.17144969 -0.8550301   1.        ]]
gcoord_shift=[1.0, 1.0 ,1.0]
singel_flag = True 
npoints=8192 
max_o_grid = 1024
max_p_grid= 32
batch_size = 1
gpunum=0
aware_flag =True


# gvoxel_size=[0.1, 0.1, 0.1]
# ggrid_size=[20, 20, 20]
# radius = 0.16

# ks = 3
# gvoxel_size=[0.05, 0.05, 0.05]
# ggrid_size=[40, 40, 40]
# radius = 0.08


ks = 1
vgvoxel_size = [0.05, 0.05, 0.05]
gvoxel_size=[0.15, 0.15, 0.15]
ggrid_size=[14, 14, 14]
radius = 0.08



# ks = 1
# gvoxel_size=[0.075, 0.075, 0.075]
# ggrid_size=[27, 27, 27]
# radius = 0.06


# ks = 3
# gvoxel_size=[0.025, 0.025, 0.025]
# ggrid_size=[80, 80, 80]
# radius = 0.042


gcoord_shift_np = np.asarray(gcoord_shift)[np.newaxis,:]

def get_seg_inputs():
    data = mx.symbol.Variable(name='data')
    actnum = mx.symbol.Variable(name='actnum')
    return data, actnum

def get_symbol_cube(max_p_grid=max_p_grid, max_o_grid=max_o_grid, kernel_size=ks, stride=1, coord_shift=[1.0,1.0,1.0],
        voxel_size=[0.05, 0.05, 0.05], grid_size=[40, 40, 40], loc=1):
    data, actnum = get_seg_inputs()
    nebidx, _, cent, _,_ = \
        Gridify(data, actnum, max_p_grid=max_p_grid, max_o_grid=max_o_grid, kernel_size=ks, stride=1,
        coord_shift=gcoord_shift,
        voxel_size=gvoxel_size, grid_size=ggrid_size, loc=1)
        # voxel_size=[0.025, 0.025, 0.025], grid_size=[80, 80, 80])
    group = mx.symbol.Group([nebidx, cent])
    return group

def get_symbol_KNN(max_p_grid=max_p_grid, max_o_grid=max_o_grid, kernel_size=ks, stride=1, coord_shift=[1.0,1.0,1.0],
        voxel_size=[0.05, 0.05, 0.05], grid_size=[40, 40, 40], loc=1):
    data, actnum = get_seg_inputs()
    nebidx, _, cent, _,_ = \
        GridifyKNN(data, actnum, max_p_grid=max_p_grid, max_o_grid=max_o_grid, kernel_size=ks, stride=1,
        coord_shift=gcoord_shift,
        voxel_size=gvoxel_size, grid_size=ggrid_size, loc=1)
        # voxel_size=[0.025, 0.025, 0.025], grid_size=[80, 80, 80])
    group = mx.symbol.Group([nebidx, cent])
    return group

def vunique(upix, data):
    data = data + gcoord_shift_np
    pts = data[upix,:]
    vx,vy,vz = vgvoxel_size
    vid = pts[:,0] // vx * ggrid_size[1]*ggrid_size[2] + pts[:,1] // vy  * ggrid_size[2] + pts[:,2] // vz
    return np.unique(vid)


if __name__ == "__main__":

    rndidx = list(range(npoints))
    downnuminput = mx.ndarray.ones((batch_size, 1), dtype="int32") * npoints
    upnuminput = mx.ndarray.ones((batch_size, 1), dtype="int32") * max_o_grid


    vis_root = "single_vis"


    # print("max, min",np.max(points_w_batch,axis=1),np.min(points_w_batch,axis=1))


    # shape_dir = shape_txt_filename.split("/")[-1].split("_")[1].split(".")[0]
    # fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
    # if not os.path.exists(fdir):
    #     os.makedirs(fdir)
    # print("write to fdir: ",fdir)
    ctx = [mx.gpu(gpunum)]
   
    
    module_cube = mx.mod.Module(get_symbol_cube(), context=ctx, data_names=['data', 'actnum'], label_names=None)
    module_cube.bind(data_shapes=[('data', (batch_size, npoints, 4)), ('actnum', (batch_size, 1), "int32")], label_shapes=None)
    module_cube.init_params(initializer=mx.init.Xavier())


    module_knn = mx.mod.Module(get_symbol_KNN(), context=ctx, data_names=['data', 'actnum'], label_names=None)
    module_knn.bind(data_shapes=[('data', (batch_size, npoints, 4)), ('actnum', (batch_size, 1), "int32")], label_shapes=None)
    module_knn.init_params(initializer=mx.init.Xavier())

    val_loader = loader(
            root="../data/modelnet40_normal_resampled",
            configs={'num_channel': 3},
            batch_size=1,
            npoints=npoints,
            normal_channel=False,
            split='test',
            normalize=True,
            shuffle=True,
            balance=False,
            dropout_ratio=0,
            point_set="partial",
            dataset = "40"
    )

    count_infe = 0;

    fps_ball_sum_time = 0;
    fps_knn_sum_time = 0
    rb_ball_sum_time = 0;
    rb_knn_sum_time = 0;
    gridfy_cube_sum_time = 0
    gridfy_knn_sum_time = 0
    gridfyaware_cube_sum_time = 0 
    gridfyaware_s_sum_time = 0



    fps_ball_uid_sum = 0;
    rb_ball_uid_sum = 0;
    fps_knn_uid_sum = 0;
    rb_knn_uid_sum = 0;
    gridfy_cube_uid_sum = 0
    gridfy_knn_uid_sum = 0
    gridfyaware_cube_uid_sum = 0 
    gridfyaware_s_uid_sum = 0


    fps_ball_uv_sum = 0;
    rb_ball_uv_sum = 0;
    fps_knn_uv_sum = 0;
    rb_knn_uv_sum = 0;
    gridfy_cube_uv_sum = 0
    gridfy_knn_uv_sum = 0
    gridfyaware_cube_uv_sum = 0 
    gridfyaware_s_uv_sum = 0


    rep=val_loader.num_samples // batch_size; 
    save_flag=False
    realindex = np.arange(val_loader.num_samples)
    np.random.shuffle(realindex)
    for i in range(rep):
        print("batch:{}/{}".format(i,rep))
        points_batch_lst = []
        count_infe += 1
        overall_unique = []
        overall_vunique = []
        for b in range(batch_size):
            index = i * batch_size + b
            db_name = "modelnet40"
            cls_name, shape_txt_filename = val_loader.datapath[realindex[index]]
            cls = val_loader.classes[cls_name]
            cls = np.array([cls]).astype(np.int32)
            # points = np.loadtxt(shape_txt_filename, delimiter=',').astype(np.float32)
            
            # single_path = '../data/modelnet40_normal_resampled/airplane/airplane_0059.txt'
            # db_name, cls_name, shape_txt_filename = "modelnet40", "airplane", single_path
            point_set = np.loadtxt(shape_txt_filename, delimiter=',')[:min(8192,npoints), ...]

            if npoints > 8192:
                indx = np.arange(npoints) % 8192
                point_set = point_set[indx, ...]
            point_set = utils.normalize_point_cloud(point_set[:, 0:3])
            np.random.shuffle(point_set)     
            points_batch_lst.append(point_set)
            overall_unique.append(np.ones(min(8192,npoints)))
            overall_vunique.append(vunique(np.arange(npoints), point_set))
            # print(np.ones(min(8192,npoints)).shape, vunique(np.arange(npoints), point_set).shape)
        points_batch = np.asarray(points_batch_lst).astype(np.float32)
        points_w_batch = np.concatenate([points_batch, np.ones([batch_size, npoints, 1], dtype=np.float32)], axis = 2)


################################################# CAVS + RVQ ######################################################
        batch = points_w_batch
        input = mx.io.DataBatch(data=[mx.ndarray.array(batch), mx.ndarray.ones((batch_size,1), dtype="int32")*npoints])
        tic = time.time()
        module_cube.forward(input, is_train=False)
        mx.nd.waitall()
        gridfy_cube_sum_time += time.time() - tic
        neighbors_arr = module_cube.get_outputs()[0].asnumpy()
        centers_arr = module_cube.get_outputs()[1].asnumpy()
        for b in range(batch_size):
            uniquearr=np.unique(neighbors_arr[b])
            vuniquearr=vunique(uniquearr, points_batch[b])
            gridfy_cube_uid_sum += uniquearr.shape[0] / overall_unique[b].shape[0];
            gridfy_cube_uv_sum += vuniquearr.shape[0] / overall_vunique[b].shape[0];
            if i == 0 and save_flag: 
                fdir_b = os.path.join(fdir)
                if not os.path.exists(fdir_b):
                    os.makedirs(fdir_b)
                fname = "GGCN_raw" + ".txt"
                fname = os.path.join(fdir_b, fname)
                np.savetxt(fname, batch[b], delimiter=";")
                print("saved:", fname)
                lidx = 1
                last_centers = batch[b]
                fname = "GGCN_ly_" + str(lidx) + ".txt"
                queryed_fname = "q_"+fname
                fname = os.path.join(fdir_b, fname)
                queryed_fname = os.path.join(fdir_b, queryed_fname) 
                centers = []
                for j in range(centers_arr.shape[1]):
                    if centers_mask[b,j] < 1:
                        break
                    centers.append(centers_arr[b,j,...])
                centers = np.asarray(centers, dtype="float32")

                np.savetxt(fname, centers, delimiter=";")
                print("saved:", fname)
                queried = batch[i,uniquearr,:]
                np.savetxt(queryed_fname, queried, delimiter=";")
                print("saved:", queryed_fname)
        # print("gridify cube unique", uniquearr.shape, vuniquearr.shape)

################################################# CAVS + KNN ######################################################
        batch = points_w_batch
        input = mx.io.DataBatch(data=[mx.ndarray.array(batch), mx.ndarray.ones((batch_size,1), dtype="int32")*npoints])
        tic = time.time()
        module_knn.forward(input, is_train=False)
        mx.nd.waitall()
        gridfy_knn_sum_time += time.time() - tic
        neighbors_arr = module_knn.get_outputs()[0].asnumpy()
        centers_arr = module_knn.get_outputs()[1].asnumpy()
        for b in range(batch_size):
            uniquearr=np.unique(neighbors_arr[b])
            vuniquearr=vunique(uniquearr, points_batch[b])
            gridfy_knn_uid_sum += uniquearr.shape[0] / overall_unique[b].shape[0];
            gridfy_knn_uv_sum += vuniquearr.shape[0] / overall_vunique[b].shape[0];
            if i == 0 and save_flag: 
                fdir_b = os.path.join(fdir)
                if not os.path.exists(fdir_b):
                    os.makedirs(fdir_b)
                fname = "GGCN_raw" + ".txt"
                fname = os.path.join(fdir_b, fname)
                np.savetxt(fname, batch[b], delimiter=";")
                print("saved:", fname)
                lidx = 1
                last_centers = batch[b]
                fname = "GGCN_ly_" + str(lidx) + ".txt"
                queryed_fname = "q_"+fname
                fname = os.path.join(fdir_b, fname)
                queryed_fname = os.path.join(fdir_b, queryed_fname) 
                centers = []
                for j in range(centers_arr.shape[1]):
                    if centers_mask[b,j] < 1:
                        break
                    centers.append(centers_arr[b,j,...])
                centers = np.asarray(centers, dtype="float32")

                np.savetxt(fname, centers, delimiter=";")
                print("saved:", fname)
                queried = batch[i,uniquearr,:]
                np.savetxt(queryed_fname, queried, delimiter=";")
                print("saved:", queryed_fname)
        # print("gridify knn unique", uniquearr.shape, vuniquearr.shape)



###################################### sumary #################################################

    print()
    print()
    print()
    print("###################################### sumary #################################################")
    print()
   
    print("ggcn+cube:        {:.4f} , {:.5f} , {:.5f}".format(gridfy_cube_sum_time*1000/count_infe, float(gridfy_cube_uid_sum/rep/batch_size), float(gridfy_cube_uv_sum/rep/batch_size)))
    print("ggcn+knn:         {:.4f} , {:.5f} , {:.5f}".format(gridfy_knn_sum_time*1000/count_infe, float(gridfy_knn_uid_sum/rep/batch_size), float(gridfy_knn_uv_sum/rep/batch_size)))
   





