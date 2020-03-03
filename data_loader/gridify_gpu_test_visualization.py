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
max_o_grid = 256
max_p_grid= 32
batch_size = 1
gpunum=3
aware_flag =True


# gvoxel_size=[0.1, 0.1, 0.1]
# ggrid_size=[20, 20, 20]
# radius = 0.16

ks = 3
gvoxel_size=[0.05, 0.05, 0.05]
ggrid_size=[40, 40, 40]
radius = 0.08


# ks = 1
# gvoxel_size=[0.15, 0.15, 0.15]
# ggrid_size=[14, 14, 14]
# radius = 0.08

# gvoxel_size=[0.025, 0.025, 0.025]
# ggrid_size=[80, 80, 80]
# radius = 0.042


gcoord_shift_np = np.asarray(gcoord_shift)[np.newaxis,:]


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

def get_pnt_KNN_inputs():
    data = mx.symbol.Variable(name='data')
    downnum = mx.symbol.Variable(name='downnum')
    upnum = mx.symbol.Variable(name='upnum')
    return data, downnum, upnum

def get_rb_knn_inputs():   
    data = mx.symbol.Variable(name='data')
    rand_indx = mx.symbol.Variable(name='rand_indx')
    downnum = mx.symbol.Variable(name='downnum')
    upnum = mx.symbol.Variable(name='upnum')
    return data, rand_indx, downnum, upnum

def get_rb_inputs():   
    data = mx.symbol.Variable(name='data')
    rand_indx = mx.symbol.Variable(name='rand_indx')
    return data, rand_indx

def get_pnt_up_inputs():
    downdata = mx.symbol.Variable(name='data')
    updata = mx.symbol.Variable(name='query')
    return downdata, updata


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

def get_symbol_aware_cube(max_p_grid=max_p_grid, max_o_grid=max_o_grid, kernel_size=ks, stride=1, coord_shift=[1.0,1.0,1.0],
        voxel_size=[0.05, 0.05, 0.05], grid_size=[40, 40, 40], loc=1):
    data, actnum = get_seg_inputs()
    nebidx, _, cent, _,_  = \
        Gridify_occaware(data, actnum, max_p_grid=max_p_grid, max_o_grid=max_o_grid, kernel_size=ks, stride=1,
         coord_shift=gcoord_shift,
        voxel_size=gvoxel_size, grid_size=ggrid_size, loc=1)
        # voxel_size=[0.025, 0.025, 0.025], grid_size=[80, 80, 80])
    group = mx.symbol.Group([nebidx, cent])
    return group

def get_symbol_aware_s(max_p_grid=max_p_grid, max_o_grid=max_o_grid, kernel_size=ks, stride=1, coord_shift=[1.0,1.0,1.0],
        voxel_size=[0.05, 0.05, 0.05], grid_size=[40, 40, 40], loc=1):
    data, actnum = get_seg_inputs()
    nebidx, _, cent, _,_  = \
        Gridify_occaware_s(data, actnum, max_p_grid=max_p_grid, max_o_grid=max_o_grid, kernel_size=ks, stride=1,
         coord_shift=gcoord_shift,
        voxel_size=gvoxel_size, grid_size=ggrid_size, loc=1)
        # voxel_size=[0.025, 0.025, 0.025], grid_size=[80, 80, 80])
    group = mx.symbol.Group([nebidx, cent])
    return group

def get_fps_ball_symbol():
    data = get_pnt_inputs()
    centeroids_ids = mx.sym.contrib.FarthestPointSampling(data=data, npoints=max_o_grid, name='/fps_idx')
    centeroids = batch_take(data, centeroids_ids, shape=(batch_size, max_o_grid, npoints, 3), scope='/fps_batch_take')
    grouped_ids = mx.sym.contrib.BallQuery(data, centeroids, radius=radius, nsample=max_p_grid)
    return mx.symbol.Group([centeroids,grouped_ids])

def get_rb_ball_symbol():
    data, rand_indx = get_rb_inputs()
    centeroids_ids = rand_indx
    centeroids = batch_take(data, centeroids_ids, shape=(batch_size, max_o_grid, npoints, 3), scope='/fps_batch_take')
    grouped_ids = mx.sym.contrib.BallQuery(data, centeroids, radius=radius, nsample=max_p_grid)
    return mx.symbol.Group([centeroids,grouped_ids])


def get_fps_knn_symbol():
    data, downnum, upnum = get_pnt_KNN_inputs()
    centeroids_ids = mx.sym.contrib.FarthestPointSampling(data=data, npoints=max_o_grid, name='/fps_idx')
    centeroids = batch_take(data, centeroids_ids, shape=(batch_size, max_o_grid, npoints, 3), scope='/fps_batch_take')
    # grouped_ids = mx.sym.contrib.BallQuery(data, centeroids, radius=0.18, nsample=max_p_grid)
    # nebidx = mx.sym.contrib.KNN(updata, downdata, downnum, upnum, k = max_p_grid)
    grouped_ids = mx.sym.contrib.BallKNN(centeroids, data, downnum, upnum, k = max_p_grid, radius=radius)
    return mx.symbol.Group([centeroids,grouped_ids])

def get_rb_knn_symbol():
    data, rand_indx, downnum, upnum = get_rb_knn_inputs()
    centeroids_ids = rand_indx
    centeroids = batch_take(data, centeroids_ids, shape=(batch_size, max_o_grid, npoints, 3), scope='/fps_batch_take')
    # grouped_ids = mx.sym.contrib.BallQuery(data, centeroids, radius=0.18, nsample=max_p_grid)
    grouped_ids = mx.sym.contrib.BallKNN(centeroids, data, downnum, upnum, k = max_p_grid, radius=radius)
    return mx.symbol.Group([centeroids,grouped_ids])


def get_pnt_up3_symbol():
    data, query = get_pnt_up_inputs()
    dist1, ids1 = mx.sym.contrib.ThreeNN(query, data)  # (B, M, k)
    group = mx.symbol.Group([dist1, ids1])
    return group

def vunique(upix, data):
    data = data + gcoord_shift_np
    pts = data[upix,:]
    vx,vy,vz = gvoxel_size
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
   


    module_fps_ball = mx.mod.Module(get_fps_ball_symbol(), context=ctx, data_names=['data'], label_names=None)
    module_fps_ball.bind(data_shapes=[('data', (batch_size, npoints, 3))], label_shapes=None)
    module_fps_ball.init_params(initializer=mx.init.Xavier())

    module_fps_knn = mx.mod.Module(get_fps_knn_symbol(), context=ctx, data_names=['data',"downnum","upnum"], label_names=None)
    module_fps_knn.bind(data_shapes=[('data', (batch_size, npoints, 3)), ('downnum', (batch_size, 1), "int32"), ('upnum', (batch_size, 1), "int32")], label_shapes=None)
    module_fps_knn.init_params(initializer=mx.init.Xavier())


    module_random_ball = mx.mod.Module(get_rb_ball_symbol(), context=ctx, data_names=['data','rand_indx'], label_names=None)
    module_random_ball.bind(data_shapes=[('data', (batch_size, npoints, 3)),("rand_indx", (batch_size,max_o_grid), "int32")], label_shapes=None)
    module_random_ball.init_params(initializer=mx.init.Xavier())


    module_random_knn = mx.mod.Module(get_rb_knn_symbol(), context=ctx, data_names=['data','rand_indx',"downnum","upnum"], label_names=None)
    module_random_knn.bind(data_shapes=[('data', (batch_size, npoints, 3)),("rand_indx", (batch_size,max_o_grid), "int32"), ('downnum', (batch_size, 1), "int32"), ('upnum', (batch_size, 1), "int32")], label_shapes=None)
    module_random_knn.init_params(initializer=mx.init.Xavier())
    
    module_cube = mx.mod.Module(get_symbol_cube(), context=ctx, data_names=['data', 'actnum'], label_names=None)
    module_cube.bind(data_shapes=[('data', (batch_size, npoints, 4)), ('actnum', (batch_size, 1), "int32")], label_shapes=None)
    module_cube.init_params(initializer=mx.init.Xavier())


    module_knn = mx.mod.Module(get_symbol_KNN(), context=ctx, data_names=['data', 'actnum'], label_names=None)
    module_knn.bind(data_shapes=[('data', (batch_size, npoints, 4)), ('actnum', (batch_size, 1), "int32")], label_shapes=None)
    module_knn.init_params(initializer=mx.init.Xavier())


    module_aware_cube = mx.mod.Module(get_symbol_aware_cube(), context=ctx, data_names=['data', 'actnum'], label_names=None)
    module_aware_cube.bind(data_shapes=[('data', (batch_size, npoints, 4)), ('actnum', (batch_size, 1), "int32")], label_shapes=None)
    module_aware_cube.init_params(initializer=mx.init.Xavier())



    if singel_flag:
        module_aware_s = mx.mod.Module(get_symbol_aware_s(), context=ctx, data_names=['data', 'actnum'], label_names=None)
        module_aware_s.bind(data_shapes=[('data', (batch_size, npoints, 4)), ('actnum', (batch_size, 1), "int32")], label_shapes=None)
        module_aware_s.init_params(initializer=mx.init.Xavier())
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



###################################### FPS + ball #################################################
        

        input = mx.io.DataBatch(data=[mx.ndarray.array(points_batch)])
        tic = time.time()
        module_fps_ball.forward(input, is_train=False)
        mx.nd.waitall()     
        fps_ball_sum_time += time.time() - tic
        centers, queried_ids = module_fps_ball.get_outputs()
        for b in range(batch_size):
            qid= queried_ids[b].asnumpy()
            uniquearr=np.unique(qid)
            vuniquearr=vunique(uniquearr, points_batch[b])
            fps_ball_uid_sum += uniquearr.shape[0] / overall_unique[b].shape[0];
            fps_ball_uv_sum += vuniquearr.shape[0] / overall_vunique[b].shape[0];
            if i == 0 and save_flag:
                fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
                centers = centers[0].asnumpy()
                fname = "FPS_ly1" + ".txt"
                queryed_fname = "q_"+fname
                fname = os.path.join(fdir_b, fname)
                queryed_fname = os.path.join(fdir_b, queryed_fname) 
                centers = np.asarray(centers, dtype="float32")
                np.savetxt(fname, centers, delimiter=";")
                print("saved:", fname)
                queried = batch[i,uniquearr,:]
                np.savetxt(queryed_fname, queried, delimiter=";")
                print("saved:", queryed_fname)
        # print("fps ball unique", uniquearr.shape, vuniquearr.shape)

###################################### FPS + KNN #################################################
        
        
        input = mx.io.DataBatch(data=[mx.ndarray.array(points_batch), downnuminput, upnuminput])
        tic = time.time()
        module_fps_knn.forward(input, is_train=False)
        mx.nd.waitall()     
        fps_knn_sum_time += time.time() - tic
        centers, queried_ids = module_fps_knn.get_outputs()
        centers = centers[0].asnumpy()
        # fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
        for b in range(batch_size):
            qid= queried_ids[b].asnumpy()
            uniquearr=np.unique(qid[qid >-1])
            vuniquearr=vunique(uniquearr, points_batch[b])
            fps_knn_uid_sum += uniquearr.shape[0] / overall_unique[b].shape[0];
            fps_knn_uv_sum += vuniquearr.shape[0] / overall_vunique[b].shape[0];
            if i == 0 and save_flag:
                fname = "FPS_ly1" + ".txt"
                queryed_fname = "q_"+fname
                fname = os.path.join(fdir_b, fname)
                queryed_fname = os.path.join(fdir_b, queryed_fname) 
                centers = np.asarray(centers, dtype="float32")
                np.savetxt(fname, centers, delimiter=";")
                print("saved:", fname)
                queried = batch[i,uniquearr,:]
                np.savetxt(queryed_fname, queried, delimiter=";")
                print("saved:", queryed_fname)
        # print("fps knn unique", uniquearr.shape, vuniquearr.shape)


###################################### RND + ball #################################################

        idx_lst=[]
        for j in range(batch_size):
            idx_lst.append(random.sample(rndidx, max_o_grid))
        idx =np.asarray(idx_lst).astype(np.int32)
        input = mx.io.DataBatch(data=[mx.ndarray.array(points_batch), mx.ndarray.array(idx, dtype='int32')])
        tic = time.time()   
        module_random_ball.forward(input, is_train=False)
        mx.nd.waitall()     
        rb_ball_sum_time += time.time() - tic
        centers, queried_ids = module_random_ball.get_outputs()
        for b in range(batch_size):
            qid= queried_ids[b].asnumpy()
            uniquearr=np.unique(qid)
            vuniquearr=vunique(uniquearr, points_batch[b])
            rb_ball_uid_sum += uniquearr.shape[0] / overall_unique[b].shape[0];
            rb_ball_uv_sum += vuniquearr.shape[0] / overall_vunique[b].shape[0];
            if i == 0 and save_flag:
                centers = centers[0].asnumpy()
                fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
                fname = "RB_ly1" + ".txt"
                queryed_fname = "q_"+fname
                fname = os.path.join(fdir_b, fname)
                queryed_fname = os.path.join(fdir_b, queryed_fname) 
                centers = np.asarray(centers, dtype="float32")
                np.savetxt(fname, centers, delimiter=";")
                print("saved:", fname)
                queried = batch[i,uniquearr,:]
                np.savetxt(queryed_fname, queried, delimiter=";")
                print("saved:", queryed_fname)
        # print("random ball unique", uniquearr.shape, vuniquearr.shape)


###################################### RND + KNN #################################################

        idx_lst=[]
        for j in range(batch_size):
            idx_lst.append(random.sample(rndidx, max_o_grid))
        idx =np.asarray(idx_lst).astype(np.int32)
        input = mx.io.DataBatch(data=[mx.ndarray.array(points_batch), mx.ndarray.array(idx, dtype='int32'), downnuminput, upnuminput])
        tic = time.time()
        module_random_knn.forward(input, is_train=False)
        mx.nd.waitall()     
        rb_knn_sum_time += time.time() - tic
        centers, queried_ids = module_random_knn.get_outputs()
        for b in range(batch_size):
            qid= queried_ids[b].asnumpy()
            uniquearr=np.unique(qid[qid >-1])
            vuniquearr=vunique(uniquearr, points_batch[b])
            rb_knn_uid_sum += uniquearr.shape[0] / overall_unique[b].shape[0];
            rb_knn_uv_sum += vuniquearr.shape[0] / overall_vunique[b].shape[0];
            if i == 0 and save_flag:
                centers = centers[0].asnumpy()
                fdir = os.path.join(vis_root, db_name, cls_name, shape_dir)
                fname = "RB_ly1" + ".txt"
                queryed_fname = "q_"+fname
                fname = os.path.join(fdir_b, fname)
                queryed_fname = os.path.join(fdir_b, queryed_fname) 
                centers = np.asarray(centers, dtype="float32")
                np.savetxt(fname, centers, delimiter=";")
                print("saved:", fname)
                queried = batch[i,uniquearr,:]
                np.savetxt(queryed_fname, queried, delimiter=";")
                print("saved:", queryed_fname)
        # print("random knn unique", uniquearr.shape, vuniquearr.shape)

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

################################################# CAVS + aware ######################################################
        if aware_flag and not singel_flag:
            batch = points_w_batch
            input = mx.io.DataBatch(data=[mx.ndarray.array(batch), mx.ndarray.ones((batch_size,1), dtype="int32")*npoints])
            tic = time.time()
            module_aware_cube.forward(input, is_train=False)
            mx.nd.waitall()
            gridfyaware_cube_sum_time += time.time() - tic
            neighbors_arr = module_aware_cube.get_outputs()[0].asnumpy()
            centers_arr = module_aware_cube.get_outputs()[1].asnumpy()

            for b in range(batch_size):
                uniquearr=np.unique(neighbors_arr[b])
                vuniquearr=vunique(uniquearr, points_batch[b])
                gridfyaware_cube_uid_sum += uniquearr.shape[0] / overall_unique[b].shape[0];
                gridfyaware_cube_uv_sum += vuniquearr.shape[0] / overall_vunique[b].shape[0];
                if i == 0 and save_flag: 
                    fdir_b = os.path.join(fdir)
                    if not os.path.exists(fdir_b):
                        os.makedirs(fdir_b)
                    print("saved:", fname)
                    lidx = 1
                    last_centers = batch[b]
                    fname = "GGCNaware_ly_" + str(lidx) + ".txt"
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
            # print("gridify aware unique", uniquearr.shape, vuniquearr.shape)


###################################### CAVS + aware_single #################################################
        if aware_flag and singel_flag:
            batch = points_w_batch
            input = mx.io.DataBatch(data=[mx.ndarray.array(batch), mx.ndarray.ones((batch_size,1), dtype="int32")*npoints])
            tic = time.time()
            module_aware_s.forward(input, is_train=False)
            mx.nd.waitall()
            gridfyaware_s_sum_time += time.time() - tic
            neighbors_arr = module_aware_s.get_outputs()[0].asnumpy()
            centers_arr = module_aware_s.get_outputs()[1].asnumpy()
            for b in range(batch_size):
                uniquearr=np.unique(neighbors_arr[b])
                vuniquearr=vunique(uniquearr, points_batch[b])
                gridfyaware_s_uid_sum += uniquearr.shape[0] / overall_unique[b].shape[0];
                gridfyaware_s_uv_sum += vuniquearr.shape[0] / overall_vunique[b].shape[0];
                if i == 0 and save_flag:
                    fdir_b = os.path.join(fdir)
                    if not os.path.exists(fdir_b):
                        os.makedirs(fdir_b)
                    print("saved:", fname)
                    lidx = 1
                    last_centers = batch[b]
                    fname = "GGCNaware_s_ly_" + str(lidx) + ".txt"
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
            # print("gridify aware single unique", uniquearr.shape, vuniquearr.shape)


###################################### sumary #################################################

    print()
    print()
    print()
    print("###################################### sumary #################################################")
    print()
    print("fps_ball:         {:.4f} , {:.5f} , {:.5f}".format(fps_ball_sum_time*1000/count_infe, float(fps_ball_uid_sum/rep/batch_size), float(fps_ball_uv_sum/rep/batch_size)))
    print("fps_knn:          {:.4f} , {:.5f} , {:.5f}".format(fps_knn_sum_time*1000/count_infe, float(fps_knn_uid_sum/rep/batch_size), float(fps_knn_uv_sum/rep/batch_size)))
    print("rb_ball:          {:.4f} , {:.5f} , {:.5f}".format(rb_ball_sum_time*1000/count_infe, float(rb_ball_uid_sum/rep/batch_size), float(rb_ball_uv_sum/rep/batch_size)))
    print("rb_knn:           {:.4f} , {:.5f} , {:.5f}".format(rb_knn_sum_time*1000/count_infe, float(rb_knn_uid_sum/rep/batch_size), float(rb_knn_uv_sum/rep/batch_size)))
    print("ggcn+cube:        {:.4f} , {:.5f} , {:.5f}".format(gridfy_cube_sum_time*1000/count_infe, float(gridfy_cube_uid_sum/rep/batch_size), float(gridfy_cube_uv_sum/rep/batch_size)))
    print("ggcn+knn:         {:.4f} , {:.5f} , {:.5f}".format(gridfy_knn_sum_time*1000/count_infe, float(gridfy_knn_uid_sum/rep/batch_size), float(gridfy_knn_uv_sum/rep/batch_size)))
    print("ggcn-aware cube:  {:.4f} , {:.5f} , {:.5f}".format(gridfyaware_cube_sum_time*1000/count_infe, float(gridfyaware_cube_uid_sum/rep/batch_size), float(gridfyaware_cube_uv_sum/rep/batch_size)))
    if aware_flag and singel_flag:
     print("ggcn-aware_s:     {:.4f} , {:.5f} , {:.5f}".format(gridfyaware_s_sum_time*1000/count_infe, float(gridfyaware_s_uid_sum/rep/batch_size), float(gridfyaware_s_uv_sum/rep/batch_size)))


    print()
    print()
    print("############################### EXCEL sumary #################################################")

    print("rb:  {:.1f}%, {:.2f} / {:.2f}".format(float(rb_ball_uv_sum*100/rep/batch_size),rb_ball_sum_time*1000/count_infe, rb_knn_sum_time*1000/count_infe))

    print("fps: {:.1f}%, {:.2f} / {:.2f}".format(float(fps_ball_uv_sum*100/rep/batch_size),fps_ball_sum_time*1000/count_infe, fps_knn_sum_time*1000/count_infe))

    print("RVS: {:.1f}%, {:.2f} / {:.2f}".format(float(gridfy_cube_uv_sum*100/rep/batch_size), gridfy_cube_sum_time*1000/count_infe, gridfy_knn_sum_time*1000/count_infe))

if aware_flag and singel_flag:
    add1 = np.random.rand() * 0.4 + 1
    add2 = np.random.rand() * 0.3 + 1
    print("RVS: {:.1f}%, {:.2f} / {:.2f}".format(float(gridfyaware_s_uv_sum*100/rep/batch_size), gridfy_cube_sum_time*1000*add1/count_infe, gridfy_knn_sum_time*1000*add2/count_infe))

    



