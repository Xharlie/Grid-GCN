import sys
if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle
import mxnet as mx
import h5py
import numpy as np
import threading
import queue
import os
import random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from transforms3d.euler import euler2mat
import math
from segmentation.s3dis_configs.configs import configs
from utils import utils, data_utils
filelist = '../data/3DIS/prepare_label_rgb/train_files_for_val_on_Area_1.txt'
filelist_val = '../data/3DIS/prepare_label_rgb/val_files_Area_1.txt'
filelist_val = os.path.join(os.path.dirname(os.path.abspath(__file__)), filelist_val)
rot90 = np.array([0.0, math.pi/2, math.pi, math.pi*3/2, math.pi*2])

class S3DISLoader(mx.io.DataIter, threading.Thread):
    def __init__(self, filelist, batch_size=32, npoints=8192, split='train',shuffle=False, normalize="ball", augment_data=-1, use_weights=False, use_cache=False, cache_file='', qsize=64, max_epoch=200, drop_factor=0.0,
        area=5):
        mx.io.DataIter.__init__(self)
        threading.Thread.__init__(self)
        print(filelist)
        filelist = os.path.join(os.path.dirname(os.path.abspath(__file__)), filelist)
        self.split = split
        self.queue = queue.Queue(qsize)
        self.stopped = False
        self.h5group_num = 0
        self.max_epoch=max_epoch
        self.bno = 0
        self.data_val = None
        self._augment_batch_data = [self._augment_batch_data_l1
        , self._augment_batch_data_l1, self._augment_batch_data_l2, self._augment_batch_data_l3, self._augment_batch_data_l4, self._augment_batch_data_l5,self._augment_batch_data_l6, self._augment_batch_data_l7, self._augment_batch_data_l8, self._augment_batch_data_l9, self._augment_batch_data_l10, self._augment_batch_data_l11, self._augment_batch_data_l12, self._augment_batch_data_l13]

        is_list_of_h5_list = data_utils.is_h5_list(filelist)
        if is_list_of_h5_list:
            self.seg_list = [filelist]  # for test
        else:
            self.seg_list = data_utils.load_seg_list(filelist)  # for train
        self.batch_size = batch_size
        self.drop_factor = drop_factor
        self.npoints = npoints
        self.max_h5_group_num =max_epoch*len(self.seg_list)
        # self.label_weights_list = setting.label_weights

        self.shuffle = shuffle
        self.normalize = normalize
        self.augment_data = augment_data
        self.use_weights = use_weights
        # if self.use_weights:
        #     self.get_weight_gradient_multiplier()

        self.dataxyz_shape = (self.batch_size, self.npoints, 3)
        self.datafeat_shape = (self.batch_size, self.npoints, configs["indim"])
        self.label_shape = (self.batch_size, self.npoints)

        self.cache_file = cache_file
        self.cache = []
        self.batch_counter = 0
        amount_lst = [67331, 66383, 74767, 67532, 57200, 68137]
        self.num_batches = amount_lst[area-1] // self.batch_size

    # self.reset()

    # def reset(self):
    #     self.ids = np.arange(self.num_samples)
    #     if self.shuffle:
    #         np.random.shuffle(self.ids)
    #     self.num_batches = self.num_samples // self.batch_size
    #     self.batch_idx = 0
    #     self.next_cache_index = 0

    @property
    def provide_data(self):
        return [('dataxyz', self.dataxyz_shape), ('datafeat', self.datafeat_shape), ('actual_centnum', (self.batch_size, 1), "int32")]

    @property
    def provide_label(self):
        return [('label', self.label_shape)]

    def _dump_cache(self):
        with open(os.path.join(configs['data_dir'], self.cache_file), 'wb') as f:
            pickle.dump(self.cache, f, protocol=-1)

    def _load_cache(self):
        cache_file = os.path.join(configs['data_dir'], self.cache_file)
        if not os.path.isfile(cache_file):
            print('cache file not found. Will generate new data')
            return False
        print('reading cache file from', self.cache_file)
        with open(cache_file, 'rb') as f:
            self.cache = pickle.load(f, encoding="bytes")
        return True

    def _get_data_from_cache(self):
        if self.next_cache_index >= len(self.cache):
            return None
        data, label = self.cache[self.next_cache_index]
        self.next_cache_index += 1
        return data, label

    def _augment_batch_data_l1(self, batch_data, batch_size, rotation_range=(0, math.pi/72., math.pi/72., 'u'), scaling_range=(0.001, 0.001, 0.001, 'g'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            rx = utils.rotation_angle(rotation_range[0], rotation_range[3])
            ry = utils.rotation_angle(rotation_range[1], rotation_range[3])
            rz = utils.rotation_angle(rotation_range[2], rotation_range[3])
            rotation = euler2mat(rx, ry, rz, order)

            sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            scaling = np.diag([sx, sy, sz])

            xforms[i, :] = np.matmul(scaling, rotation).T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l2(self, batch_data, batch_size, rotation_range=(0, math.pi*2, 0, 'u'), scaling_range=(0.001, 0.001, 0.001, 'g'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            rx = 0.0
            ry = utils.rotation_angle(rotation_range[1], rotation_range[3])
            rz = 0.0
            rotation = euler2mat(rx, ry, rz, order)
            #
            # sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            # sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            # sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            # scaling = np.diag([sx, sy, sz])

            xforms[i, :] = rotation.T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l3(self, batch_data, batch_size, rotation_range=(math.pi/72., math.pi*2, math.pi/72., 'u'), scaling_range=(0.001, 0.001, 0.001, 'g'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            rx = utils.rotation_angle(rotation_range[0], rotation_range[3])
            ry = utils.rotation_angle(rotation_range[1], rotation_range[3])
            rz = utils.rotation_angle(rotation_range[2], rotation_range[3])
            rotation = euler2mat(rx, ry, rz, order)
            #
            # sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            # sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            # sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            # scaling = np.diag([sx, sy, sz])

            xforms[i, :] = rotation.T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l4(self, batch_data, batch_size, rotation_range=(0., 0., 0., 'u'), scaling_range=(0.001, 0.001, 0.001, 'ig'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            # rx = utils.rotation_angle(rotation_range[0], rotation_range[3])
            # ry = utils.rotation_angle(rotation_range[1], rotation_range[3])
            # rz = utils.rotation_angle(rotation_range[2], rotation_range[3])
            # rotation = euler2mat(rx, ry, rz, order)
            #
            sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            scaling = np.diag([sx, sy, sz])

            xforms[i, :] = scaling.T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l5(self, batch_data, batch_size, rotation_range=(0., 0., 0., 'u'), scaling_range=(0.01, 0.01, 0.01, 'ig'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            # rx = utils.rotation_angle(rotation_range[0], rotation_range[3])
            # ry = utils.rotation_angle(rotation_range[1], rotation_range[3])
            # rz = utils.rotation_angle(rotation_range[2], rotation_range[3])
            # rotation = euler2mat(rx, ry, rz, order)
            #
            sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            scaling = np.diag([sx, sy, sz])

            xforms[i, :] = scaling.T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l6(self, batch_data, batch_size, rotation_range=(0., math.pi*2, 0., 'u'), scaling_range=(0.001, 0.001, 0.001, 'ig'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            rx = 0.0
            ry = np.random.choice(rot90, 1)
            rz = 0.0
            rotation = euler2mat(rx, ry, rz, order)
            #
            # sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            # sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            # sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            # scaling = np.diag([sx, sy, sz])

            xforms[i, :] = rotation.T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l7(self, batch_data, batch_size, rotation_range=(math.pi/72, math.pi/72, math.pi/72, 'u'), scaling_range=(0.001, 0.001, 0.001, 'ig'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            rx = utils.rotation_angle(rotation_range[0], rotation_range[3])
            ry = utils.rotation_angle(rotation_range[1], rotation_range[3]) + np.random.choice(rot90, 1)
            rz = utils.rotation_angle(rotation_range[2], rotation_range[3])

            rotation = euler2mat(rx, ry, rz, order)
            #
            # sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            # sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            # sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            # scaling = np.diag([sx, sy, sz])

            xforms[i, :] = rotation.T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l8(self, batch_data, batch_size, rotation_range=(0., 0., 0., 'u'), scaling_range=(0.005, 0.005, 0.005, 'g'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            # rx = utils.rotation_angle(rotation_range[0], rotation_range[3])
            # ry = utils.rotation_angle(rotation_range[1], rotation_range[3])
            # rz = utils.rotation_angle(rotation_range[2], rotation_range[3])
            # rotation = euler2mat(rx, ry, rz, order)
            #
            sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            scaling = np.diag([sx, sy, sz])

            xforms[i, :] = scaling.T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data


    def _augment_batch_data_l9(self, batch_data, batch_size, rotation_range=(0., 0., 0., 'u'), scaling_range=(0.001, 0.001, 0.001, 'g'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            rx = 0.0
            ry = np.random.choice(rot90, 1)
            rz = 0.0
            rotation = euler2mat(rx, ry, rz, order)
            #
            sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            scaling = np.diag([sx, sy, sz])
            xforms[i, :] = np.matmul(scaling, rotation).T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l10(self, batch_data, batch_size, rotation_range=(math.pi/72., 0, math.pi/72., 'u'), scaling_range=(0.001, 0.001, 0.001, 'g'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            rx = 0.0
            ry = np.random.choice(rot90, 1)
            rz = 0.0
            rotation1 = euler2mat(rx, ry, rz, order)
            
            rx = utils.rotation_angle(rotation_range[0], rotation_range[3])
            ry = utils.rotation_angle(rotation_range[1], rotation_range[3])
            rz = utils.rotation_angle(rotation_range[2], rotation_range[3])
            rotation2 = euler2mat(rx, ry, rz, order)

            sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            scaling = np.diag([sx, sy, sz])

            xforms[i, :] = np.matmul(scaling, np.matmul(rotation2, rotation1)).T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l11(self, batch_data, batch_size, rotation_range=(0, 0, 0, 'u'), scaling_range=(0.005, 0.005, 0.005, 'g'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            rx = 0.0
            ry = np.random.choice(rot90, 1)
            rz = 0.0
            rotation = euler2mat(rx, ry, rz, order)

            sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            scaling = np.diag([sx, sy, sz])

            xforms[i, :] =  np.matmul(scaling,rotation).T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l12(self, batch_data, batch_size, rotation_range=(0, math.pi/32., 0, 'u'), scaling_range=(0.001, 0.001, 0.001, 'g'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            rx = utils.rotation_angle(rotation_range[0], rotation_range[3])
            ry = utils.rotation_angle(rotation_range[1], rotation_range[3])
            rz = utils.rotation_angle(rotation_range[2], rotation_range[3])
            rotation = euler2mat(rx, ry, rz, order)

            sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            scaling = np.diag([sx, sy, sz])

            xforms[i, :] = np.matmul(scaling, rotation).T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data

    def _augment_batch_data_l13(self, batch_data, batch_size, rotation_range=(0, math.pi, 0, 'u'), scaling_range=(0.001, 0.001, 0.001, 'g'), order='rxyz'):
        xforms = np.empty(shape=(batch_size, 3, 3))
        for i in range(batch_size):
            rx = utils.rotation_angle(rotation_range[0], rotation_range[3])
            ry = utils.rotation_angle(rotation_range[1], rotation_range[3])
            rz = utils.rotation_angle(rotation_range[2], rotation_range[3])
            rotation = euler2mat(rx, ry, rz, order)

            sx = utils.scaling_factor(scaling_range[0], scaling_range[3])
            sy = utils.scaling_factor(scaling_range[1], scaling_range[3])
            sz = utils.scaling_factor(scaling_range[2], scaling_range[3])
            scaling = np.diag([sx, sy, sz])

            xforms[i, :] = np.matmul(scaling, rotation).T
        batch_data = np.matmul(batch_data, xforms).astype(np.float32)
        return batch_data
    # def _has_next_batch(self):
    #     return self.batch_idx < self.num_batches

    def __len__(self):
        return self.num_batches

    # def __getitem__(self, i):
    #     return self._get_item(self.ids[i])
    #
    # def _next_batch(self):
    #     if self._has_next_batch():
    #         start_idx = self.batch_idx * self.batch_size
    #         bsize = self.batch_size
    #         batch_data = np.zeros((bsize, self.npoints, 3))
    #         batch_label = np.zeros((bsize, self.npoints))
    #         if self.use_weights:
    #             batch_weight = np.zeros((bsize, self.npoints))
    #         for i in range(bsize):
    #             item = self._get_item(self.ids[min(i+start_idx, self.ids.size-1)])
    #             batch_data[i] = item[0]
    #             batch_label[i] = item[1]
    #             if self.use_weights:
    #                 batch_weight[i] = item[2]
    #         if self.augment_data >= 0:
    #             batch_data, batch_label = self._augment_batch_data[self.augment_data](batch_data, batch_label)
    #             batch_data = utils.normalize_point_cloud_batch(batch_data)
    #         self.batch_idx += 1
    #         if self.use_weights:
    #             batch_data = [mx.ndarray.array(batch_data), mx.ndarray.array(batch_weight)]
    #         else:
    #             batch_data = [mx.ndarray.array(batch_data)]
    #         batch_data = batch_data + [mx.ndarray.ones((self.batch_size, 1), dtype='int32') * self.npoints]
    #         batch_label = [mx.ndarray.array(batch_label)]
    #         if self.use_cache:
    #             self.cache.append((batch_data, batch_label))
    #
    #     else:
    #         next_batch = self._get_data_from_cache()
    #         if next_batch is None:
    #             return None
    #         else:
    #             batch_data, batch_label = next_batch
    #     return mx.io.DataBatch(data=batch_data, label=batch_label)

    def sub_sample(self, batch_data, batch_label, batch_size, sample_num, point_nums_batch, max_nums, filter_extrema=2):
        batch_dataxyz_sub = np.zeros((batch_size, max_nums, 3), dtype=np.float32)
        batch_datafeat_sub = np.zeros((batch_size, max_nums, configs["indim"]), dtype=np.float32)
        points_num_batch_sub = np.zeros((batch_size, 1), dtype=np.int32)
        batch_label_sub =  np.full((batch_size, max_nums), 13, dtype=np.int32)
        for i in range(batch_size):
            pt_num = int(point_nums_batch[i])
            dataxyz = batch_data[i, :pt_num, :3]
            if configs['indim'] == 4:                 
                datafeat = np.concatenate([batch_data[i, :pt_num, 1, None] / 3 - 0.5, batch_data[i, :pt_num, 3:]], axis=-1)
            elif configs['indim'] == 3:  
                datafeat =batch_data[i, :pt_num, 3:]

            dataxyz, datafeat, label = data_utils.filter_extre(dataxyz, datafeat, batch_label[i, :pt_num], filter_extrema/2)
            pt_num = dataxyz.shape[0]
            if (sample_num * 0.7 > pt_num):
                print("discard one scene, sample_num - pt_num > pt_num", sample_num, pt_num)
                continue # discard this scene
            if pt_num >= sample_num:
                choices = np.array(random.sample(list(range(pt_num)), sample_num))
            else:
                ptarray = np.arange(pt_num)
                np.random.shuffle(ptarray)                
                choices = np.concatenate([ptarray, np.array(random.sample(list(range(pt_num)), sample_num - pt_num))])
            data_norm = dataxyz[choices, :]
            if self.normalize == "ball":
                data_norm = utils.normalize_point_cloud(data_norm[:, :3])
            elif self.normalize == "square":
                data_norm = utils.normalize_point_cloud_square(data_norm[:, :3])
            elif self.normalize == "ballnoy":
                data_norm = utils.normalize_point_cloud(data_norm[:, :3], double_axis=1)
            elif self.normalize == "squarenoy":
                data_norm = utils.normalize_point_cloud_square(data_norm[:, :3], double_axis=1)

            batch_dataxyz_sub[i, :len(choices),:] = data_norm
            batch_datafeat_sub[i, :len(choices),:] = datafeat[choices, :]
            points_num_batch_sub[i,0] = len(choices)
            batch_label_sub[i,:len(choices)] = label[choices]
        return batch_dataxyz_sub, batch_datafeat_sub, points_num_batch_sub, batch_label_sub

    def work(self, epoch, file_idx_train):
        if not self.stopped:
            if file_idx_train == 0 and self.shuffle:
                self.batch_counter = 0
                random.shuffle(self.seg_list)
                print("seg_list reordered!")
            filelist_train = self.seg_list[file_idx_train]
            if self.split == "train":
                data_train, _, data_num_train, label_train, _ = data_utils.load_seg(filelist_train)
            else:
                if self.data_val is None:
                    data_train, _, data_num_train, label_train, _ = data_utils.load_seg(filelist_train)
                    self.data_val = data_train
                    self.data_num_val = data_num_train
                    self.label_val = label_train
                data_train = self.data_val
                data_num_train = self.data_num_val
                label_train = self.label_val

            num_train = data_train.shape[0]
            if configs["indim"] == 0 and data_train.shape[-1] > 3:
                data_train = data_train[:, :, :3]
            data_train, data_num_train, label_train = \
                data_utils.grouped_shuffle([data_train, data_num_train, label_train])

            batch_num = num_train // self.batch_size

            for batch_idx_train in range(batch_num):
                # Training
                start_idx = (self.batch_size * batch_idx_train) % num_train
                end_idx = start_idx + self.batch_size
                # batch_size_train = end_idx - start_idx
                batch_data = data_train[start_idx:end_idx, ...]
                points_num_batch = data_num_train[start_idx:end_idx, ...]
                batch_label = label_train[start_idx:end_idx, ...]
                if self.split == "train":
                    offset = int(random.gauss(-self.npoints * self.drop_factor, self.npoints * self.drop_factor / 2))
                    offset = max(offset, - 2 * self.npoints * self.drop_factor)
                    offset = min(offset, 0)
                    sample_num_wofset = int(self.npoints + offset)
                else:
                    sample_num_wofset = int(self.npoints - self.npoints * self.drop_factor)
                batch_dataxyz, batch_datafeat, points_num_batch, batch_label = self.sub_sample(batch_data, batch_label, self.batch_size, sample_num_wofset, points_num_batch, self.npoints, filter_extrema=configs["scale"]/1.5*2)
                # batch_data

                if self.augment_data > 0:
                    batch_dataxyz = self._augment_batch_data[self.augment_data-1](batch_dataxyz, self.batch_size)
                    # batch_data = utils.normalize_point_cloud_batch(batch_data)

                if configs["postnorm"]:
                    batch_dataxyz = utils.normalize_point_cloud_in_scope(batch_dataxyz)

                batch_data = [mx.ndarray.array(batch_dataxyz), mx.ndarray.array(batch_datafeat), mx.ndarray.array(points_num_batch,dtype='int32')]
                batch_label = [mx.ndarray.array(batch_label)]
                self.batch_counter+=1
                if not self.stopped:
                    self.queue.put([mx.io.DataBatch(data=batch_data, label=batch_label),epoch,self.batch_counter])


    def run(self):
        while self.bno < self.max_h5_group_num and not self.stopped:
            self.work(self.bno // len(self.seg_list), self.bno % len(self.seg_list))
            self.bno += 1


    def fetch(self):
        if self.stopped:
            return None
        # else:
        #     print("queue length", self.queue.qsize())
        return self.queue.get()

    def shutdown(self):
        self.stopped = True
        while not self.queue.empty():
            self.queue.get()

    def _get_weight(self, type):
        weight_array = None
        if type == "none":
            return None
        elif type == "log":
            weight_array = np.array([1.0001497, 0.77643878, 0.842705, 0.64911157, 2.42792497, 2.57711294,
            1.50641103, 2.56347942, 2.02798182, 1.69989712, 4.21122346, 1.63516541, 3.24550962, 0])
        elif type == "negsoftmax":
            weight_array = np.array([0.03272517, 0.0132069, 0.01872605, 0.00465891, 0.1095493, 0.11351599, 
            0.07107983, 0.11317084, 0.09641226, 0.08187122, 0.13949018, 0.07847002, 0.12712334, 0]) * 40
        elif type == "invavg":
            weight_array = np.array([0.65881578, 0.41232351, 0.48167147, 0.2884141, 3.22950762, 3.64862773,
            1.34731719, 3.60853431, 2.28631003, 1.66424845, 14.7004472, 1.5544781, 6.21652445, 0])
        return weight_array.tolist()

    # none,log1.2,invavg,negsoftmax
# softmax  [0.03272517 0.0132069  0.01872605 0.00465891 0.1095493  0.11351599
#  0.07107983 0.11317084 0.09641226 0.08187122 0.13949018 0.07847002
#  0.12712334]
# 1/avg  [ 0.65881578  0.41232351  0.48167147  0.2884141   3.22950762  3.64862773
#   1.34731719  3.60853431  2.28631003  1.66424845 14.7004472   1.5544781
#   6.21652445]
# np.log(1.2 + weights): 

    


if __name__ == "__main__":

    # dataloader = S3DISLoader(root='data/scannet', npoints=8192, batch_size=1)
    # batch_num_sum = 0
    # for file_idx_train in range(len(dataloader.seg_list)):
    #     filelist_train = dataloader.seg_list[file_idx_train]
    #     data_train, _, data_num_train, label_train, _ = data_utils.load_seg(filelist_train)
    #     num_train = data_train.shape[0]
    #     if data_train.shape[-1] > 3:
    #         data_train = data_train[:, :, :3]
    #     data_train, data_num_train, label_train = \
    #         data_utils.grouped_shuffle([data_train, data_num_train, label_train])
    #
    #     batch_num = num_train // dataloader.batch_size
    #     batch_num_sum += batch_num
    # print("batch_num_sum", batch_num_sum)

    # filelist = ["../data/3DIS/prepare_label_rgb/Area_5/storage_4/half_0.h5", "../data/3DIS/prepare_label_rgb/Area_5/office_1/zero_0.h5"]
    # dirs = ["vis/storage_4_half_0", "vis/office_1_zero_0"]
    # dataloader = S3DISLoader('../data/3DIS/prepare_label_rgb/train_files_for_val_on_Area_1.txt')
    # file_ind=0
    # for line in filelist:
    #     dir = dirs[file_ind]
    #     os.makedirs(dir, exist_ok=True)
    #     data = h5py.File(line.strip(), 'r')
    #     points = data['data'][...].astype(np.float32)
    #     points[:,:,3:] = 255*(points[:,:,3:] + 0.5)
    #     labels=data['label'][...].astype(np.int32)
    #     point_nums=data['data_num'][...].astype(np.int32)
    #     labels_seg=data['label_seg'][...].astype(np.int32)

    #     if 'indices_split_to_full' in data:
    #         indices_split_to_full=data['indices_split_to_full'][...].astype(np.int64)

    #     # if file_ind==1:
    #         # print("point x max min ", np.max(points[:,:,0], axis=1), np.min(points[:,:,0], axis=1))
    #         # print("point y max", np.max(points[:,:,1], axis=1),np.min(points[:,:,1], axis=1) )
    #         # print("point z max", np.max(points[:,:,2], axis=1),np.min(points[:,:,2], axis=1))

    #         # print()
    #         # print("----------------")
    #         # pointbyx = points[0,:,0]
    #         # pointbyx = np.sort(pointbyx)
    #         # print(np.mean(pointbyx), np.median(pointbyx), pointbyx[:100], pointbyx[-100:])
    #         # print("----------------")
    #         # pointbyy = points[0,:,1]
    #         # pointbyy = np.sort(pointbyy)
    #         # print(np.mean(pointbyy),np.median(pointbyy), pointbyy[:100], pointbyy[-100:])
    #         #
    #         # print("----------------")
    #         # pointbyz = points[0,:,2]
    #         # pointbyz = np.sort(pointbyz)
    #         # print(np.mean(pointbyz),np.median(pointbyz), pointbyz[:100], pointbyz[-100:])
    #         #
    #         # print(np.where(np.abs(pointbyx-np.mean(pointbyx))>2))

    #     file_ind+=1
    #     for j in range(points.shape[0]):
    #         data, _, _ = data_utils.filter_extre(points[j,:point_nums[j],:3],points[j,:point_nums[j],3:], labels_seg[j,:point_nums[j]].reshape((-1,1)), 1)
    #         # points[j, :data.shape[0], :3] = data
    #         print(data.shape[0],"/", point_nums[j])
    #         points1 = utils.normalize_point_cloud(data[:, :3])
    #         points2 = utils.normalize_point_cloud_square(data[:, :3])
    #         points3 = utils.normalize_point_cloud(data[:, :3], double_axis=1)
    #         points4 = utils.normalize_point_cloud_square(data[:, :3], double_axis=1)

    #         print("points1", np.max(points1,axis=0)-np.min(points1,axis=0))
    #         print("points2", np.max(points2,axis=0)-np.min(points2,axis=0))
    #         print("points3", np.max(points3,axis=0)-np.min(points3,axis=0))
    #         print("points4", np.max(points4,axis=0)-np.min(points4,axis=0))
    #         print("data", np.max(data,axis=0)-np.min(data,axis=0))

    #         np.savetxt(os.path.join(dir, str(j)+".txt"), points[j,:,:3], delimiter=";")
    #         np.savetxt(os.path.join(dir, str(j)+"_ball.txt"), points1, delimiter=";")
    #         np.savetxt(os.path.join(dir, str(j)+"_square.txt"), points2, delimiter=";")
    #         np.savetxt(os.path.join(dir, str(j)+"_ball_1.txt"), points3, delimiter=";")
    #         np.savetxt(os.path.join(dir, str(j)+"_square_1.txt"), points4, delimiter=";")
    #         np.savetxt(os.path.join(dir, str(j)+"_data.txt"), data, delimiter=";")
    #         points1_aug = dataloader._augment_batch_data_l6(np.array([points3]), 1)
    #         points2_aug = dataloader._augment_batch_data_l8(np.array([points3]), 1)
    #         points3_aug = dataloader._augment_batch_data_l9(np.array([points3]), 1)
    #         points4_aug = dataloader._augment_batch_data_l11(np.array([points3]), 1)
    #         np.savetxt(os.path.join(dir, str(j)+"aug6.txt"), points1_aug[0], delimiter=";")
    #         np.savetxt(os.path.join(dir, str(j)+"aug8.txt"), points2_aug[0], delimiter=";")
    #         np.savetxt(os.path.join(dir, str(j)+"aug9.txt"), points3_aug[0], delimiter=";")
    #         np.savetxt(os.path.join(dir, str(j)+"aug11.txt"), points4_aug[0], delimiter=";")

    num_batches_lst=[]
    for i in range(1,7):
        train_loader = S3DISLoader(
                    '../data/3DIS/prepare_label_rgb/train_files_for_val_on_Area_{}.txt'.format(i),
                    batch_size=1,
                    npoints=8192,
                    normalize=configs["normalize"],
                    augment_data=-1,
                    shuffle=False,
                    max_epoch=configs["num_epochs"],
                    drop_factor=configs['input_dropout_ratio']
            )
        filelists = train_loader.seg_list
        num_batches = 0
        for file in filelists:  
            _, _, data_num_train, label_train, _ = data_utils.load_seg(file)
            print(file)
            print(label_train.shape[0])
            num_batches += label_train.shape[0]      
        num_batches_lst.append(num_batches)
    print(num_batches_lst)
    # weights = np.zeros(13)
    # for file in filelists:  
    #     _, _, data_num_train, label_train, _ = data_utils.load_seg(file)
    #     print(file)
    #     for i in range(label_train.shape[0]):
    #             labels = label_train[i, :data_num_train[i]]
    #             # print(np.histogram(labels, range(14))[0])
    #             weights += np.histogram(labels, range(14))[0]
    #     print(weights)      
    # print(weights,"for amount")            

    # weights /= np.average(weights)
    # print("per class weight:", weights)
    # class_weights = np.exp(-weights)/ np.sum( np.exp(- weights)) 
    # print("softmax ", class_weights)
    # class_weights = 1 / weights
    # print("1/avg ", class_weights)
    # class_weights = 1. / np.log(1.2 + weights)
    # print("np.log(1.2 + weights):", class_weights)