import os, sys, h5py
import mxnet as mx
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from utils import utils
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT = os.path.join(BASE_DIR,"../data")

def _get_data_files(list_filename):
    with open(list_filename) as f:
        return [line.rstrip()[5:] for line in f]


def _load_data_file(name):
    f = h5py.File(name,'r')
    data = f['data'][:]
    label = f['label'][:]

    return data, label


class ModelNet40Loader(mx.io.DataIter):

    def __init__(
            self, root, configs, batch_size=32, npoints=1024, split='train', normalize="ball", normal_channel=False, augment_level=0, balance=False, cache_size=15000, shuffle=False, dropout_ratio=0, include_trailing=False, point_set = "partial", dataset="40"):
        
        # super().__init__()
        self._augment_batch_data = [self._augment_batch_data_level1, self._augment_batch_data_level2, self._augment_batch_data_level3, self._augment_batch_data_level4, self._augment_batch_data_level5,self._augment_batch_data_level6, self._augment_batch_data_level7, self._augment_batch_data_level8, self._augment_batch_data_level9, self._augment_batch_data_level10]
        self.configs = configs
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.include_trailing = include_trailing
        self.augment_level = augment_level
        self.folder = "modelnet{}_ply_hdf5_2048".format(dataset)
        self.data_dir = os.path.join(ROOT, self.folder)
        self.balance = balance
        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple
        self.npoints = npoints
        self.point_set = point_set
        self.normal_channel = normal_channel
        self.split = split
        if split=='train':
            self.files = _get_data_files( \
                os.path.join(self.data_dir, 'train_files.txt'))
        else:
            self.files = _get_data_files( \
                os.path.join(self.data_dir, 'test_files.txt'))

        point_list, label_list = [], []
        for f in self.files:
            points, labels = _load_data_file(os.path.join(ROOT, f))
            point_list.append(points)
            label_list.append(labels)

        self.dropout_ratio = dropout_ratio
        self.points = np.concatenate(point_list, 0)
        self.label = np.concatenate(label_list, 0)
        self.num_samples = self.label.shape[0]
        print("num_samples", self.num_samples)
        self.label_shape = (self.batch_size,)
        self.rotation_angle = 0

        self.data_shape = (self.batch_size, self.npoints, self.configs['num_channel'])
        print("self.data_shape", self.data_shape)
        self.reset()

    def reset(self):
        self.ids = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.ids)
        print("shuffled: self.ids[0] is ", self.ids[0])
        self.num_batches = self.num_samples // self.batch_size
        if self.include_trailing and self.num_batches * self.batch_size < self.num_samples:
            self.num_batches += 1  # final batch
            self.trailing_count = self.num_samples - self.num_batches * self.batch_size
        self.batch_idx = 0


    def _get_item(self, idx):
        """
        get point_set (with shape (npoints,num_channel)), cls for training data `index`
        index: index for self.datapath
        """

        if self.point_set == "full":
            pt_idxs = np.arange(0, self.points.shape[1])  # 2048
            np.random.shuffle(pt_idxs)
            point_set = self.points[idx, pt_idxs].copy()
        else:
            point_set = self.points[idx, ...].copy()
        cls = np.array(self.label[[idx]]).astype(np.int32)
        point_set = point_set[0:self.npoints, :]
        if self.normalize == "ball":
            point_set[:, 0:3] = utils.normalize_point_cloud(point_set[:, 0:3])
        if self.normalize == "square":
            point_set[:, 0:3] = utils.normalize_point_cloud_square(point_set[:, 0:3])
        if not self.normal_channel:
            point_set = point_set[:, 0:3]
        return point_set, cls

    @property
    def provide_data(self):
        data_shapes = [('data', self.data_shape), ('actual_centnum', (self.batch_size, 1), "int32")]
        return data_shapes

    @property
    def provide_label(self):
        return [('label', self.label_shape)]

    def next(self):
        if self._has_next_batch():
            return self._next_batch()
        else:
            raise StopIteration



    def _has_next_batch(self):
        return self.batch_idx < self.num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, i):
        return self._get_item(self.ids[i])

    def _next_batch(self):
        start_idx = self.batch_idx * self.batch_size
        bsize = self.batch_size
        batch_data = np.zeros(self.data_shape)
        batch_label = np.zeros(self.label_shape, dtype=np.int32)
        for i in range(bsize):
            if self.balance:
                cls = np.random.choice(np.arange(len(self.classes)))
                index = np.random.choice(self.class_ids[cls])
                ps, cls = self._get_item(index)
            else:
                ps, cls = self._get_item(self.ids[min(i + start_idx, self.ids.size - 1)])
            batch_data[i] = ps
            batch_label[i] = cls
        if self.augment_level >0:
            batch_data = self._augment_batch_data[self.augment_level-1](batch_data)

        if self.split == 'test' and self.rotation_angle != 0:
            batch_data = self._rotate_batch_data(batch_data)
        if self.normalize == "ball" and self.configs["postnorm"]:
            batch_data[:, :, 0:3] = utils.normalize_point_cloud_batch(batch_data[:, :, 0:3])
        if self.normalize == "square" and self.configs["postnorm"]:
            batch_data[:, :, 0:3] = utils.normalize_point_cloud_batch_square(batch_data[:, :, 0:3])
        if self.split != 'test':
            batch_data = utils.shuffle_points(batch_data)
        self.batch_idx += 1
        batch_data_lst = [mx.ndarray.array(batch_data)]
        batch_data_lst = batch_data_lst + [mx.ndarray.ones((self.batch_size, 1), dtype='int32') * self.npoints]
        batch_label_lst = [mx.ndarray.array(batch_label)]
        return mx.io.DataBatch(data=batch_data_lst, label=batch_label_lst)

    def _augment_batch_data_level1(self, batch_data):
        if self.normal_channel:
            batch_data = utils.rotate_point_cloud_with_normal(batch_data)
        else:
            batch_data = utils.rotate_point_cloud(batch_data)
        # batch_data[:,:,:3] = utils.random_scale_point_cloud(batch_data[:,:,:3])
        batch_data[:, :, :3] = utils.jitter_point_cloud(batch_data[:, :, :3])
        if self.dropout_ratio > 0:
            batch_data = utils.random_point_dropout(batch_data, max_dropout_ratio=self.dropout_ratio)
        return batch_data

    def _augment_batch_data_level2(self, batch_data):
        if self.normal_channel:
            rotated_data = utils.rotate_point_cloud_with_normal(batch_data)
            rotated_data = utils.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = utils.rotate_point_cloud(batch_data)
            rotated_data = utils.rotate_perturbation_point_cloud(rotated_data)

        jittered_data = utils.random_scale_point_cloud(rotated_data[:, :, 0:3], scale_low=1.0, scale_high=1.15)
        # jittered_data = utils.shift_point_cloud(jittered_data)
        jittered_data = utils.jitter_point_cloud(jittered_data)
        rotated_data[:, :, 0:3] = jittered_data
        if self.dropout_ratio > 0:
            rotated_data = utils.random_point_dropout(rotated_data, max_dropout_ratio=self.dropout_ratio)
        return rotated_data

    def _augment_batch_data_level3(self, batch_data):
        if self.normal_channel:
            rotated_data = utils.rotate_point_cloud_with_normal(batch_data)
            rotated_data = utils.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = utils.rotate_point_cloud(batch_data)
            rotated_data = utils.rotate_perturbation_point_cloud(rotated_data)

        jittered_data = utils.random_scale_point_cloud(rotated_data[:, :, 0:3])
        jittered_data = utils.shift_point_cloud(jittered_data)
        jittered_data = utils.jitter_point_cloud(jittered_data)
        rotated_data[:, :, 0:3] = jittered_data
        if self.dropout_ratio > 0:
            rotated_data = utils.random_point_dropout(rotated_data, max_dropout_ratio=self.dropout_ratio)
        return rotated_data

    def _augment_batch_data_level4(self, batch_data):
        if self.normal_channel:
            rotated_data = utils.rotate_point_cloud_with_normal(batch_data)
            rotated_data = utils.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = utils.rotate_point_cloud(batch_data)
            rotated_data = utils.rotate_perturbation_point_cloud(rotated_data)
        if self.dropout_ratio > 0:
            rotated_data = utils.random_point_dropout(rotated_data, max_dropout_ratio=self.dropout_ratio)
        return rotated_data

    def _augment_batch_data_level5(self, batch_data):
        batch_data[:, :, 0:3] = utils.jitter_point_cloud(batch_data[:,:,0:3])
        if self.dropout_ratio > 0:
            batch_data = utils.random_point_dropout(batch_data, max_dropout_ratio=self.dropout_ratio)
        return batch_data

    def _augment_batch_data_level6(self, batch_data):
        jittered_data = utils.random_scale_point_cloud(batch_data[:, :, 0:3])
        jittered_data = utils.shift_point_cloud(jittered_data)
        batch_data[:, :, 0:3] = jittered_data
        if self.dropout_ratio > 0:
            batch_data = utils.random_point_dropout(batch_data, max_dropout_ratio=self.dropout_ratio)
        return batch_data

    def _augment_batch_data_level7(self, batch_data):
        jittered_data = utils.random_scale_point_cloud(batch_data[:, :, 0:3], scale_low=0.7, scale_high=1.4)
        jittered_data = utils.shift_point_cloud(jittered_data, shift_range=0.1)
        batch_data[:, :, 0:3] = jittered_data
        if self.dropout_ratio > 0:
            batch_data = utils.random_point_dropout(batch_data, max_dropout_ratio=self.dropout_ratio)
        return batch_data

    def _augment_batch_data_level8(self, batch_data):
        jittered_data = utils.random_scale_point_cloud(batch_data[:, :, 0:3], scale_low=0.75, scale_high=1)
        jittered_data = utils.shift_point_cloud(jittered_data, shift_range=0.05)
        batch_data[:, :, 0:3] = jittered_data
        if self.dropout_ratio > 0:
            batch_data = utils.random_point_dropout(batch_data, max_dropout_ratio=self.dropout_ratio)
        return batch_data

    def _augment_batch_data_level9(self, batch_data):
        jittered_data = utils.random_scale_point_cloud(batch_data[:, :, 0:3])
        batch_data[:, :, 0:3] = jittered_data
        if self.dropout_ratio > 0:
            batch_data = utils.random_point_dropout(batch_data, max_dropout_ratio=self.dropout_ratio)
        return batch_data

    def _augment_batch_data_level10(self, batch_data):
        jittered_data = utils.shift_point_cloud(batch_data[:, :, 0:3])
        batch_data[:, :, 0:3] = jittered_data
        if self.dropout_ratio > 0:
            batch_data = utils.random_point_dropout(batch_data, max_dropout_ratio=self.dropout_ratio)
        return batch_data

    def _rotate_batch_data(self, batch_data):
        if self.normal_channel:
            return utils.rotate_point_cloud_by_angle_with_normal(batch_data, self.rotation_angle)
        else:
            return utils.rotate_point_cloud_by_angle(batch_data, self.rotation_angle)

    def _cal_label_distribution(self):
        x = np.zeros(len(self.classes))
        for shape_name, _ in self.datapath:
            x[self.classes[shape_name]] += 1
        x /= x.sum()
        res = 1. / np.log(1.2 + x)
        print(res)
        return res.tolist()

    def get_weight_gradient_multiplier(self):
        return self._cal_label_distribution()


if __name__ == "__main__":
    CLASS_NAMES10 = {0: 'bathtub', 1: 'bed', 2: 'chair', 3: 'desk', 4: 'dresser', 5: 'monitor', 6: 'night_stand',
                   7: 'sofa', 8: 'table', 9: 'toilet'}

    CLASS_NAMES40 = {0: 'airplane', 1: 'bathtub', 2: 'bed', 3: 'bench', 4: 'bookshelf', 5: 'bottle', 6: 'bowl', 7: 'car', 8: 'chair',
     9: 'cone', 10: 'cup', 11: 'curtain', 12: 'desk', 13: 'door', 14: 'dresser', 15: 'flower_pot', 16: 'glass_box',
     17: 'guitar', 18: 'keyboard', 19: 'lamp', 20: 'laptop', 21: 'mantel', 22: 'monitor', 23: 'night_stand',
     24: 'person', 25: 'piano', 26: 'plant', 27: 'radio', 28: 'range_hood', 29: 'sink', 30: 'sofa', 31: 'stairs',
     32: 'stool', 33: 'table', 34: 'tent', 35: 'toilet', 36: 'tv_stand', 37: 'vase', 38: 'wardrobe', 39: 'xbox'}

    matchclass={1:0, 2:1, 8:2, 13:3, 14:4, 22:5, 23:6, 30:7, 33:8, 35:9}
    infolder = "data/modelnet40_ply_hdf5_2048"
    outfolder = "data/modelnet10_ply_hdf5_2048"
    os.makedirs(outfolder, exist_ok=True)
    intrainfiles = _get_data_files( \
            os.path.join(infolder, 'train_files.txt'))
    intestfiles = _get_data_files( \
            os.path.join(infolder, 'test_files.txt'))
    outtrainfile = "ply_data_train0.h5"
    outtestfile = "ply_data_test0.h5"



    for trainfiles, outtrainfile in zip([intrainfiles, intestfiles],[outtrainfile, outtestfile]):
        intrain_point_list = []
        intrain_label_list = []
        for f in trainfiles:
            points, labels = _load_data_file(os.path.join("data", f))
            intrain_point_list.append(points)
            intrain_label_list.append(labels)
        points = np.concatenate(intrain_point_list, 0)
        label = np.concatenate(intrain_label_list, 0)
        pass_p_lst=[]
        pass_l_lst=[]
        for i in range(points.shape[0]):
            if label[i][0] in matchclass:
                pass_p_lst.append(points[i])
                pass_l_lst.append([matchclass[label[i][0]]])
        points = np.array(pass_p_lst)
        label = np.array(pass_l_lst)
        with h5py.File(os.path.join(outfolder, outtrainfile), 'w') as f:
            f.create_dataset("data", data=points)
            f.create_dataset("label", data=label)
