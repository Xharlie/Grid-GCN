import os, sys, h5py
import mxnet as mx
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from classification.configs.configs import configs
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
            self, root, batch_size=32, npoints=1024, split='train', normalize=True, normal_channel=False, augment_level=0, balance=False, cache_size=15000, shuffle=False, dropout_ratio=0, include_trailing=False, point_set = "partial"):
        
        super().__init__()
        self.normalize = normalize
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.include_trailing = include_trailing
        self.augment_level = augment_level
        self.folder = "modelnet40_ply_hdf5_2048"
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

        self.data_shape = (self.batch_size, self.npoints, configs['num_channel'])
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
        pt_idxs = np.arange(0, self.points.shape[1])  # 2048
        if self.split=='train':
            np.random.shuffle(pt_idxs)
        point_set = self.points[idx, pt_idxs].copy()
        cls = np.array(self.label[[idx]]).astype(np.int32)
        if self.point_set != "full":
            point_set = point_set[0:self.npoints, :]
        else:
            choice = np.asarray(random.sample(range(point_set.shape[0]), configs["num_points"]), dtype=np.int32)
            point_set = point_set[choice, ...]
        if self.normalize:
            point_set[:, 0:3] = utils.normalize_point_cloud(point_set[:, 0:3])
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
        if self.augment_level == 1:
            batch_data = self._augment_batch_data_level1(batch_data)
        elif self.augment_level == 2:
            batch_data = self._augment_batch_data_level2(batch_data)
        if self.split == 'test' and self.rotation_angle != 0:
            batch_data = self._rotate_batch_data(batch_data)
        if self.normalize and configs["postnorm"]:
            batch_data[:, :, 0:3] = utils.normalize_point_cloud_batch(batch_data[:, :, 0:3])
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
    dset = ModelNet40Loader(16, "./")
