import _init_paths
import mxnet as mx
import numpy as np
import os
import random

from configs.configs import configs
from utils import utils

class ModelNet40Loader(mx.io.DataIter):
    """
    Data loader for ModelNet40
    """
    def __init__(self, root, batch_size=32, npoints=1024, split='train', normalize=True, normal_channel=False, augment_level=0, balance=False, cache_size=15000, shuffle=False, dropout_ratio=0, include_trailing=False, tile_2d=None):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.split = split
        self.normalize = normalize
        self.augment_level = augment_level
        self.balance = balance
        self.classes_file = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.classes = {line.rstrip(): i for i, line in enumerate(open(self.classes_file))}
        self.normal_channel = normal_channel
        self.shuffle = shuffle
        self.dropout_ratio = dropout_ratio
        self.rotation_angle = 0
        self.tile_2d = tile_2d

        # whether to include the trailing batch with less than batch_size samples
        # if True, the last batch will be padded by the last sample to make batch_size the same
        self.include_trailing = include_trailing
        assert self.split == 'test' or self.include_trailing == False, "include_trailing is only supported for testing"

        shape_ids = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_{}.txt'.format(split)))]
        shape_names = ['_'.join(idx.split('_')[0:-1]) for idx in shape_ids]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], idx+'.txt')) for i, idx in enumerate(shape_ids)]
        self.num_samples = len(self.datapath)

        self.cache_size = cache_size # how many data points to cache in memory
        self.cache = {} # from index to (point_set, cls) tuple

        if self.balance:
            # calculate #samples for each class
            self.class_ids = []
            for i in xrange(len(self.classes)):
                self.class_ids.append([])
            for i in xrange(self.num_samples):
                cls_name, shape_txt_filename = self.datapath[i]
                cls = self.classes[cls_name]
                self.class_ids[cls].append(i)

        self.data_shape = (self.batch_size, self.npoints, configs['num_channel'])
        if self.tile_2d is None:
            self.label_shape = (self.batch_size,)
        else:
            self.label_shape = (self.batch_size, self.tile_2d)
        self.reset()

    def reset(self):
        self.ids = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.ids)
        self.num_batches = self.num_samples // self.batch_size
        if self.include_trailing and self.num_batches * self.batch_size < self.num_samples:
            self.num_batches += 1  # final batch
            self.trailing_count = self.num_samples - self.num_batches * self.batch_size
        self.batch_idx = 0

    def set_rotation_angle(self, rotation_angle):
        assert self.split == 'test', "rotation_angle can only be set in test mode"
        self.rotation_angle = rotation_angle  # rotation angle is used to rotate the point cloud by a certain angle. Used in testing

    @property
    def provide_data(self):
        return [('data', self.data_shape)]

    @property
    def provide_label(self):
        return [('label', self.label_shape)]

    def next(self):
        if self._has_next_batch():
            return self._next_batch()
        else:
            raise StopIteration

    def _get_item(self, index):
        """
        get point_set (with shape (npoints,num_channel)), cls for training data `index`
        index: index for self.datapath
        """
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            cls_name, shape_txt_filename = self.datapath[index]
            cls = self.classes[cls_name]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(shape_txt_filename, delimiter=',').astype(np.float32)
            if configs["point_set"] != "full":
                point_set = point_set[0:self.npoints, :]
            if not self.normal_channel:
                point_set = point_set[:,0:3]
            if len(self.cache) < self.cache_size:
                if self.normalize:
                    point_set[:, 0:3] = utils.normalize_point_cloud(point_set[:, 0:3])
                self.cache[index] = (point_set, cls)
        if configs["point_set"] == "full":
            choice = np.asarray(random.sample(range(point_set.shape[0]), configs["num_points"]), dtype=np.int32)
            point_set = point_set[choice,...]
            if self.normalize:
                point_set[:, 0:3] = utils.normalize_point_cloud(point_set[:, 0:3])
        return point_set, cls

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
        for i in xrange(bsize):
            if self.balance:
                cls = np.random.choice(np.arange(len(self.classes)))
                index = np.random.choice(self.class_ids[cls])
                ps, cls = self._get_item(index)
            else:
                ps, cls = self._get_item(self.ids[min(i+start_idx, self.ids.size-1)])
            batch_data[i] = ps
            batch_label[i] = cls
        if self.augment_level == 1:
            batch_data = self._augment_batch_data_level1(batch_data)
        elif self.augment_level == 2:
            batch_data = self._augment_batch_data_level2(batch_data)
        if self.split == 'test' and self.rotation_angle != 0:
            batch_data = self._rotate_batch_data(batch_data)
        self.batch_idx += 1
        batch_data = [mx.ndarray.array(batch_data)]
        batch_label = [mx.ndarray.array(batch_label)]
        return mx.io.DataBatch(data=batch_data, label=batch_label)

    def _augment_batch_data_level1(self, batch_data):
        if self.normal_channel:
            batch_data = utils.rotate_point_cloud_with_normal(batch_data)
        else:
            batch_data = utils.rotate_point_cloud(batch_data)
        batch_data[:,:,:3] = utils.random_scale_point_cloud(batch_data[:,:,:3])
        batch_data = utils.shuffle_points(batch_data)
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

        jittered_data = utils.random_scale_point_cloud(rotated_data[:,:,0:3])
        jittered_data = utils.shift_point_cloud(jittered_data)
        jittered_data = utils.jitter_point_cloud(jittered_data)
        rotated_data[:,:,0:3] = jittered_data
        rotated_data = utils.shuffle_points(rotated_data)
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
        print res
        return res.tolist()

    def get_weight_gradient_multiplier(self):
        return self._cal_label_distribution()


if __name__ == "__main__":
    dataloader = ModelNet40Loader(root='data/modelnet40_normal_resampled', split='train')
    dataloader.reset()
    data = next(dataloader)
    print data.data[0].shape
    print data.label[0].shape
    print data.label[0]
    data = next(dataloader)
    print data.data[0].shape
    print data.label[0].shape
    print data.label[0]
