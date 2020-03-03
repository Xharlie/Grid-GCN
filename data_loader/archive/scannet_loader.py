# import _init_paths

import mxnet as mx
import numpy as np

import os
import random
import pickle
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from segmentation.configs.configs import configs
from utils import utils


class BaseScanNetLoader(mx.io.DataIter):
    def __init__(self, root, batch_size=32, npoints=8192, split='train', shuffle=False, normalize=False, augment_data=False, dropout_ratio=0, use_weights=False, use_cache=False, cache_file=''):
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.split = split
        assert self.split in ['train', 'test']
        print('Loading {} data'.format(self.split))
        with open(os.path.join(self.root, 'scannet_{}.pickle'.format(self.split)), "rb") as f:
            self.data = pickle.load(f, encoding="bytes")
            self.label = pickle.load(f, encoding="bytes")
        self.num_samples = len(self.label)

        self.shuffle = shuffle
        self.normalize = normalize
        self.augment_data = augment_data
        self.dropout_ratio = dropout_ratio
        self.use_weights = use_weights
        if self.use_weights:
            self.get_weight_gradient_multiplier()

        self.data_shape = (self.batch_size, self.npoints, 3)
        self.label_shape = (self.batch_size, self.npoints)
        if self.use_weights:
            self.weight_shape = (self.batch_size, self.npoints)

        self.use_cache = use_cache
        self.cache_file = cache_file
        self.cache = []
        if use_cache:
            self.first_run = not self._load_cache()
        else:
            self.first_run = True
        self.reset()

    def reset(self):
        self.ids = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.ids)
        self.num_batches = self.num_samples // self.batch_size
        self.batch_idx = 0
        self.next_cache_index = 0

    @property
    def provide_data(self):
        if self.use_weights:
            return [('data', self.data_shape), ('weight', self.weight_shape)]
        else:
            return [('data', self.data_shape)]

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
            self.cache = pickle.load(f)
        return True

    def _get_data_from_cache(self):
        if self.next_cache_index >= len(self.cache):
            return None
        data, label = self.cache[self.next_cache_index]
        self.next_cache_index += 1
        return data, label

    def next(self):
        batch = self._next_batch()
        if batch is None:
            raise StopIteration
        else:
            return batch

    def _augment_batch_data(self, batch_data, batch_label):
        #batch_data = utils.rotate_point_cloud_z(batch_data)
        batch_data = utils.rotate_perturbation_point_cloud(batch_data)
        #batch_data = utils.random_scale_point_cloud(batch_data)
        #batch_data = utils.shift_point_cloud(batch_data)
        #batch_data = utils.jitter_point_cloud(batch_data)
        if self.dropout_ratio > 0:
            batch_data, batch_label = utils.random_point_dropout(batch_data, labels=batch_label, max_dropout_ratio=self.dropout_ratio)
        return batch_data, batch_label

    def _cal_label_distribution(self):
        """ weights in softmax for each label """
        if self.split == 'train':
            self.label_weights = np.zeros(21, dtype=np.float32)
            for label_list in self.label:
                self.label_weights += np.histogram(label_list, range(22))[0]
            self.label_weights /= np.sum(self.label_weights)
            print("per class weight:", self.label_weights)
            self.label_weights = 1. / np.log(1.2 + self.label_weights)
            print("np.log(1.2 + self.label_weights):", self.label_weights)
        elif self.split == 'test':
            self.label_weights = np.ones(21)
        return self.label_weights.tolist()

    def get_weight_gradient_multiplier(self):
        return self._cal_label_distribution()


class ScanNetLoader(BaseScanNetLoader):
    """
    Data loader for ScanNet
    """
    def __init__(self, root, batch_size=32, npoints=8192, split='train', shuffle=False, normalize=False, augment_data=False, dropout_ratio=0, use_weights=False, use_cache=False, cache_file=''):
        super(ScanNetLoader, self).__init__(root, batch_size, npoints, split, shuffle, normalize, augment_data, dropout_ratio, use_weights, use_cache, cache_file)

    def _get_item(self, index):
        """
        get point_set (with shape (npoints, 3)), labels (with shape (npoints,)) for training data `index`, (optionally) weights (with shape (npoints,))
        randomly crop the whole scene
        index: index for self.data (axis 0)
        """
        data, label = self.data[index], self.label[index]
        zmax, zmin = data.max(axis=0)[2], data.min(axis=0)[2]
        for ind in range(10):
            center_idx = random.randint(0, data.shape[0]-1)  # randomly select a crop center, then check if it is a valid choice
            center = data[center_idx]
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
        ids = np.random.choice(crop_label.size, self.npoints, replace=True)
        data = crop_data[ids]
        label = crop_label[ids]
        mask = mask[ids]
        if self.normalize:
            data = utils.normalize_point_cloud(data)
        if self.use_weights:
            weight = self.label_weights[label] * mask
            return data, label, weight
        else:
            return data, label * mask

    def _has_next_batch(self):
        return self.batch_idx < self.num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, i):
        return self._get_item(self.ids[i])

    def _next_batch(self):
        if not self.use_cache or self.first_run:
            if self._has_next_batch():
                start_idx = self.batch_idx * self.batch_size
                bsize = self.batch_size
                batch_data = np.zeros((bsize, self.npoints, 3))
                batch_label = np.zeros((bsize, self.npoints))
                if self.use_weights:
                    batch_weight = np.zeros((bsize, self.npoints))
                for i in range(bsize):
                    item = self._get_item(self.ids[min(i+start_idx, self.ids.size-1)])
                    batch_data[i] = item[0]
                    batch_label[i] = item[1]
                    if self.use_weights:
                        batch_weight[i] = item[2]
                if self.augment_data:
                    batch_data, batch_label = self._augment_batch_data(batch_data, batch_label)
                self.batch_idx += 1
                if self.use_weights:
                    batch_data = [mx.ndarray.array(batch_data), mx.ndarray.array(batch_weight)]
                else:
                    batch_data = [mx.ndarray.array(batch_data)]
                batch_label = [mx.ndarray.array(batch_label)]
                if self.use_cache:
                    self.cache.append((batch_data, batch_label))
            else:
                if self.use_cache:
                    self._dump_cache()
                    self.first_run = False
                return None
        else:
            next_batch = self._get_data_from_cache()
            if next_batch is None:
                return None
            else:
                batch_data, batch_label = next_batch
        return mx.io.DataBatch(data=batch_data, label=batch_label)


class ScanNetWholeSceneLoader(BaseScanNetLoader):
    def __init__(self, root, batch_size=32, npoints=8192, split='train', shuffle=False, normalize=False, augment_data=False, dropout_ratio=0, use_weights=False, use_cache=False, cache_file=''):
        super(ScanNetWholeSceneLoader, self).__init__(root, batch_size, npoints, split, shuffle, normalize, augment_data, dropout_ratio, use_weights, use_cache, cache_file)

    def reset(self):
        self.buffer_data = np.zeros((0, self.npoints, 3))
        self.buffer_label = np.zeros((0, self.npoints))
        self.next_read_index = 0
        self.next_cache_index = 0

    def _read_buffer(self, index):
        next_data, next_label = self._get_item(index)
        self.buffer_data = np.concatenate((self.buffer_data, next_data), axis=0)
        self.buffer_label = np.concatenate((self.buffer_label, next_label), axis=0)

    def _get_item(self, index):
        """
        get point_set (with shape (B, npoints, 3)), labels (with shape (B, npoints)) for training data `index`
        randomly crop the whole scene
        index: index for self.data (axis 0)
        """
        data, label = self.data[index], self.label[index]
        coordmax = np.max(data, axis=0)
        coordmin = np.min(data, axis=0)
        nsubvolume_x = np.ceil((coordmax[0]-coordmin[0])/1.5).astype(np.int32)
        nsubvolume_y = np.ceil((coordmax[1]-coordmin[1])/1.5).astype(np.int32)
        batch_data, batch_label = [], []
        for i in range(nsubvolume_x):
            for j in range(nsubvolume_y):
                curmin = coordmin + [i*1.5, j*1.5, 0]
                curmax = coordmin+ [(i+1)*1.5, (j+1)*1.5, coordmax[2]-coordmin[2]]
                crop_ids = np.sum((data>=(curmin-0.2)) * (data<=(curmax+0.2)), axis=1) == 3
                if sum(crop_ids) == 0: continue
                crop_data = data[crop_ids]
                crop_label = label[crop_ids]
                mask = np.sum((crop_data>=(curmin-0.001)) * (crop_data<=(curmax+0.001)), axis=1) == 3
                ids = np.random.choice(crop_label.size, self.npoints, replace=True)
                this_data = crop_data[ids]
                this_label = crop_label[ids]
                this_mask = mask[ids]
                if sum(this_mask) * 1. / this_mask.size < 0.01: continue
                this_label *= this_mask
                if self.normalize:
                    this_data = utils.normalize_point_cloud(this_data)
                batch_data.append(this_data[None,:,:])
                batch_label.append(this_label[None,:])
        batch_data = np.concatenate(tuple(batch_data), axis=0)
        batch_label = np.concatenate(tuple(batch_label), axis=0)
        return batch_data, batch_label

    def _next_batch(self):
        """
        Get next batch of data by using a buffer and (optionally) using the cached data
        When no more batch_data is available, None is returned
        """
        if not self.use_cache or self.first_run:
            while self.buffer_label.shape[0] < self.batch_size:
                # read more samples into buffer
                if self.next_read_index >= self.num_samples:
                    if self.use_cache:
                        self._dump_cache()
                        self.first_run = False
                    return None
                self._read_buffer(self.next_read_index)
                self.next_read_index += 1
            # get the next batch from buffer
            data, self.buffer_data = self.buffer_data[:self.batch_size], self.buffer_data[self.batch_size:]
            label, self.buffer_label = self.buffer_label[:self.batch_size], self.buffer_label[self.batch_size:]
            if self.use_cache:
                self.cache.append((data, label))
        else:
            next_batch = self._get_data_from_cache()
            if next_batch is None:
                return None
            else:
                data, label = next_batch
        return mx.io.DataBatch(data=[mx.nd.array(data)], label=[mx.nd.array(label)])


if __name__ == "__main__":
    dataloader = ScanNetWholeSceneLoader(root='data/scannet', npoints=10000, split='test', batch_size=1)
    dataloader.reset()
    # draw some randomly selected points of the whole scene
    #for i in xrange(5):
    #    data = dataloader.data[i]
    #    label = dataloader.label[i]
    #    ids = np.random.choice(label.size, 1024, replace=False)
    #    data = data[ids]
    #    label = label[ids]
    #    utils.draw_point_cloud_with_labels(data, label)
    # draw some samples
    VALID_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    label_num = {}
    for id in VALID_CLASS_IDS:
        label_num[id] = 0
    total = 0
    for batchid in range(dataloader.num_samples):
        label = dataloader.label[batchid]
        print(batchid, label_num)
        for i in range(label.shape[0]):
            label_pt = label[i]
        # utils.draw_point_cloud_with_labels(data, label)
            if label_pt in VALID_CLASS_IDS:
                total+=1
                label_num[label_pt]+=1
    print(label_num, total)
    print(label_num/total)



    #