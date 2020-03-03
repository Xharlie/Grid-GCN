import sys
if sys.version_info[0] == 3:
    from . import _init_paths
    import pickle
else:
    import _init_paths
    import cPickle as pickle
import mxnet as mx
import numpy as np
import os, json
import random
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import mxnet as mx
from shapenetpart.configs.configs import configs
from utils import utils


CLS_NUM = 16
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ShapeNetPart(mx.io.DataIter):
    def __init__(self, root, num_points = 2048, split='train', augment_data = -1, normalize=True, dropout_ratio=0, use_cache=False, cache_file='', shuffle=False, batch_size=8):
        self.dropout_ratio =dropout_ratio
        self._augment_batch_data = [self._augment_batch_data_l1, self._augment_batch_data_l2,self._augment_batch_data_l3]
        self.augment_data = augment_data
        self.num_points = num_points
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.normalize = normalize
        self.shuffle=shuffle
        self.batch_size=batch_size
        self.use_cache =use_cache
        self.cat = {}
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k:v for k,v in self.cat.items()}
        self.cache_file = cache_file
        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            if split=='trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split=='train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split=='val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split=='test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..'%(split))
                exit(-1)
                
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
        self.num_samples = len(self.datapath)  
         
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        self.cache = []
        self.cache_collect = {}
        self.cache_size = 2000000

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

    def _load_cache(self):
        os.makedirs(os.path.join(self.root,"cached_test"),exist_ok=True)
        cache_file = os.path.join(self.root,"cached_test", self.cache_file)
        if not os.path.isfile(cache_file):
            print('cache file not found. Will generate new data')
            return False
        print('reading cache file from', self.cache_file)
        with open(cache_file, 'rb') as f:
            self.cache = pickle.load(f, encoding="bytes")
        return True

    @property
    def provide_data(self):
        return [('data', (self.batch_size, self.num_points, 3)), ('actual_centnum', (self.batch_size, 1), "int32")
                ,('cls', (self.batch_size, CLS_NUM), "float32")]

    @property
    def provide_label(self):
        return [('label', (self.batch_size, self.num_points))]

    def _dump_cache(self):
        with open(os.path.join(self.root, "cached_test", self.cache_file), 'wb') as f:
            pickle.dump(self.cache, f, protocol=-1)

    def _get_data_from_cache(self):
        if self.next_cache_index >= len(self.cache):
            return None, None, None
        data, label, cls = self.cache[self.next_cache_index]
        self.next_cache_index += 1
        return data, label, cls

    def __getitem__(self, index):
        if index in self.cache_collect:
            point_set, seg, cls = self.cache_collect[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int64)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:,0:3]
            if self.normalize:
                point_set = pc_normalize(point_set)
            seg = data[:,-1].astype(np.int64)
            if len(self.cache_collect) < self.cache_size:
                self.cache_collect[index] = (point_set, seg, cls)
                
        choice = np.random.choice(len(seg), self.num_points, replace=True)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        
        return point_set, seg, cls 
        
    def __len__(self):
        return self.num_batches // self.batch_size

    def next(self):
        batch = self._next_batch()
        if batch is None:
            raise StopIteration
        else:
            return batch

    def _has_next_batch(self):
        return self.batch_idx < self.num_batches

    def _next_batch(self):
        if not self.use_cache or self.first_run:
            if self._has_next_batch():
                start_idx = self.batch_idx * self.batch_size
                bsize = self.batch_size
                batch_data = np.zeros((bsize, self.num_points, 3))
                batch_label = np.zeros((bsize, self.num_points))
                batch_cls_onehot = np.zeros((bsize, CLS_NUM), dtype=np.float32)
                batch_cls = np.zeros((bsize, 1), dtype=np.int32)
                for i in range(bsize):
                    item = self.__getitem__(self.ids[min(i+start_idx, self.ids.size-1)])
                    batch_data[i] = item[0]
                    batch_label[i] = item[1]
                    batch_cls[i,0] = item[2]
                    batch_cls_onehot[i, item[2]] = 1.
                if self.augment_data >= 1:
                    batch_data, batch_label = self._augment_batch_data[self.augment_data-1](batch_data, batch_label)
                    if configs["postnorm"] and self.normalize:
                        batch_data = utils.normalize_point_cloud_batch(batch_data)
                    elif configs["postnorm"] and self.normalize:
                        batch_data = utils.normalize_point_cloud_batch_square(batch_data)
                self.batch_idx += 1
                batch_data = [mx.ndarray.array(batch_data),
                    mx.ndarray.ones((self.batch_size, 1), dtype='int32') * self.num_points,
                    mx.ndarray.array(batch_cls_onehot, dtype='float32')]
                batch_label = [mx.ndarray.array(batch_label)]
                if self.use_cache:
                    self.cache.append((batch_data, batch_label, batch_cls))
                    print("append new batch", batch_cls.shape)
            else:
                if self.use_cache:
                    self._dump_cache()
                    self.first_run = False
                return None
        else:
            next_batch = self._get_data_from_cache()
            self.batch_idx += 1
            # print(self.batch_idx, self.num_batches)
            if next_batch[0] is None:
                return None
            else:
                batch_data, batch_label, batch_cls = next_batch
        return mx.io.DataBatch(data=batch_data, label=batch_label), batch_cls

    def _augment_batch_data_l1(self, batch_data, batch_label):
        #batch_data = utils.rotate_point_cloud_z(batch_data)
        # batch_data = utils.rotate_perturbation_point_cloud(batch_data)
        batch_data = utils.random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25)
        batch_data = utils.shift_point_cloud(batch_data, shift_range=0.1)
        #batch_data = utils.jitter_point_cloud(batch_data)
        if self.dropout_ratio > 0:
            batch_data, batch_label = utils.random_point_dropout(batch_data, labels=batch_label, max_dropout_ratio=self.dropout_ratio)
        return batch_data, batch_label

    def _augment_batch_data_l2(self, batch_data, batch_label):
        #batch_data = utils.rotate_point_cloud_z(batch_data)
        # batch_data = utils.rotate_perturbation_point_cloud(batch_data)
        batch_data = utils.random_scale_point_cloud(batch_data, scale_low=2./3., scale_high=3./ 2.)
        batch_data = utils.shift_point_cloud(batch_data, shift_range=0.2)
        #batch_data = utils.jitter_point_cloud(batch_data)
        if self.dropout_ratio > 0:
            batch_data, batch_label = utils.random_point_dropout(batch_data, labels=batch_label, max_dropout_ratio=self.dropout_ratio)
        return batch_data, batch_label

    def _augment_batch_data_l3(self, batch_data, batch_label):
        #batch_data = utils.rotate_point_cloud_z(batch_data)
        # batch_data = utils.rotate_perturbation_point_cloud(batch_data)
        batch_data = utils.random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25)
        # batch_data = utils.shift_point_cloud(batch_data, shift_range=0.1)
        #batch_data = utils.jitter_point_cloud(batch_data)
        if self.dropout_ratio > 0:
            batch_data, batch_label = utils.random_point_dropout(batch_data, labels=batch_label, max_dropout_ratio=self.dropout_ratio)
        return batch_data, batch_label

if __name__ == "__main__":
    dataloader = ShapeNetPart(root='../data/shapenetcore_partanno_segmentation_benchmark_v0_normal',
                             num_points=2048, split='trainval', batch_size=1)
    print(dataloader.datapath)
