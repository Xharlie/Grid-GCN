import mxnet as mx
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

import models.pnt2_module
from models.pnt2_models import get_symbol_seg
from data_loader.scannet_loader import ScanNetLoader
from train.base_solver_pnt import BaseSolver
from utils.utils import point_cloud_label_to_surface_voxel_label
from utils import metrics
from configs.configs import configs_pnt as configs
import mxnet.profiler as profiler
import time

# if configs['profiling']:
#     # os.environ["MXNET_EXEC_BULK_EXEC_INFERENCE"] = "0"
#     profiler.set_config(profile_all=True,
#                         aggregate_stats=True,
#                         filename='{}.json'.format(configs["pf_names"]))
class ScanNetSolver(BaseSolver):
    def __init__(self):
        super(ScanNetSolver, self).__init__()

    def _specify_input_names(self):
        self.data_names = ['data']
        self.label_names = ['label']


    def _specify_input_names(self):
        self.data_names = ['data']
        self.label_names = ['label']

    def _get_data_loaders(self):
        self.train_loader = ScanNetLoader(
                root=configs['data_dir'],
                batch_size=self.batch_size,
                npoints=self.num_points,
                split='train',
                normalize=True,
                augment_data=True,
                dropout_ratio=configs['input_dropout_ratio'],
                shuffle=True,
        )
        self.val_loader = ScanNetLoader(
                root=configs['data_dir'],
                batch_size=self.batch_size,
                npoints=self.num_points,
                split='test',
                normalize=True,
        )

    def _get_symbol(self):
        self.symbol = get_symbol_seg(self.batch_size // self.num_devices, self.num_points, bn_decay=self.bn_decay, weights=self.weights)

    def _get_metric(self):
        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.AccuracyWithIgnore(axis=1, ignore_label=0),
            metrics.CrossEntropyWithIgnore(ndim=3, axis=1, ignore_label=0, label_weights=self.weights)])
        self.val_metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.AccuracyWithIgnore(axis=1, ignore_label=0),
            metrics.CrossEntropyWithIgnore(ndim=3, axis=1, ignore_label=0, label_weights=None), metrics.PerClassAccuracy(num_class=21, axis=1, ignore_label=0)])

    def evaluate(self, epoch):
        """
        When evaluating, convert per-point accuracy to per-voxel accuracy
        """
        self.val_loader.reset()
        corr, total = 0., 0.
        corr_vox_02, total_vox_02 = 0., 0.
        corr_vox_0484, total_vox_0484 = 0., 0.
        self.val_metric.reset()
        total_seen_class_vox = [0 for _ in range(21)]
        total_correct_class_vox = [0 for _ in range(21)]
        if configs['profiling']:
            profiler.set_state('run')
        all = 0
        count=0
        for batch in self.val_loader:
            count+=1
            if count > 3:
                tic = time.time()
            self.module.forward(batch, is_train=False)
            if not configs['profiling']:
                if count > 3:
                    mx.nd.waitall()
                    all += time.time() - tic
                    if count > 33:
                        print("count:{}, count-3: {}, all inference time is:{}, average inference time is: {}s".format(count, count-3, all, all/(count-3)))
                        break;
            self.module.update_metric(self.val_metric, batch.label)
            pred = self.module.get_outputs()[0].asnumpy().argmax(axis=1)
            data = batch.data[0].asnumpy()
            labels = batch.label[0].asnumpy()
            corr += np.sum((pred == labels) & (labels > 0))
            total += np.sum(labels > 0)
            for b in range(self.val_loader.batch_size):
                valid_ids = np.where(labels[b] > 0)[0]
                if valid_ids.size > 0:
                    valid_pred = pred[b, valid_ids]
                    valid_labels = labels[b, valid_ids]
                    valid_data = data[b, valid_ids]
                    stacked_label = np.hstack((valid_pred[:,None], valid_labels[:,None]))
                    _, uvlabel_02 = point_cloud_label_to_surface_voxel_label(valid_data, stacked_label, res=0.02)
                    _, uvlabel_0484 = point_cloud_label_to_surface_voxel_label(valid_data, stacked_label, res=0.0484)
                    corr_vox_02 += np.sum(uvlabel_02[:,0] == uvlabel_02[:,1])
                    corr_vox_0484 += np.sum(uvlabel_0484[:,0] == uvlabel_0484[:,1])
                    total_vox_02 += uvlabel_02.shape[0]
                    total_vox_0484 += uvlabel_0484.shape[0]
                    for l in range(21):
                        total_seen_class_vox[l] += np.sum(uvlabel_02[:, 0] == l)
                        total_correct_class_vox[l] += np.sum(np.bitwise_and((uvlabel_02[:, 0] == l),uvlabel_02[:, 1] == l))
        if configs['profiling']:
            
            profiler.set_state('stop')
            print(profiler.dumps())
            profiler.dump()
        caliweights = np.array(
            [0.388, 0.357, 0.038, 0.033, 0.017, 0.02, 0.016, 0.025, 0.002, 0.002, 0.002, 0.007, 0.006, 0.022, 0.004,
             0.0004, 0.003, 0.002, 0.024, 0.029])
        caliacc = np.average(
            np.array(total_correct_class_vox[1:]) / (np.array(total_seen_class_vox[1:], dtype=np.float) + 1e-6),
            weights=caliweights)
        print ('Epoch %d, Val (point: %s  0.02 voxel: %s,  0.0484 voxel: %s, calibrated voxel: %s)' % \
              (epoch, corr / total, corr_vox_02 / total_vox_02, corr_vox_0484 / total_vox_0484, caliacc))
        print (self.val_metric.get())
        # print("count:{}, count-10: {}, all inference time is:{}, average inference time is: {}s".format(count, count-10, all, all/(count-10)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    solver = ScanNetSolver()
    solver.evl_only()



