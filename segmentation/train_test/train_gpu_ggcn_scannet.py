import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import ctypes
_ = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../gridifyop/additional.so'))
print(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../gridifyop/additional.so'))

import mxnet as mx

from models.ggcn_models_g import get_symbol_seg_ggcn
from data_loader.ggcn_gpu_scannet_loader import ScanNetLoader, ScanNetWholeSceneLoader
from base_solver import BaseSolver
from utils.utils import point_cloud_label_to_surface_voxel_label
from utils import metrics
from configs.configs import configs
import mxnet.profiler as profiler
import time
# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

class ScanNetSolver(BaseSolver):
    def __init__(self):
        super(ScanNetSolver, self).__init__()

    # def _specify_input_names(self):
    #     self.data_names = ['data']
    #     self.label_names = ['label']

    def _specify_input_names(self):
        self.data_names = ['data', "actual_centnum"]
        self.label_names = ['label']

    def _get_data_loaders(self):
        self.train_loader = ScanNetLoader(
                root=configs['data_dir'],
                batch_size=self.batch_size,
                npoints=self.num_points,
                split='train',
                normalize=configs["normalize"],
                augment_data=configs["aug_lv"],
                dropout_ratio=configs['input_dropout_ratio'],
                shuffle=True,
        )
        self.val_loader = ScanNetWholeSceneLoader(
            root=configs['data_dir'], batch_size=configs['batch_size'],
            npoints=configs['num_points'], split='test',
            normalize=configs["normalize"], use_cache=True,
            cache_file='ggcn_gpu_whole_scene_cache_{}_{}_py3.pickle'.format(configs["num_points"], configs["batch_size"]) if sys.version_info[0] == 3
            else 'ggcn_gpu_whole_scene_cache_{}_{}.pickle'.format(configs["num_points"],configs["batch_size"])
        )
    def _get_symbol(self):
        take_shapes = []
        take_up_shapes = []
        for i in range(len(configs['voxel_size_lst'])):
            take_shapes.append([self.batch_size, configs['max_o_grid_lst'][i],
                                configs['max_p_grid_lst'][i],
                                configs['num_points'] if i == 0 else configs['max_o_grid_lst'][i - 1],
                                configs['inputDim'][i]])
        for i in range(len(configs['up_voxel_size_lst'])):
            take_up_shapes.append([self.batch_size, configs['up_max_o_grid_lst'][i],
                                configs['up_max_p_grid_lst'][i],
                                # configs['max_o_grid_lst'][-1] if i == 0 else configs['up_max_o_grid_lst'][i - 1],
                                configs['max_o_grid_lst'][-i - 1],
                                configs['up_inputDim'][i]])
        print("take_up_shapes",take_up_shapes)
        self.symbol = get_symbol_seg_ggcn(self.batch_size // self.num_devices, self.num_points,
                take_shapes, take_up_shapes, bn_decay=self.bn_decay, weights=self.weights)


    def _get_metric(self):
        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.AccuracyWithIgnore(axis=1, ignore_label=0),
            metrics.CrossEntropyWithIgnore(ndim=3, axis=1, ignore_label=0, label_weights=self.weights)])
        self.val_metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.AccuracyWithIgnore(axis=1, ignore_label=0),
            metrics.CrossEntropyWithIgnore(ndim=3, axis=1, ignore_label=0, label_weights=None), metrics.PerClassAccuracy(num_class=21, axis=1, ignore_label=0)])
        CLASS_NAMES = {1: 'wall', 2: 'floor', 3: 'chair', 4: 'table', 5: 'desk', 6: 'bed', 7: 'bookshelf', 8: 'sofa', 9: 'sink', 10: 'bathtub', 11: 'toilet', 12: 'curtain', 13: 'counter', 14: 'door', 15: 'window', 16: 'shower_curtain', 17: 'refrigerator', 18: 'picture', 19: 'cabinet', 20: 'other'}
        # CLASS_NAMES_SHORT = {1: 'wall', 2: 'floor', 3: 'chair', 4: 'table', 5: 'desk', 6: 'bed', 7: 'shelf', 8: 'sofa',
        #                      9: 'sink', 10: 'bathtub', 11: 'toilet', 12: 'curtain', 13: 'counter', 14: 'door',
        #                      15: 'window', 16: 'shwcur', 17: 'fridge', 18: 'picture', 19: 'cabinet', 20: 'other'}
        self.per_class_accuracy = metrics.PerClassAccuracy(num_class=21, axis=1, ignore_label=0, class_dict=CLASS_NAMES, report_occurrence=True)
        self.per_class_iou = metrics.PerClassIoU(num_class=21, axis=1, ignore_label=0, class_dict=CLASS_NAMES, report_occurrence=True)
        # self.confusion_matrix = metrics.ConfusionMatrix(num_class=21, axis=1, ignore_label=0)

    def evaluate(self, epoch):
        """
        When evaluating, convert per-point accuracy to per-voxel accuracy
        """
        besthappend=False
        self.val_loader.reset()
        corr, total = 0., 0.
        corr_vox_02, total_vox_02 = 0., 0.
        corr_vox_0484, total_vox_0484 = 0., 0.
        self.val_metric.reset()
        self.per_class_accuracy.reset()
        self.per_class_iou.reset()
        total_seen_class_vox = [0 for _ in range(21)]
        total_correct_class_vox = [0 for _ in range(21)]
        for i, batch in enumerate(self.val_loader):
            self.module.forward(batch, is_train=False)
            self.module.update_metric(self.val_metric, batch.label)
            self.module.update_metric(self.per_class_accuracy, batch.label)
            self.module.update_metric(self.per_class_iou, batch.label)
            # self.module.update_metric(self.confusion_matrix, batch.label)
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
        caliweights = np.array(
            [0.388, 0.357, 0.038, 0.033, 0.017, 0.02, 0.016, 0.025, 0.002, 0.002, 0.002, 0.007, 0.006, 0.022, 0.004,
             0.0004, 0.003, 0.002, 0.024, 0.029])
        caliacc = np.average(
            np.array(total_correct_class_vox[1:]) / (np.array(total_seen_class_vox[1:], dtype=np.float) + 1e-6),
            weights=caliweights)
        print('Epoch %d, Val (point: %s  0.02 voxel: %s,  0.0484 voxel: %s, calibrated voxel: %s)' % \
              (epoch, corr / total, corr_vox_02 / total_vox_02, corr_vox_0484 / total_vox_0484, caliacc))
        print('Per Class Accuracy/IoU:')
        acc = self.per_class_accuracy.get()[1]
        iou = self.per_class_iou.get()[1]
        iou_sum = 0
        iou_cnt = 0
        for (label, acc_, count), (_, iou_, _) in zip(acc, iou):
            print('{:^15s}{:10.5f}{:10.5f}{:9d}'.format(label, acc_, iou_, count))
            iou_sum += iou_
            iou_cnt+=1
        iou_avg = iou_sum / iou_cnt
        print("iouavg: ",iou_avg)
        if self.best_val_acc < caliacc:
            self.best_val_acc = caliacc
            print("new best val cali acc:", self.best_val_acc)
            besthappend=True

        if self.best_mciou < iou_avg:
            self.best_mciou = iou_avg
            print("new best iou_avg:", self.best_mciou)
            besthappend=True
        return besthappend

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    solver = ScanNetSolver()
    solver.train()



