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
import utils
from models.s3dis_ggcn_models_g import get_symbol_seg_ggcn
from data_loader.s3dis_gpu_loader import S3DISLoader
from train.s3dis_base_solver import BaseSolver
from utils.utils import point_cloud_label_to_surface_voxel_label
from utils import metrics
from s3dis_configs.configs import configs
import mxnet.profiler as profiler
import time
# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

class S3disSolver(BaseSolver):
    def __init__(self):
        super(S3disSolver, self).__init__()

    # def _specify_input_names(self):
    #     self.data_names = ['data']
    #     self.label_names = ['label']

    def _specify_input_names(self):
        self.data_names = ['dataxyz', "datafeat", "actual_centnum"]
        self.label_names = ['label']

    def _get_data_loaders(self):
        area = configs["area"]
        scale = configs["scale"]
        self.train_loader = S3DISLoader(
                '../data/3DIS/prepare_label_rgb_{}/train_files_for_val_on_Area_{}.txt'.format(scale, area),
                batch_size=self.batch_size,
                npoints=self.num_points,
                normalize=configs["normalize"],
                augment_data=configs["aug_lv"],
                shuffle=True,
                max_epoch=configs["num_epochs"],
                drop_factor=configs['input_dropout_ratio'],
                area=area
        )
        self.val_loader = S3DISLoader(
            '../data/3DIS/prepare_label_rgb_{}/val_files_Area_{}.txt'.format(scale, area),
            batch_size=configs['batch_size'],
            npoints=configs['num_points'], split='test',
            normalize=configs["normalize"], 
            use_cache=True, max_epoch=configs["num_epochs"],
            drop_factor=configs['input_dropout_ratio'],
            area=area
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
                                configs['up_max_p_grid_lst'][i] + configs['up_max_p_grid_lstknn'][i] if configs["multi"] else configs['up_max_p_grid_lst'][i],
                                # configs['max_o_grid_lst'][-1] if i == 0 else configs['up_max_o_grid_lst'][i - 1],
                                configs['max_o_grid_lst'][-i - 1],
                                configs['up_inputDim'][i]])
        print("take_up_shapes",take_up_shapes)
        self.symbol = get_symbol_seg_ggcn(self.batch_size // self.num_devices, self.num_points,
                take_shapes, take_up_shapes, bn_decay=self.bn_decay, weights=self.weights)


    def _get_metric(self):
        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.AccuracyWithIgnore(axis=1, ignore_label=13),
            metrics.CrossEntropyWithIgnore(ndim=3, axis=1, ignore_label=13, label_weights=None)])
        self.val_metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.AccuracyWithIgnore(axis=1),
            metrics.CrossEntropyWithIgnore(ndim=3, axis=1, label_weights=None), metrics.PerClassAccuracy(num_class=14, axis=1, ignore_label=13)])
        CLASS_NAMES = {0:'ceiling', 1:'floor', 2:'wall', 3:'beam', 4:'column', 5:'window',
    6:'door', 7:'chair', 8:'table', 9:'bookcase', 10:'sofa', 11:'board', 12:'clutter'}
        self.per_class_accuracy = metrics.PerClassAccuracy(num_class=14, axis=1, class_dict=CLASS_NAMES, report_occurrence=True, ignore_label=13)
        self.per_class_iou = metrics.PerClassIoU(num_class=14, axis=1, class_dict=CLASS_NAMES, report_occurrence=True, ignore_label=13)


    def evaluate(self):
        """
        When evaluating, convert per-point accuracy to per-voxel accuracy
        """
        self.val_loader.reset()
        corr, total = 0., 0.
        self.val_metric.reset()
        self.per_class_accuracy.reset()
        self.per_class_iou.reset()
        fetched_epoch = self.val_epoch
        besthappend = False
        while fetched_epoch == self.val_epoch:
            batch, fetched_epoch, batch_counter = self.val_loader.fetch()
            self.module.forward(batch, is_train=False)
            self.module.update_metric(self.val_metric, batch.label)
            self.module.update_metric(self.per_class_accuracy, batch.label)
            self.module.update_metric(self.per_class_iou, batch.label)
            # self.module.update_metric(self.confusion_matrix, batch.label)
            pred = self.module.get_outputs()[0].asnumpy().argmax(axis=1)
            # data = batch.data[0].asnumpy()
            labels = batch.label[0].asnumpy()
            corr += np.sum((pred == labels) & (labels >= 0))
            total += np.sum(labels >= 0)
        print("val total batch_counter: ", batch_counter)
        oa =  corr / total
        print('Per Class Accuracy/IoU:')
        acc = self.per_class_accuracy.get()[1]
        iou = self.per_class_iou.get()[1]
        iou_sum = 0
        acc_sum = 0
        class_cnt = 0
        for (label, acc_, count), (_, iou_, _) in zip(acc, iou):
            print('{:^15s}{:10.5f}{:10.5f}{:9d}'.format(label, acc_, iou_, count))
            iou_sum += iou_
            acc_sum += acc_
            class_cnt+=1
        iou_avg = iou_sum / class_cnt
        acc_class_avg = acc_sum / class_cnt
        print('Epoch %d, Val (point: %s ; iou_avg: %s ; macc: %s)' % (self.val_epoch, oa, iou_avg, acc_class_avg))
        if self.best_val_acc < oa:
            self.best_val_acc = oa
            print("new best val overall acc:", self.best_val_acc)
            besthappend = True

        if self.best_mciou < iou_avg:
            self.best_mciou = iou_avg
            print("new best iou_avg:", self.best_mciou)
            besthappend = True

        self.val_epoch += 1
        return besthappend

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    solver = S3disSolver()
    solver.train()



