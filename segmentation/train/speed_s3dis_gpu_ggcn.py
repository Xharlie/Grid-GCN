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
                drop_factor=configs['input_dropout_ratio']
        )
        self.val_loader = S3DISLoader(
            '../data/3DIS/prepare_label_rgb_{}/val_files_Area_{}.txt'.format(scale, area),
            batch_size=configs['batch_size'],
            npoints=configs['num_points'], split='test',
            normalize=configs["normalize"], use_cache=True, max_epoch=configs["num_epochs"],
            drop_factor=configs['input_dropout_ratio']
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
                                configs['up_max_p_grid_lst'][i] + 3 if configs["multi"] else configs['up_max_p_grid_lst'][i],
                                # configs['max_o_grid_lst'][-1] if i == 0 else configs['up_max_o_grid_lst'][i - 1],
                                configs['max_o_grid_lst'][-i - 1],
                                configs['up_inputDim'][i]])
        print("take_up_shapes",take_up_shapes)
        self.symbol = get_symbol_seg_ggcn(self.batch_size // self.num_devices, self.num_points,
                take_shapes, take_up_shapes, bn_decay=self.bn_decay, weights=self.weights)


    def _get_metric(self):
        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.AccuracyWithIgnore(axis=1, ignore_label=13),
            metrics.CrossEntropyWithIgnore(ndim=3, axis=1, ignore_label=13, label_weights=self.weights)])
        self.val_metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.AccuracyWithIgnore(axis=1),
            metrics.CrossEntropyWithIgnore(ndim=3, axis=1, label_weights=None), metrics.PerClassAccuracy(num_class=14, axis=1, ignore_label=13)])
        CLASS_NAMES = {0:'ceiling', 1:'floor', 2:'wall', 3:'beam', 4:'column', 5:'window',
    6:'door', 7:'table', 8:'chair', 9:'sofa', 10:'bookcase', 11:'board', 12:'clutter'}
        self.per_class_accuracy = metrics.PerClassAccuracy(num_class=14, axis=1, class_dict=CLASS_NAMES, report_occurrence=True, ignore_label=13)
        self.per_class_iou = metrics.PerClassIoU(num_class=14, axis=1, class_dict=CLASS_NAMES, report_occurrence=True, ignore_label=13)


    def evaluate(self, epoch):
        """
        When evaluating, convert per-point accuracy to per-voxel accuracy
        """
        self.val_loader.start() 
        self.val_loader.reset()
        corr, total = 0., 0.
        corr_vox_02, total_vox_02 = 0., 0.
        corr_vox_0484, total_vox_0484 = 0., 0.
        self.val_metric.reset()
        total_seen_class_vox = [0 for _ in range(21)]
        total_correct_class_vox = [0 for _ in range(21)]
        sum_time = 0
        count_infe = 0
        # os.environ["MXNET_EXEC_BULK_EXEC_INFERENCE"] = "0"
        # profiler.set_state('run')
        fetched_epoch = self.val_epoch
        i = 0
        while fetched_epoch == self.val_epoch:
            batch, fetched_epoch, batch_counter = self.val_loader.fetch()
            print("number batch,", i)   
            tic = time.time()   
            self.module.forward(batch, is_train=False)
            mx.nd.waitall()
            if i > 5:
                sum_time += time.time() - tic
                print(i)
                count_infe+=1
                if i > 500:
                    break;
            i+=1    
        print("avg inference time:", sum_time / count_infe)
        self.val_loader.shutdown()
        exit()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    solver = S3disSolver()
    solver.evl_only()



