import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
import ctypes
_ = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../gridifyop/additional.so'))
# _ = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../gridifyop/additional.so'))
import mxnet as mx
from models.ggcn_models_g import get_symbol_seg_ggcn
from data_loader.ggcn_gpu_scannet_loader import ScanNetLoader, ScanNetWholeSceneLoader
from train.base_solver import BaseSolver
from utils.utils import point_cloud_label_to_surface_voxel_label
from utils import metrics
from configs.configs import configs
import mxnet.profiler as profiler
import time
# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
# os.environ["MXNET_EXEC_BULK_EXEC_INFERENCE"] = "0"

    
# profiler.set_config(profile_all=True,
#                     aggregate_stats=True,
#                     filename='ggcn_{}.json'.format(configs["pf_name"]))

class ScanNetSolver(BaseSolver):
    def __init__(self):
        super(ScanNetSolver, self).__init__()

    # def _specify_input_names(self):
    #     self.data_names = ['data']
    #     self.label_names = ['label']

    def _specify_input_names(self, layer_num=None):
        if layer_num is None:
            layer_num = len(configs['voxel_size_lst'])

        self.data_names = ['data', "actual_centnum"]
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
                evel_only=True
        )
        self.val_loader = ScanNetWholeSceneLoader(
            root=configs['data_dir'], batch_size=configs['batch_size'],
            npoints=configs['num_points'], split='test',
            normalize=True, use_cache=False
        )
        print("get dataloader!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
        self.confusion_matrix = metrics.ConfusionMatrix(num_class=21, axis=1, ignore_label=0)

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
        sum_time = 0
        count_infe = 0
        # profiler.set_state('run')
        for i, batch in enumerate(self.val_loader):
            print("number batch,", i)
            tic = time.time()
            self.module.forward(batch, is_train=False)
            if i > 3:
                mx.nd.waitall() 
                sum_time += time.time() - tic
                count_infe+=1
            if i >= 33:
                mx.nd.waitall()
                # profiler.set_state('stop')
                # print(profiler.dump())
                print("avg inference time:", sum_time / count_infe) 
                break
        

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    solver = ScanNetSolver()

    solver.evl_only()



