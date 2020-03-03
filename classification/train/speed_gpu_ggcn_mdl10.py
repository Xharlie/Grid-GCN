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
from models.ggcn_models_g_10 import get_symbol_cls_ggcn
from base_solver import BaseSolver
from configs.configs_10 import configs
from utils import metrics
import time
if configs["new"]:
    from data_loader.new_ggcn_gpu_modelnet_loader import ModelNet40Loader as loader
else:
    from data_loader.ggcn_gpu_modelnet_loader import ModelNet40Loader as loader



class ModelNet10Solver(BaseSolver):
    def __init__(self, configs):
        super(ModelNet10Solver, self).__init__(configs)

    def _specify_input_names(self):

        self.data_names = ['data', "actual_centnum"]
        self.label_names = ['label']


    def _get_symbol(self):
        take_shapes = []
        for i in range(len(configs['voxel_size_lst'])):
            take_shapes.append([self.batch_size, configs['max_o_grid_lst'][i],
                configs['max_p_grid_lst'][i],  configs['num_points'] if i == 0 else configs['max_o_grid_lst'][i-1],
                configs['inputDim'][i]])
            # print("take_shapes {}/{}".format(i, take_shapes[i]))
        if configs['task'] == 'cls':
            self.symbol = get_symbol_cls_ggcn(self.batch_size//self.num_devices, self.num_points,
                take_shapes, gcn_outDim=configs['gcn_outDim'], bn_decay=self.bn_decay, weights=self.weights)
        else:
            raise NotImplementedError("Task not identified")

    def _get_data_loaders(self):
        # self.train_loader = loader(
        #     root=configs['data_dir'],
        #     batch_size=self.batch_size,
        #     npoints=self.num_points,
        #     normal_channel=configs['use_normal'],
        #     split='train',
        #     augment_level=configs['augment_level'],
        #     shuffle=True,
        #     balance=False,
        #     dropout_ratio=configs['input_dropout_ratio'],
        #     voxel_size_lst=configs['voxel_size_lst'], grid_size_lst=configs['grid_size_lst'],
        #     lidar_coord=configs['lidar_coord'], max_p_grid_lst=configs['max_p_grid_lst'],
        #     max_o_grid_lst=configs['max_o_grid_lst'], kernel_size_lst=configs['kernel_size_lst'],
        #     stride_lst=configs['stride_lst'], single_padding_lst=configs['single_padding_lst'],
        #     reverse_index=configs['reverse_index'],
        #     point_set = configs["point_set"]
        # )
        # self.val_loader = loader(
        #     root=configs['data_dir'],
        #     batch_size=self.batch_size,
        #     npoints=self.num_points,
        #     normal_channel=configs['use_normal'],
        #     split='test',
        #     dropout_ratio=0,
        #     voxel_size_lst=configs['voxel_size_lst'], grid_size_lst=configs['grid_size_lst'],
        #     lidar_coord=configs['lidar_coord'], max_p_grid_lst=configs['max_p_grid_lst'],
        #     max_o_grid_lst=configs['max_o_grid_lst'], kernel_size_lst=configs['kernel_size_lst'],
        #     stride_lst=configs['stride_lst'], single_padding_lst=configs['single_padding_lst'],
        #     reverse_index=configs['reverse_index'],
        #     point_set="partial"
        # )

        self.train_loader = loader(
            root=configs['data_dir'],
            configs=configs,
            batch_size=self.batch_size,
            npoints=self.num_points,
            normal_channel=configs['use_normal'],
            split='train',
            normalize=configs["normalize"],
            augment_level=configs['augment_level'],
            shuffle=True,
            balance=False,
            dropout_ratio=configs['input_dropout_ratio'],
            point_set=configs["point_set"],
            dataset = "10"
        )
        self.val_loader = loader(
            root=configs['data_dir'],
            configs=configs,
            batch_size=self.batch_size,
            npoints=self.num_points,
            normal_channel=configs['use_normal'],
            split='test',
            normalize=configs["normalize"],
            shuffle=True,
            balance=False,
            dropout_ratio=0,
            point_set=configs["point_set"],
            dataset = "10"
        )


    def _get_metric(self):
        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(), mx.metric.CrossEntropy()])
        self.val_metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(), mx.metric.CrossEntropy()])

        CLASS_NAMES = {0:'bathtub',1:'bed',2:'chair',3:'desk',4:'dresser',5:'monitor',6:'night_stand',7:'sofa',8:'table',9:'toilet'}
        self.per_class_accuracy = metrics.PerClassAccuracy(num_class=10, axis=1, class_dict=CLASS_NAMES,
                                                           report_occurrence=True)

    def evaluate(self, epoch):
        """
        When evaluating, convert per-point accuracy to per-voxel accuracy
        """
        
        # os.environ["MXNET_EXEC_BULK_EXEC_INFERENCE"] = "0"
        # profiler.set_state('run')
        sum_time=0
        count_infe=0
        while True:
            self.val_loader.reset()
            for i, batch in enumerate(self.val_loader):
                print("number batch,", i)
                tic = time.time()
                self.module.forward(batch, is_train=False)
                mx.nd.waitall()
                if i > 10:
                    sum_time += time.time() - tic
                    count_infe+=1
                    if count_infe > 300:
                        print("avg inference time:", sum_time / count_infe)
                        exit()
                # if i >= 300:
                #     profiler.set_state('stop')
                #     profiler.dump()
                #     exit()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    solver = ModelNet10Solver(configs)
    solver.train(evlonly=True)





