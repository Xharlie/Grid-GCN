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
from models.ggcn_models_cnt import get_symbol_cls_ggcn
# from data_loader.ggcn_gpu_modelnet40_loader import ModelNet40Loader as loader
from base_solver import BaseSolver
from configs.configs import configs
from utils import metrics

if configs["new"]:
    from data_loader.new_ggcn_gpu_modelnet_loader import ModelNet40Loader as loader
else:
    from data_loader.ggcn_gpu_modelnet_loader import ModelNet40Loader as loader



class ModelNet40Solver(BaseSolver):
    def __init__(self, configs):
        super(ModelNet40Solver, self).__init__(configs)

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
            dataset = "40"
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
            dataset = "40"
        )


    def _get_metric(self):
        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(), mx.metric.CrossEntropy()])
        self.val_metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(), mx.metric.CrossEntropy()])

        CLASS_NAMES = {0:'airplane',1:'bathtub',2:'bed',3:'bench',4:'bookshelf',5:'bottle',6:'bowl',7:'car',8:'chair',9:'cone',10:'cup',11:'curtain',12:'desk',13:'door',14:'dresser',15:'flower_pot',16:'glass_box',17:'guitar',18:'keyboard',19:'lamp',20:'laptop',21:'mantel',22:'monitor',23:'night_stand',24:'person',25:'piano',26:'plant',27:'radio',28:'range_hood',29:'sink',30:'sofa',31:'stairs',32:'stool',33:'table',34:'tent',35:'toilet',36:'tv_stand',37:'vase',38:'wardrobe',39:'xbox'}

        self.per_class_accuracy = metrics.PerClassAccuracy(num_class=40, axis=1, class_dict=CLASS_NAMES,
                                                           report_occurrence=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    solver = ModelNet40Solver(configs)
    solver.train()




