import mxnet as mx
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

from configs.configs import configs
from classification.models import get_symbol_cls_ssg, get_symbol_cls_msg
from data_loader.modelnet40_loader import ModelNet40Loader
from train.base_solver import BaseSolver

class ModelNet40Solver(BaseSolver):
    def __init__(self):
        super(ModelNet40Solver, self).__init__()

    def _specify_input_names(self):
        self.data_names = ['data']
        self.label_names = ['label']

    def _get_symbol(self):
        if configs['task'] == 'cls_ssg':
            self.symbol = get_symbol_cls_ssg(self.batch_size/self.num_devices, self.num_points, bn_decay=self.bn_decay, weights=self.weights)
        elif configs['task'] == 'cls_msg':
            self.symbol = get_symbol_cls_msg(self.batch_size/self.num_devices, self.num_points, bn_decay=self.bn_decay, weights=self.weights)
        else:
            raise NotImplementedError("Task not identified")

    def _get_data_loaders(self):
        self.train_loader = ModelNet40Loader(
                root=configs['data_dir'],
                batch_size=self.batch_size,
                npoints=self.num_points,
                normal_channel=configs['use_normal'],
                split='train',
                augment_level=2,
                shuffle=True,
                balance=False,
                dropout_ratio=configs['input_dropout_ratio'],
        )
        self.val_loader = ModelNet40Loader(
                root=configs['data_dir'],
                batch_size=self.batch_size,
                npoints=self.num_points,
                normal_channel=configs['use_normal'],
                split='test',
                dropout_ratio=0,
        )

    def _get_metric(self):
        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(), mx.metric.CrossEntropy()])


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    solver = ModelNet40Solver()
    solver.train()



