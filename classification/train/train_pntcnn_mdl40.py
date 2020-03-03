import mxnet as mx
import numpy as np
import logging
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from configs.configs import configs
from classification.models import get_symbol_cls
from data_loader.modelnet40_loader import ModelNet40Loader
from train.base_solver import BaseSolver
from utils import metrics

class ModelNet40Solver(BaseSolver):
    def __init__(self):
        super(ModelNet40Solver, self).__init__()

    def _specify_input_names(self):
        self.data_names = ['data']
        self.label_names = ['label']

    def _get_symbol(self):
        self.symbol = get_symbol_cls(self.batch_size/self.num_devices, self.num_points, bn_decay=self.bn_decay, weights=self.weights)

    def _get_data_loaders(self):
        self.train_loader = ModelNet40Loader(
                root=configs['data_dir'],
                batch_size=self.batch_size,
                npoints=self.num_points,
                normal_channel=configs['use_normal'],
                split='train',
                normalize=False,
                augment_level=1,
                shuffle=True,
                balance=False,
                dropout_ratio=configs['input_dropout_ratio'],
                tile_2d=128,
        )
        self.val_loader = ModelNet40Loader(
                root=configs['data_dir'],
                batch_size=self.batch_size,
                npoints=self.num_points,
                normal_channel=configs['use_normal'],
                split='test',
                normalize=False,
                dropout_ratio=0,
                tile_2d=128,
        )

    def _get_metric(self):
        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.CrossEntropyWithIgnore(ndim=3, axis=1)])

    def evaluate(self, epoch):
        """ evaluate one epoch. Can be overridden """
        self.val_loader.reset()
        self.metric.reset()
        corr, total = 0., 0.
        for batch in self.val_loader:
            self.module.forward(batch, is_train=False)
            self.module.update_metric(self.metric, batch.label)
            logits = self.module.get_outputs()[0].asnumpy()
            pred = logits.sum(axis=2).argmax(axis=1)
            labels = batch.label[0].asnumpy()[:,0].astype(int)
            print(pred)
            corr += np.sum(pred == labels)
            total += labels.size
        print('Per shape accuracy:', corr / total)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    solver = ModelNet40Solver()
    solver.train()



