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
from data_loader.shapenetseg_loader import ShapeNetPart
from train.base_solver import BaseSolver
from utils.utils import point_cloud_label_to_surface_voxel_label
from utils import metrics
from configs.configs import configs
import time
# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
SEG_NUM_CLS=50
class ShapenetSegSolver(BaseSolver):
    def __init__(self):
        super(ShapenetSegSolver, self).__init__()

    # def _specify_input_names(self):
    #     self.data_names = ['data']
    #     self.label_names = ['label']

    def _specify_input_names(self):
        self.data_names = ['data', "actual_centnum", "cls"]
        self.label_names = ['label']

    def _get_data_loaders(self):
        self.train_loader = ShapeNetPart(
                root="../data/shapenetcore_partanno_segmentation_benchmark_v0_normal",
                batch_size=self.batch_size,
                num_points=self.num_points,
                split='trainval',
                normalize=configs["normalize"],
                augment_data=configs["aug_lv"],
                dropout_ratio=configs['input_dropout_ratio'],
                shuffle=True
        )
        self.val_loader = ShapeNetPart(
            root="../data/shapenetcore_partanno_segmentation_benchmark_v0_normal", batch_size=configs['batch_size'],
            num_points=configs['num_points'], split='test',
            normalize=configs["normalize"], 
            use_cache=True,
            cache_file='shapenet_test_cache_{}_{}_py3.pickle'.format(configs["num_points"], configs["batch_size"]) if sys.version_info[0] == 3
            else 'shapenet_test__cache_{}_{}.pickle'.format(configs["num_points"],configs["batch_size"])
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
        self.seg_classes = self.val_loader.seg_classes  
        self.seg_label_to_cat = {}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat

        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.CrossEntropyWithIgnore(ndim=3, axis=1)])
        self.val_metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.CrossEntropyWithIgnore(ndim=3, axis=1), metrics.PerClassAccuracy(num_class=SEG_NUM_CLS, axis=1, ignore_label=None)])
        CLASS_NAMES = {0: 'Airplane', 1: 'Bag', 2: 'Cap', 3: 'Car', 4: 'Chair', 5: 'Earphone', 6: 'Guitar', 7: 'Knife', 8: 'Lamp', 9: 'Laptop', 10: 'Motorbike', 11: 'Mug', 12: 'Pistol', 13: 'Rocket', 14: 'Skateboard', 15: 'Table'}


    def evaluate(self, epoch):
        """
        When evaluating, convert per-point accuracy to per-voxel accuracy
        """
        besthappend=False
        self.val_loader.reset()
        corr, total = 0., 0.
        self.val_metric.reset()
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}
        for i, batchNcls in enumerate(self.val_loader):
            batch, batch_cls = batchNcls
            bsize= batch_cls.shape[0]
            pred_val = np.zeros((bsize, configs["num_points"]))
            self.module.forward(batch, is_train=False)
            self.module.update_metric(self.val_metric, batch.label)
            # self.module.update_metric(self.confusion_matrix, batch.label)
            pred = self.module.get_outputs()[0].asnumpy()
            labels = batch.label[0].asnumpy()
            for b in range(bsize):
                cat = self.seg_label_to_cat[labels[b, 0]]
                logits = pred[b, :, :]   # (num_points, num_classes)
                pred_val[b, :] = logits[self.seg_classes[cat],:].argmax(axis=0) + self.seg_classes[cat][0]

            for b in range(bsize):
                segp = pred_val[b, :]
                segl = labels[b, :]
                cat = self.seg_label_to_cat[segl[0]]
                part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
                for l in self.seg_classes[cat]:
                    if np.sum((segl == l) | (segp == l)) == 0:
                        # part is not present in this shape
                        part_ious[l - self.seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - self.seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                shape_ious[cat].append(np.mean(part_ious))
        
        instance_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                instance_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_class_ious = np.mean(list(shape_ious.values()))
        mean_part_ious = np.mean(instance_ious)
        for cat in sorted(shape_ious.keys()):
            print('****** %s: %0.6f'%(cat, shape_ious[cat]))
        print('************ Class_mIoU: %0.6f' % (mean_class_ious))
        print('************ Part_mIoU: %0.6f' % (mean_part_ious))

        if self.best_mpiou < mean_part_ious:
            self.best_mpiou = mean_part_ious
            print("new best mpiou:", self.best_mpiou)
            besthappend=True

        if self.best_mciou < mean_class_ious:
            self.best_mciou = mean_class_ious
            print("new best iou_avg:", self.best_mciou)
            besthappend=True
        return besthappend

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    solver = ShapenetSegSolver()
    solver.train()



