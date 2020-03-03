"""
Test a trained model on ScanNet whole scene
Need to change settings in configs/configs.yaml
"""

import os
import sys

import mxnet as mx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))

from models.ggcn_models import get_symbol_seg_ggcn
from data_loader.ggcn_scannet_loader import ScanNetWholeSceneLoader
from configs.configs import configs
import time
from utils.utils import point_cloud_label_to_surface_voxel_label, draw_point_cloud_with_labels_compare
from utils import metrics

SHOW = False
CLASS_NAMES = {1: 'wall', 2: 'floor', 3: 'chair', 4: 'table', 5: 'desk', 6: 'bed', 7: 'bookshelf', 8: 'sofa', 9: 'sink', 10: 'bathtub', 11: 'toilet', 12: 'curtain', 13: 'counter', 14: 'door', 15: 'window', 16: 'shower_curtain', 17: 'refrigerator', 18: 'picture', 19: 'cabinet', 20: 'other'}
CLASS_NAMES_SHORT = {1: 'wall', 2: 'floor', 3: 'chair', 4: 'table', 5: 'desk', 6: 'bed', 7: 'shelf', 8: 'sofa', 9: 'sink', 10: 'bathtub', 11: 'toilet', 12: 'curtain', 13: 'counter', 14: 'door', 15: 'window', 16: 'shwcur', 17: 'fridge', 18: 'picture', 19: 'cabinet', 20: 'other'}


def get_symbol():
    take_shapes = []
    take_up_shapes = []
    for i in range(len(configs['voxel_size_lst'])):
        take_shapes.append([configs['batch_size'], configs['max_o_grid_lst'][i],
                            configs['max_p_grid_lst'][i],
                            configs['num_points'] if i == 0 else configs['max_o_grid_lst'][i - 1],
                            configs['inputDim'][i]])
    for i in range(len(configs['up_voxel_size_lst'])):
        take_up_shapes.append([configs['batch_size'], configs['up_max_o_grid_lst'][i],
                               configs['up_max_p_grid_lst'][i],
                               configs['max_o_grid_lst'][-i - 1],
                               configs['up_inputDim'][i]])
    print("take_up_shapes", take_up_shapes)
    symbol = get_symbol_seg_ggcn(configs['batch_size'] / 1, configs['num_points'],
                                      take_shapes, take_up_shapes, bn_decay=configs['bn_decay'], weights=None)
    return symbol

def specify_input_names(layer_num=None):
    if layer_num is None:
        layer_num = len(configs['voxel_size_lst'])

    data_names = ['data']
    label_names = ['label']
    for i in range(layer_num):
        if configs["agg"] in ["max", "max_pooling"] and configs["neigh_fetch"] == 0:
            if configs["up_type"] == "inter":
                data_names += ['neighbors_arr' + str(i),
                                    'centers_arr' + str(i),
                                    'centers_mask' + str(i)]
            else:
                if configs["up_agg"] in ["max", "max_pooling"] and configs["up_neigh_fetch"] == 0:
                    data_names += ['neighbors_arr' + str(i),
                                        'neighbors_arr_up' + str(i),
                                        'centers_arr' + str(i),
                                        'centers_mask' + str(i)]
                else:
                    data_names += ['neighbors_arr' + str(i),
                                'neighbors_arr_up' + str(i),
                                'neighbors_mask_up' + str(i),
                                'centers_arr' + str(i),
                                'centers_mask' + str(i)]
        else:
            if configs["up_type"] == "inter":
                data_names += ['neighbors_arr' + str(i),
                                    'neighbors_mask' + str(i),
                                    'centers_arr' + str(i),
                                    'centers_mask' + str(i)]
            else:
                if configs["up_agg"] in ["max", "max_pooling"] and configs["up_neigh_fetch"] == 0:
                    data_names += ['neighbors_arr' + str(i),
                                        'neighbors_arr_up' + str(i),
                                        'neighbors_mask' + str(i),
                                        'centers_arr' + str(i),
                                        'centers_mask' + str(i)]
                else:
                    data_names += ['neighbors_arr' + str(i),
                                        'neighbors_arr_up' + str(i),
                                        'neighbors_mask' + str(i),
                                        'neighbors_mask_up' + str(i),
                                        'centers_arr' + str(i),
                                        'centers_mask' + str(i)]
    return data_names

if __name__ == "__main__":
    model_prefix = configs['load_model_prefix']
    model_epoch = configs['load_model_epoch']
    _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)
    symbol = get_symbol()
    if configs['use_cpu']:
        ctx = mx.cpu()
    else:
        ctx = [mx.gpu(int(i)) for i in configs['gpus'].split(',')]
    val_loader = ScanNetWholeSceneLoader(
            root=configs['data_dir'], batch_size=configs['batch_size'],
            npoints=configs['num_points'], split='test',
            normalize=True, use_cache=True,
            cache_file='ggcn_whole_scene_cache_8192_{}_{}_{}_{}_{}.pickle'.format(configs["batch_size"],
                        configs["up_max_p_grid_lst"][0], configs["up_max_p_grid_lst"][1],
                        configs["up_max_p_grid_lst"][2], configs["up_max_p_grid_lst"][3]),
    )
    module = mx.mod.Module(symbol, context=ctx, data_names=specify_input_names(), label_names=['label'])
    module.bind(data_shapes=val_loader.provide_data, label_shapes=val_loader.provide_label)
    module.set_params(arg_params, aux_params)
    val_loader.reset()
    metric = mx.metric.CompositeEvalMetric([metrics.AccuracyWithIgnore(axis=1, ignore_label=0), metrics.CrossEntropyWithIgnore(ndim=3, axis=1, ignore_label=0)])
    per_class_accuracy = metrics.PerClassAccuracy(num_class=21, axis=1, ignore_label=0, class_dict=CLASS_NAMES, report_occurrence=True)
    per_class_iou = metrics.PerClassIoU(num_class=21, axis=1, ignore_label=0, class_dict=CLASS_NAMES, report_occurrence=True)
    confusion_matrix = metrics.ConfusionMatrix(num_class=21, axis=1, ignore_label=0)
    corr_vox, total_vox = 0., 0.
    corr_vox_class = np.zeros(21)
    total_vox_class = np.zeros(21)
    start = time.time()
    for i, batch in enumerate(val_loader):
        module.forward(batch, is_train=False)
        pred = module.get_outputs()[0].asnumpy()
        module.update_metric(metric, batch.label)
        module.update_metric(per_class_accuracy, batch.label)
        module.update_metric(per_class_iou, batch.label)
        module.update_metric(confusion_matrix, batch.label)
        data = batch.data[0].asnumpy()
        labels = batch.label[0].asnumpy().astype(int)
        pred = pred.argmax(axis=1)
        print metric.get()
        if SHOW:
            for i in xrange(data.shape[0]):
                data0 = data[i].copy()
                pred0 = pred[i].copy()
                labels0 = labels[i].copy()
                ids = labels0 > 0
                data0 = data0[ids]
                pred0 = pred0[ids]
                labels0 = labels0[ids]
                print (np.sum(pred0==labels0) * 1. / labels0.size)
                draw_point_cloud_with_labels_compare(data0, pred0, labels0)
        for b in xrange(val_loader.batch_size):
            valid_ids = np.where(labels[b] > 0)[0]
            if valid_ids.size > 0:
                valid_pred = pred[b, valid_ids]
                valid_labels = labels[b, valid_ids]
                valid_data = data[b, valid_ids]
                stacked_label = np.hstack((valid_pred[:,None], valid_labels[:,None]))
                _, uvlabel = point_cloud_label_to_surface_voxel_label(valid_data, stacked_label, res=0.02)
                corr_vox += np.sum(uvlabel[:,0] == uvlabel[:,1])
                total_vox += uvlabel.shape[0]
                for j in xrange(1, 21):
                    corr_vox_class[j] += np.sum((uvlabel[:,0] == j) & (uvlabel[:,1] == j))
                    total_vox_class[j] += np.sum(uvlabel[:,1] == j)
        print '\r{}'.format(i),
    print("use {}s".format(time.time()-start))
    print '*'*30
    print metric.get()
    print 'Per Voxel Accuracy = {:.5f}'.format(corr_vox / total_vox)
    cali_weights = np.array([0.388,0.357,0.038,0.033,0.017,0.02,0.016,0.025,0.002,0.002,0.002,0.007,0.006,0.022,0.004,0.0004,0.003,0.002,0.024,0.029])
    cali_acc = np.average(corr_vox_class[1:] / (total_vox_class[1:] + 1e-6), weights=cali_weights)
    print 'Calibrated Per Voxel Accuracy = {:.5f}'.format(cali_acc)
    print '*'*30
    print 'Per Class Accuracy/IoU:'
    acc = per_class_accuracy.get()[1]
    iou = per_class_iou.get()[1]
    for (label, acc_, count), (_, iou_, _) in zip(acc, iou):
        print '{:^15s}{:10.5f}{:10.5f}{:9d}'.format(label, acc_, iou_, count)
    print '*'*30
    print 'Confusion Matrix:'
    matrix = confusion_matrix.get()[1]
    print ' '*7,
    for i in xrange(21):
        print '{:>7s}'.format(CLASS_NAMES_SHORT.get(i, '')),
    print
    for i in xrange(21):
        print '{:^7s}'.format(CLASS_NAMES_SHORT.get(i, '')),
        for j in xrange(21):
            print '{:7d}'.format(matrix[i,j]),
        print


