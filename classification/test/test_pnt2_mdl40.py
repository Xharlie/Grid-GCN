"""
Test a trained model on ModelNet40 test set
Need to change settings in configs/configs.yaml
"""

import mxnet as mx
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from configs.configs import configs
from data_loader.modelnet40_loader import ModelNet40Loader
from classification.models import get_symbol_cls_ssg, get_symbol_cls_msg
from utils import Timer

SHOW = False
NUM_VOTES = 1

if __name__ == "__main__":
    if SHOW:
        shape_names = {}
        # read shape names
        with open(os.path.join(configs['data_dir'], 'modelnet40_shape_names.txt')) as f:
            for i, line in enumerate(f):
                shape_names[i] = line.strip()
    model_prefix = configs['load_model_prefix']
    model_epoch = configs['load_model_epoch']
    print("loading: {} | {}".format(model_prefix, model_epoch))
    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)
    if configs['task'] == 'cls_ssg':
        symbol = get_symbol_cls_ssg(configs['batch_size'], configs['num_points'])
    else:
        symbol = get_symbol_cls_msg(configs['batch_size'], configs['num_points'])
    if configs['use_cpu']:
        ctx = mx.cpu()
    else:
        ctx = [mx.gpu(int(i)) for i in configs['gpus'].split(',')]
    val_loader = ModelNet40Loader(
            root=configs['data_dir'],
            batch_size=configs['batch_size'],
            npoints=configs['num_points'],
            split='test',
            include_trailing=True,
            dropout_ratio=0)
    # symbol = symbol.get_internals()
    # print symbol.list_outputs()

    module = mx.mod.Module(symbol, context=ctx, data_names=['data'], label_names=['label'])
    module.bind(data_shapes=val_loader.provide_data, label_shapes=val_loader.provide_label)
    # module = mx.mod.Module(symbol, context=ctx, data_names=['data'])
    # module.bind(data_shapes=val_loader.provide_data)
    module.set_params(arg_params, aux_params)
    arg_params, aux_params = module.get_params()
    # print(sorted([(k, v.shape, v[0][0]) for (k, v) in arg_params.iteritems()]))
    # print 'Total number of parameters:', sum([np.prod(v.shape) for (k, v) in arg_params.iteritems()])
    # print(sorted([(k, v.shape, v[0][0]) for (k, v) in aux_params.iteritems()]))

    # pred_score: sum over all angles
    pred_score = np.zeros((val_loader.num_samples, 40))
    timer = Timer()
    for rot in np.linspace(0, np.pi*2, NUM_VOTES, endpoint=False):
        val_loader.set_rotation_angle(rot)
        for i, batch in enumerate(val_loader):
            timer.tic()
            module.forward(batch, is_train=False)
            # print(module.get_outputs()[0].asnumpy()[0,0,...])
            # exit()
            pred = module.get_outputs()[0].asnumpy()
            timer.toc()
            begin, end = i*val_loader.batch_size, (i+1)*val_loader.batch_size
            # print("batch {}, data[0,0,...] {}".format(i, batch.data[0][0]))
            # print("pred: {}".format(pred[:val_loader.trailing_count]))
            if end > pred_score.shape[0]:
                pred_score[begin:] += pred[:val_loader.trailing_count]
            else:
                pred_score[begin:end] += pred


    print 'inference time = {} s/shape'.format(timer.get() / val_loader.batch_size)
    # get gt labels
    labels = np.zeros(pred_score.shape[0], dtype=int)
    for i in xrange(labels.size):
        labels[i] = val_loader[i][1]
    # compare pred with labels
    pred = pred_score.argmax(axis=1)
    corr = sum(labels == pred)
    acc = corr * 1. / labels.size
    print 'accuracy:', acc
    '''
    if SHOW:
        pred = module.get_outputs()[0].asnumpy().argmax(axis=1)
        label = batch.label[0].asnumpy().astype(int)
        wrong_ids = np.where(pred != label)[0]
        for idx in wrong_ids:
            data = batch.data[0].asnumpy()[idx]
            title = 'label {} predict {}'.format(shape_names[label[idx]], shape_names[pred[idx]])
            draw_point_cloud(data, title=title)
    '''


