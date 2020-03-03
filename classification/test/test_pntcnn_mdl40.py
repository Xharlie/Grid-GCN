"""
Test a trained PointCNN model on ModelNet40 test set
Need to change settings in configs/configs.yaml
"""

import mxnet as mx
import numpy as np

from segmentation.configs.configs import configs
from data_loader.modelnet40_loader import ModelNet40Loader
from classification.models import get_symbol_cls
from utils.utils import Timer

if __name__ == "__main__":
    model_prefix = configs['load_model_prefix']
    model_epoch = configs['load_model_epoch']
    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, model_epoch)
    symbol = get_symbol_cls(configs['batch_size'], configs['num_points'])
    if configs['use_cpu']:
        ctx = mx.cpu()
    else:
        ctx = [mx.gpu(int(i)) for i in configs['gpus'].split(',')]
    val_loader = ModelNet40Loader(
            root=configs['data_dir'],
            batch_size=configs['batch_size'],
            npoints=configs['num_points'],
            normal_channel=configs['use_normal'],
            split='test',
            include_trailing=True,
            normalize=False,
            dropout_ratio=0,
            tile_2d=128,
    )
    module = mx.mod.Module(symbol, context=ctx, data_names=['data'], label_names=['label'])
    module.bind(data_shapes=val_loader.provide_data, label_shapes=val_loader.provide_label)
    module.set_params(arg_params, aux_params)
    timer = Timer()
    val_loader.reset()
    corr, total = 0., 0.
    for i, batch in enumerate(val_loader):
        timer.tic()
        module.forward(batch, is_train=False)
        logits = module.get_outputs()[0].asnumpy()
        timer.toc()
        pred = logits.sum(axis=2).argmax(axis=1)
        labels = batch.label[0].asnumpy()[:,0].astype(int)
        begin, end = i*val_loader.batch_size, (i+1)*val_loader.batch_size
        if end > val_loader.num_samples:
            pred = pred[:val_loader.trailing_count]
            labels = labels[:val_loader.trailing_count]
        corr += np.sum(pred == labels)
        total += labels.size
    print 'inference time = {} s/shape'.format(timer.get() / val_loader.batch_size)
    acc = corr * 1. / total
    print 'accuracy:', acc


