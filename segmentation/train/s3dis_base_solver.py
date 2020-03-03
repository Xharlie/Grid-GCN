import mxnet as mx
import os
from pprint import pprint
import numpy as np

from s3dis_configs.configs import configs
import utils.utils as utils

class BaseSolver(object):
    def __init__(self):
        self.batch_size = configs['batch_size']
        self.num_points = configs['num_points']
        if configs['use_cpu']:
            self.ctx = mx.cpu()
            self.num_devices = 1
        else:
            self.ctx = [mx.gpu(int(i)) for i in configs['gpus'].split(',')]
            self.num_devices = len(self.ctx)
        self._specify_input_names()
        self.best_val_acc = 0
        self.best_mciou = 0
        self.best_macc = 0
        print("pid: {}".format(os.getpid()))
        self.val_epoch=0
    def _specify_input_names(self):
        """
        specify data_names and label_names
        """
        raise NotImplementedError

    def _get_weights(self):
        self.weights = self.train_loader._get_weight(configs['per_class_weights']) 

    def do_checkpoint(self, epoch):
        arg_params, aux_params = self.module.get_params()
        mx.model.save_checkpoint(self.model_prefix, epoch+1, self.symbol, arg_params, aux_params)

    def _get_symbol(self):
        """ get self.symbol """
        raise NotImplementedError

    def prepare_for_training(self, params=None, loader=None, inTraining=False):
        """
        get symbol and module,
        init or set params,
        get optimizer
        """
        if loader is None:
            loader = self.train_loader
        self._get_symbol()
        self.module = mx.mod.Module(self.symbol, context=self.ctx, data_names=self.data_names, label_names=self.label_names)
        self.module.bind(data_shapes=loader.provide_data, label_shapes=loader.provide_label)
        if params is None:
            self.module.init_params(initializer=mx.init.Xavier())
        else:
            arg_params, aux_params = params
            self.module.init_params(initializer=mx.init.Xavier(), arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)
        self.module.init_optimizer(optimizer=self.optimizer, optimizer_params=self.optimizer_params, kvstore=configs['kvstore'])
        if not inTraining:
            arg_params, aux_params = self.module.get_params()
            pprint(sorted([(k, v.shape) for (k, v) in arg_params.items()]))
            print('Total number of parameters:', sum([np.prod(v.shape) for (k, v) in arg_params.items()]))
            pprint(sorted([(k, v.shape) for (k, v) in aux_params.items()]))
        print("done prepared data")

    def reset_bn_decay(self):
        """ update module for a new bn_decay """
        self.bn_decay_step += 1
        self.bn_decay = min(1 - (1 - configs['bn_decay']) * (configs['bn_decay_factor'] ** self.bn_decay_step), configs['bn_decay_clip'])
        self.prepare_for_training(self.module.get_params(), inTraining=True)
        print("self.bn_decay", self.bn_decay)

    def _get_data_loaders(self):
        """ get self.train_loader, self.val_loader """
        raise NotImplementedError

    def _get_metric(self):
        """ get self.metric (for training metric or optionally validation metric """
        raise NotImplementedError

    def evaluate(self, epoch):
        """ evaluate one epoch. Can be overridden """
        self.metric.reset()
        for batch in self.val_loader:
            self.module.forward(batch, is_train=False)
            self.module.update_metric(self.metric, batch.label)
        print('Epoch %d, Val %s' % (epoch, self.metric.get()))
        acc = self.metric.get()[1][0]
        if self.best_val_acc < acc:
            self.best_val_acc = acc
            print("new best val acc:", self.best_val_acc)

    def train(self, evlonly=False, queued=False):
        # get data loaders
        self._get_data_loaders()
        # training per_class_weights
        self._get_weights()
        # evaluation metrics
        self._get_metric()
        # optimizer
        self.optimizer = configs['optimizer']
        self.optimizer_params = {
                'learning_rate': configs['lr'],
                'wd': configs['weight_decay']}
        if self.optimizer == 'sgd':
            self.optimizer_params['momentum'] = configs['momentum']
        elif self.optimizer == 'adam':
            self.optimizer_params['beta1'] = configs['beta1']
            self.optimizer_params['beta2'] = configs['beta2']
        else:
            raise RuntimeError("unknown optimizer")
        if configs['lr_policy'] == 'step':
            self.optimizer_params['lr_scheduler'] = mx.lr_scheduler.MultiFactorScheduler(
                      step=[self.train_loader.num_batches*int(epc) for epc in configs['lr_factor_epochs']], factor=configs['lr_factor'])
        # save directory
        self.model_dir = os.path.join(configs['model_dir'], configs['save_model_prefix'], configs['timestamp'])
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_prefix = os.path.join(self.model_dir, configs['save_model_prefix'])
        # bn_decay settings
        self.bn_decay = configs['bn_decay']
        self.bn_decay_step = 0
        # load pretrained model
        if configs['load_model_epoch'] > 0:
            _, arg_params, aux_params = mx.model.load_checkpoint(configs['load_model_prefix'], configs['load_model_epoch'])
            params = arg_params, aux_params
        else:
            params = None
        # get symbol and module
        self.prepare_for_training(params)
        # start fitting
        print(configs)
        train_timer = utils.Timer()
        data_timer = utils.Timer()
        self.train_loader.start()
        self.val_loader.start()
        # self.evaluate()
        fetched_epoch=0
        besthappend = False
        for epoch in range(configs['num_epochs']):
            self.metric.reset()
            train_timer.reset()
            data_timer.reset()
            data_timer.tic()
            if not evlonly:
                while fetched_epoch == epoch:
                    data_timer.tic()
                    batch, fetched_epoch, batch_counter = self.train_loader.fetch()
                    data_timer.toc()
                    train_timer.tic()
                    self.module.forward(batch, is_train=True)
                    self.module.update_metric(self.metric, batch.label)
                    self.module.backward()
                    self.module.update()
                    train_timer.toc()
                    # display metrics on training set
                    if (batch_counter) % configs['display_interval'] == 0:
                        print('Epoch {}, Batch {}, Train {}, lr {}, dataload {:.4f}s, train {:.4f}s' .format(epoch+1, batch_counter, self.metric.get(), self.optimizer_params['lr_scheduler'].base_lr, data_timer.get(), train_timer.get()))
            # evaluate
            if (epoch+1) % configs['val_interval'] == 0 and (epoch >= configs["num_epochs"] // 2 or epoch>0 or configs["load_model_epoch"] > 0) or ((epoch+1) %(configs['val_interval']*5) == 0):
                print('Running evaluation...')
                besthappend = self.evaluate()
            # checkpoint
            if not evlonly and (epoch+1) % configs['checkpoint_interval'] == 0 or besthappend:
                self.do_checkpoint(epoch)
                besthappend = False
            if 'bn_decay_epochs' in configs.keys() and epoch in configs['bn_decay_epochs']:
                self.reset_bn_decay()
        print("final best acc: {}".format(self.best_val_acc))
        print("final best miou: {}".format(self.best_mciou)) 
        self.train_loader.shutdown()
        self.val_loader.shutdown()
        self.train_loader.join()
        self.val_loader.join()
        # save final model
        self.do_checkpoint(epoch)

    def evl_only(self):
        self._get_data_loaders()
        # training per_class_weights
        self._get_weights()
        # evaluation metrics
        self._get_metric()
        self.model_dir = os.path.join(configs['model_dir'], configs['save_model_prefix'], configs['timestamp'])
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_prefix = os.path.join(self.model_dir, configs['save_model_prefix'])
        # bn_decay settings
        self.bn_decay = configs['bn_decay']
        self.bn_decay_step = 0
        self.bn_decay_counter = 0
        # load pretrained model
        if configs['load_model_epoch'] > 0:
            _, arg_params, aux_params = mx.model.load_checkpoint(configs['load_model_prefix'],
                                                                 configs['load_model_epoch'])
            params = arg_params, aux_params
        else:
            params = None
        # get symbol and module

        # optimizer
        self.optimizer = configs['optimizer']
        self.optimizer_params = {
            'learning_rate': configs['lr'],
            'wd': configs['weight_decay'],
            # 'rescale_grad': 1. / self.batch_size,
        }
        if self.optimizer == 'sgd':
            self.optimizer_params['momentum'] = configs['momentum']
        elif self.optimizer == 'adam':
            self.optimizer_params['beta1'] = configs['beta1']
            self.optimizer_params['beta2'] = configs['beta2']
        else:
            raise RuntimeError("unknown optimizer")

        self.prepare_for_training(params, loader=self.val_loader)
        # start fitting
        print(configs)
        self.evaluate(0)


