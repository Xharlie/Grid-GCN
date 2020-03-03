import mxnet as mx
import os
from pprint import pprint
import numpy as np

import utils.utils as utils

class BaseSolver(object):
    def __init__(self,configs):
        self.configs=configs
        self.batch_size = self.configs['batch_size']
        self.num_points = self.configs['num_points']
        if self.configs['use_cpu']:
            self.ctx = mx.cpu()
            self.num_devices = 1
        else:
            self.ctx = [mx.gpu(int(i)) for i in configs['gpus'].split(',')]
            self.num_devices = len(self.ctx)
        self._specify_input_names()
        self.best_val_acc=0
        self.best_val_mcacc=0
        print("pid: {}".format(os.getpid()))

    def _specify_input_names(self):
        """
        specify data_names and label_names
        """
        raise NotImplementedError

    def _get_weights(self):
        if self.configs['per_class_weights'] == 'none':
            self.weights = None
        elif self.configs['per_class_weights'] == 'manual':
            self.weights = self.configs['manual_weights']
        elif self.configs['per_class_weights'] == 'auto':
            self.weights = self.train_loader.get_weight_gradient_multiplier()
        else:
            raise RuntimeError("unknown per_class_weights parameter")

    def do_checkpoint(self, epoch):
        arg_params, aux_params = self.module.get_params()
        mx.model.save_checkpoint(self.model_prefix, epoch+1, self.symbol, arg_params, aux_params)

    def _get_symbol(self):
        """ get self.symbol """
        raise NotImplementedError

    def prepare_for_training(self, params=None, evlonly=False):
        """
        get symbol and module,
        init or set params,
        get optimizer
        """
        self._get_symbol()
        self.module = mx.mod.Module(self.symbol, context=self.ctx, data_names=self.data_names, label_names=self.label_names)
        self.module.bind(data_shapes=self.train_loader.provide_data, label_shapes=self.train_loader.provide_label)
        if params is None:
            self.module.init_params(initializer=mx.init.Xavier())
        else:
            arg_params, aux_params = params
            self.module.init_params(initializer=mx.init.Xavier(), arg_params=arg_params, aux_params=aux_params, allow_missing=False, allow_extra=False)
        if not evlonly:
            self.module.init_optimizer(optimizer=self.optimizer, optimizer_params=self.optimizer_params, kvstore=self.configs['kvstore'])
        arg_params, aux_params = self.module.get_params()
        pprint(sorted([(k, v.shape) for (k, v) in arg_params.items()]))
        print('Total number of parameters:', sum([np.prod(v.shape) for (k, v) in arg_params.items()]))
        pprint(sorted([(k, v.shape) for (k, v) in aux_params.items()]))

    def reset_bn_decay(self):
        """ update module for a new bn_decay """
        self.bn_decay_step += 1
        self.bn_decay = min(1 - (1 - self.configs['bn_decay']) * (self.configs['bn_decay_factor'] ** self.bn_decay_step), self.configs['bn_decay_clip'])
        self.prepare_for_training(self.module.get_params())
        print("self.bn_decay", self.bn_decay)

    def _get_data_loaders(self):
        """ get self.train_loader, self.val_loader """
        raise NotImplementedError

    def _get_metric(self):
        """ get self.metric (for training metric or optionally validation metric """
        raise NotImplementedError

    def evaluate(self, epoch):
        """ evaluate one epoch. Can be overridden """
        besthappend = False
        self.val_loader.reset()
        self.val_metric.reset()
        self.per_class_accuracy.reset()
        for batch in self.val_loader:
            self.module.forward(batch, is_train=False)
            self.module.update_metric(self.val_metric, batch.label)
            self.module.update_metric(self.per_class_accuracy, batch.label)
        print('Epoch %d, Val %s' % (epoch, self.val_metric.get()))
        acc = self.val_metric.get()[1][0]
        if self.best_val_acc < acc:
            self.best_val_acc = acc
            print("new best val acc:", self.best_val_acc)
            besthappend = True
        perclassacc = self.per_class_accuracy.get()[1]
        acc_sum = 0
        class_cnt = 0
        for label, acc_, count in perclassacc:
            print('{:^15s}{:10.5f}{:9d}'.format(label, acc_, count))
            acc_sum += acc_
            class_cnt += 1
        acc_class_avg = acc_sum / class_cnt
        print("mcacc: ", acc_class_avg)
        if self.best_val_mcacc < acc_class_avg:
            self.best_val_mcacc = acc_class_avg
            print("new best mean class acc:", self.best_val_mcacc)
            besthappend = True
        return besthappend


    def train(self, evlonly=False):
        # get data loaders
        self._get_data_loaders()
        # training per_class_weights
        self._get_weights()
        # evaluation metrics
        self._get_metric()
        # optimizer
        if not evlonly:
            self.optimizer = self.configs['optimizer']
            self.optimizer_params = {
                    'learning_rate': self.configs['lr'],
                    'wd': self.configs['weight_decay'],
                    #'rescale_grad': 1. / self.batch_size,
                    }
            if self.optimizer == 'sgd':
                self.optimizer_params['momentum'] = self.configs['momentum']
            elif self.optimizer == 'adam':
                self.optimizer_params['beta1'] = self.configs['beta1']
                self.optimizer_params['beta2'] = self.configs['beta2']
            else:
                raise RuntimeError("unknown optimizer")
            if self.configs['lr_policy'] == 'step':
                # self.optimizer_params['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
                #         step=int(configs['lr_factor_sample']/self.batch_size),
                #         factor=configs['lr_factor'])
                self.optimizer_params['lr_scheduler'] = mx.lr_scheduler.MultiFactorScheduler(
                          step=[len(self.train_loader)*int(epc) for epc in self.configs['lr_factor_epochs']], factor=self.configs['lr_factor'])
        # save directory
        self.model_dir = os.path.join(self.configs['model_dir'], self.configs['save_model_prefix'], self.configs['timestamp'])
        if not os.path.isdir(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_prefix = os.path.join(self.model_dir, self.configs['save_model_prefix'])
        # bn_decay settings
        self.bn_decay = self.configs['bn_decay']
        self.bn_decay_step = 0
        self.bn_decay_counter = 0
        # load pretrained model
        if self.configs['load_model_epoch'] > 0:
            _, arg_params, aux_params = mx.model.load_checkpoint(self.configs['load_model_prefix'], self.configs['load_model_epoch'])
            params = arg_params, aux_params
        else:
            params = None
        # get symbol and module

        self.prepare_for_training(params, evlonly=evlonly)
        # start fitting
        print(self.configs)
        train_timer = utils.Timer()
        data_timer = utils.Timer()
        self.evaluate(0)
        besthappend=False
        for epoch in range(self.configs['num_epochs']):
            self.train_loader.reset()
            self.metric.reset()
            train_timer.reset()
            data_timer.reset()
            data_timer.tic()
            if not evlonly:
                for i, batch in enumerate(self.train_loader):
                    data_timer.toc()
                    train_timer.tic()
                    self.module.forward(batch, is_train=True)
                    self.module.update_metric(self.metric, batch.label)
                    self.module.backward()
                    self.module.update()
                    train_timer.toc()
                    # display metrics on training set
                    if (i+1) % self.configs['display_interval'] == 0:
                        print('Epoch %d, Batch %d, Train %s, lr %f' % (epoch+1, i+1, self.metric.get(), self.optimizer_params['lr_scheduler'].base_lr))
                    #pred = self.module.get_outputs()[0].asnumpy().argmax(axis=1)[:,0]
                    #labels = batch.label[0].asnumpy()[:,0].astype(int)
                    #print pred
                    #print labels
                    # reset bn_decay

                    data_timer.tic()
                print('Speed: data {:.4f}, train {:.4f}'.format(data_timer.get(), train_timer.get()))
            if 'bn_decay_epochs' in self.configs.keys() and epoch in self.configs['bn_decay_epochs']:
                self.reset_bn_decay()
            # evaluate
            if (epoch+1) % self.configs['val_interval'] == 0 and epoch >= 20 or self.configs["load_model_epoch"] > 0 or evlonly:
                print('Running evaluation...')
                besthappend = self.evaluate(epoch+1)
            # checkpoint
            if (epoch+1) % self.configs['checkpoint_interval'] == 0 or besthappend:
                self.do_checkpoint(epoch)
                besthappend=False
        print("final best acc: {}".format(self.best_val_acc))
        print("final best mean acc: {}".format(self.best_val_mcacc))
        # save final model
        self.do_checkpoint(epoch)



