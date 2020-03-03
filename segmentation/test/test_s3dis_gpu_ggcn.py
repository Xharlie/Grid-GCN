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
import utils
from models.s3dis_ggcn_models_g import get_symbol_seg_ggcn
from data_loader.s3dis_gpu_loader import S3DISLoader
from train.s3dis_base_solver import BaseSolver
from utils.utils import point_cloud_label_to_surface_voxel_label
from utils import metrics
from s3dis_configs.configs import configs
import mxnet.profiler as profiler
import time, math
from datetime import datetime
import h5py
from utils import utils, data_utils
# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

class S3disSolver(BaseSolver):
    def __init__(self):
        super(S3disSolver, self).__init__()

    # def _specify_input_names(self):
    #     self.data_names = ['data']
    #     self.label_names = ['label']

    def _specify_input_names(self):
        self.data_names = ['dataxyz', "datafeat", "actual_centnum"]
        self.label_names = ['label']

    def _get_data_loaders(self):
        area=configs["area"]
        val_loader = S3DISLoader(
            '../data/3DIS/prepare_label_rgb/val_files_Area_{}.txt'.format(area),
            batch_size=configs['batch_size'],
            npoints=configs['num_points'], split='test',
            normalize=configs["normalize"], use_cache=True, max_epoch=configs["num_epochs"],
            drop_factor=configs['input_dropout_ratio'],
            area=area
        )
        return val_loader


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
                                configs['up_max_p_grid_lst'][i],
                                # configs['max_o_grid_lst'][-1] if i == 0 else configs['up_max_o_grid_lst'][i - 1],
                                configs['max_o_grid_lst'][-i - 1],
                                configs['up_inputDim'][i]])
        print("take_up_shapes",take_up_shapes)
        self.symbol = get_symbol_seg_ggcn(self.batch_size // self.num_devices, self.num_points,
                take_shapes, take_up_shapes, bn_decay=self.bn_decay, weights=self.weights if configs["per_class_weights"] != "none" else None)


    def _get_metric(self):
        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.AccuracyWithIgnore(axis=1, ignore_label=13),
            metrics.CrossEntropyWithIgnore(ndim=3, axis=1, ignore_label=13, label_weights=self.weights)])
        self.val_metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.AccuracyWithIgnore(axis=1),
            metrics.CrossEntropyWithIgnore(ndim=3, axis=1, label_weights=None), metrics.PerClassAccuracy(num_class=14, axis=1, ignore_label=13)])
        CLASS_NAMES = {0:'ceiling', 1:'floor', 2:'wall', 3:'beam', 4:'column', 5:'window',
    6:'door', 7:'table', 8:'chair', 9:'sofa', 10:'bookcase', 11:'board', 12:'clutter'}
        self.per_class_accuracy = metrics.PerClassAccuracy(num_class=14, axis=1, class_dict=CLASS_NAMES, report_occurrence=True, ignore_label=13)
        self.per_class_iou = metrics.PerClassIoU(num_class=14, axis=1, class_dict=CLASS_NAMES, report_occurrence=True, ignore_label=13)




    def get_sample(self, sample_num, data, labels, batch_indices_shuffle):
        batch_dataxyz_sub = np.zeros((self.batch_size, sample_num, 3), dtype=np.float32)
        batch_datafeat_sub = np.zeros((self.batch_size, sample_num, configs["indim"]), dtype=np.float32)
        points_num_batch_sub = np.zeros((self.batch_size, 1), dtype=np.int32)
        batch_label_sub =  np.full((self.batch_size, sample_num), 13, dtype=np.int32)
        for i in range(self.batch_size):
            indices = batch_indices_shuffle[i*sample_num: (i+1)*sample_num]
            dataxyz = data[indices, :3].copy()                
            datafeat = np.concatenate([data[indices, 1, None] / 3 - 0.5, data[indices, 3:]], axis=-1) if configs['indim'] == 4 else data[indices, 3:]
            if configs['normalize'] == "ball":
                dataxyz = utils.normalize_point_cloud(dataxyz[:, :3])
            elif configs['normalize'] == "square":
                dataxyz = utils.normalize_point_cloud_square(dataxyz[:, :3])
            elif configs['normalize'] == "ballnoy":
                dataxyz = utils.normalize_point_cloud(dataxyz[:, :3], double_axis=1)
            elif configs['normalize'] == "squarenoy":
                dataxyz = utils.normalize_point_cloud_square(dataxyz[:, :3], double_axis=1)

            batch_dataxyz_sub[i, :sample_num,:] = dataxyz
            batch_datafeat_sub[i, :sample_num,:] = datafeat
            points_num_batch_sub[i, 0] = sample_num
            batch_label_sub[i,:sample_num] = labels[indices]

        batch_data = [mx.ndarray.array(batch_dataxyz_sub), mx.ndarray.array(batch_datafeat_sub), mx.ndarray.array(points_num_batch_sub,dtype='int32')]
        batch_label = [mx.ndarray.array(batch_label_sub)]
        return mx.io.DataBatch(data=batch_data, label=batch_label)

    def prepare_for_testing(self, params=None):
        """
        get symbol and module,
        init or set params,
        get optimizer
        """

        self._get_symbol()
        self.module = mx.mod.Module(self.symbol, context=self.ctx, data_names=self.data_names, label_names=self.label_names)
        self.module.bind(data_shapes=[('dataxyz', (self.batch_size, configs['num_points'], 3)), ('datafeat', (self.batch_size, configs['num_points'], configs["indim"])), ('actual_centnum', (self.batch_size, 1), "int32")], label_shapes=[('label', (self.batch_size, configs['num_points']))])
        if params is None:
            self.module.init_params(initializer=mx.init.Xavier())
        else:
            arg_params, aux_params = params
            self.module.init_params(initializer=mx.init.Xavier(), arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)
        print("done prepared data")


    def inference_create(self, repeat_num = 4):
        filelist="../data/3DIS/prepare_label_rgb/val_files_Area_{}.txt".format(configs["area"]) 
        max_point_num = 8192
        filter_extrema= 2
        tolerance = 0.7
        sample_num = configs["num_points"]
        self.batch_size = repeat_num * math.ceil(max_point_num / sample_num)
        folder = os.path.dirname(filelist)
        filenames = [line.strip() for line in open(filelist)]

        # save directory and load model
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
        self.prepare_for_testing(params)
        print("model loaded!!")

        folder = os.path.dirname(filelist)
        filenames = [os.path.join(folder, line.strip()) for line in open(filelist)]

        """
        When evaluating, convert per-point accuracy to per-voxel accuracy
        """
        self.weights = None
        for fileindex in range(len(filenames)):
            filename = filenames[fileindex]
            print('{}-Reading {}...'.format(datetime.now(), filename))
            data_h5 = h5py.File(filename, 'r')
            data = data_h5['data'][...].astype(np.float32)
            data_num = data_h5['data_num'][...].astype(np.int32)
            label_seg = data_h5['label_seg'][...].astype(np.int32)
            batch_num = data.shape[0]
            labels_pred = np.full((batch_num, max_point_num), 13, dtype=np.int32)
            confidences_pred = np.zeros((batch_num, max_point_num), dtype=np.float32)
            print('{}-{:d} testing batches. {}/{} rooms'.format(datetime.now(), batch_num, fileindex+1, len(filenames)))
            for batch_idx in range(batch_num):
                if batch_idx % 10 == 0:
                    print('{}-Processing {} of {} batches.'.format(datetime.now(), batch_idx, batch_num))
                points_batch = data[batch_idx, ...]
                point_num_gt = data_num[batch_idx]
                labels_gt = label_seg[batch_idx]
                dataxyz = points_batch[:point_num_gt,:3]
                indx_filtered, point_num = data_utils.filter_extre_indx(dataxyz, filter_extrema)
                tile_num = math.ceil((sample_num * self.batch_size) / point_num)
                if point_num > sample_num * tolerance:
                    indices_shuffle = np.tile(indx_filtered, tile_num)[0: sample_num * self.batch_size]
                    np.random.shuffle(indices_shuffle)
                    batch = self.get_sample(sample_num, points_batch, labels_gt, indices_shuffle)
                    self.module.forward(batch, is_train=False)
                    probs_2d = self.module.get_outputs()[0].asnumpy().swapaxes(1,2).reshape(sample_num * self.batch_size, -1)
                    predictions = [(13, 0.0)] * point_num_gt
                    for idx in range(sample_num * self.batch_size):
                        point_idx = indices_shuffle[idx]
                        probs = probs_2d[idx, :]
                        confidence = np.amax(probs)
                        label = np.argmax(probs)
                        if confidence > predictions[point_idx][1]:
                            predictions[point_idx] = [label, confidence]
                    labels_pred[batch_idx, 0:point_num_gt] = np.array([label for label, _ in predictions])
                    confidences_pred[batch_idx, 0:point_num_gt] = np.array([confidence for _, confidence in predictions])
                else:
                    print("point_num <= sample_num * tolerance",point_num, sample_num, tolerance)
            corr_vote_num_all = np.sum((labels_pred == label_seg) & (label_seg < 13))
            all_vote_num_all = np.sum(label_seg != 13)
            all_vote_num_all_ig13 = np.sum((label_seg < 13) & (labels_pred < 13))
            print("corr_vote: {}, all pt num {}, true all pt num {}, acc{}, true acc{}"
                .format(corr_vote_num_all, all_vote_num_all, all_vote_num_all_ig13, 
                    corr_vote_num_all/all_vote_num_all, corr_vote_num_all/all_vote_num_all_ig13))

            filename_pred = filename[:-3] + '_pred.h5'
            print('{}-Saving {}...'.format(datetime.now(), filename_pred))
            file = h5py.File(filename_pred, 'w')
            file.create_dataset('data_num', data=data_num)
            file.create_dataset('label_seg', data=labels_pred)
            file.create_dataset('confidence', data=confidences_pred)
            has_indices = 'indices_split_to_full' in data_h5
            if has_indices:
                file.create_dataset('indices_split_to_full', data=data_h5['indices_split_to_full'][...])
            file.close()
        print("all tested!")



if __name__ == "__main__":
    solver = S3disSolver()
    solver.inference_create()

    # for f in $(find ./ -name "*" -maxdepth 1); do echo $f; grep 'best iou' $f | tail -1; done# 



