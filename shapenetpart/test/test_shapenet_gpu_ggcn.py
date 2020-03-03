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
import time, math
from datetime import datetime
import h5py
from utils import utils, data_utils
# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
CLS_NUM = 16
SEG_NUM_CLS = 50


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


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
                take_shapes, take_up_shapes, bn_decay=self.bn_decay, weights=None)


    def _get_metric(self):
        self.seg_classes = self.val_loader.seg_classes  
        self.seg_label_to_cat = {}
        for cat in self.seg_classes.keys():
            for label in self.seg_classes[cat]:
                self.seg_label_to_cat[label] = cat

        self.metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.CrossEntropyWithIgnore(ndim=3, axis=1)])
        self.val_metric = mx.metric.CompositeEvalMetric([mx.metric.Accuracy(axis=1), metrics.CrossEntropyWithIgnore(ndim=3, axis=1), metrics.PerClassAccuracy(num_class=SEG_NUM_CLS, axis=1, ignore_label=None)])
        CLASS_NAMES = {0: 'Airplane', 1: 'Bag', 2: 'Cap', 3: 'Car', 4: 'Chair', 5: 'Earphone', 6: 'Guitar', 7: 'Knife', 8: 'Lamp', 9: 'Laptop', 10: 'Motorbike', 11: 'Mug', 12: 'Pistol', 13: 'Rocket', 14: 'Skateboard', 15: 'Table'}




    def get_sample(self, sample_num, data, labels, obj_labels, batch_indices_shuffle):
        batch_dataxyz_sub = np.zeros((self.batch_size, sample_num, 3), dtype=np.float32)
        batch_obj_label_sub = np.zeros((self.batch_size, CLS_NUM), dtype=np.float32)
        # batch_datafeat_sub = np.zeros((self.batch_size, sample_num, configs["indim"]), dtype=np.float32)
        points_num_batch_sub = np.zeros((self.batch_size, 1), dtype=np.int32)
        batch_label_sub =  np.full((self.batch_size, sample_num), 13, dtype=np.int32)
        for i in range(self.batch_size):
            indices = batch_indices_shuffle[i*sample_num: (i+1)*sample_num]
            dataxyz = data[indices, :3].copy()                
            # datafeat = np.concatenate([data[indices, 1, None] / 3 - 0.5, data[indices, 3:]], axis=-1) if configs['indim'] == 4 else data[indices, 3:]
            dataxyz = utils.normalize_point_cloud(dataxyz[:, :3])   
            
            batch_dataxyz_sub[i, :sample_num,:] = dataxyz
            # batch_datafeat_sub[i, :sample_num,:] = datafeat
            points_num_batch_sub[i, 0] = sample_num
            batch_label_sub[i,:sample_num] = labels[indices]
            batch_obj_label_sub[i,:] = obj_labels

        batch_data = [mx.ndarray.array(batch_dataxyz_sub),
                    mx.ndarray.array(points_num_batch_sub, dtype='int32'),
                    mx.ndarray.array(batch_obj_label_sub, dtype='float32')]
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
        self.module.bind(data_shapes=[('data', (self.batch_size, configs['num_points'], 3)), ('actual_centnum', (self.batch_size, 1), "int32") ,('cls', (self.batch_size, CLS_NUM), "float32")], label_shapes=[('label', (self.batch_size, configs['num_points']))])
        if params is None:
            self.module.init_params(initializer=mx.init.Xavier())
        else:
            arg_params, aux_params = params
            self.module.init_params(initializer=mx.init.Xavier(), arg_params=arg_params, aux_params=aux_params, allow_missing=True, allow_extra=True)
        print("done prepared data")

    def inference_create(self, repeat_num = 10, filelist="../data/3DIS/prepare_label_rgb/val_files_Area_5.txt"):
        max_point_num = 2560
        self._get_data_loaders()
        self._get_metric()
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
            print("error, load_model_epoch is zero!")
            exit()
        # get symbol and module
        self.prepare_for_testing(params)
        print("model loaded!!")
        """
        When evaluating, convert per-point accuracy to per-voxel accuracy
        """
        self.weights = None
        shape_ious = {cat:[] for cat in self.seg_classes.keys()}
        shape_ious_vote = {cat:[] for cat in self.seg_classes.keys()}
        for fileindex in range(len(self.val_loader.datapath)):
            fn = self.val_loader.datapath[fileindex]
            cat = self.val_loader.datapath[fileindex][0]
            cls = self.val_loader.classes[cat]
            cls = np.array([cls]).astype(np.int64)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:,0:3]
            point_set = pc_normalize(point_set)
            label_seg = data[:,-1].astype(np.int64)

            # filename = filenames[fileindex]
            # print('{}-Reading {}...'.format(datetime.now(), filename))
            # data_h5 = h5py.File(filename, 'r')
            # data = data_h5['data'][...].astype(np.float32)
            # data_num = data_h5['data_num'][...].astype(np.int32)
            # label_seg = data_h5['label_seg'][...].astype(np.int32)
            batch_num = 1
            point_num_gt = data.shape[0]
            labels_pred = np.full((batch_num, point_num_gt), -1, dtype=np.int32)
            confidences_pred = np.zeros((batch_num, point_num_gt), dtype=np.float32)
            labels_pred_vote = np.full((batch_num, point_num_gt), -1, dtype=np.int32)
            confidences_pred_vote = np.zeros((batch_num, point_num_gt), dtype=np.float32)
            print('{}-{:d} testing batches. {}/{} shapes'.format(datetime.now(), batch_num, fileindex+1, len(self.val_loader.datapath)))
            for batch_idx in range(batch_num):
                points_batch = data
                
                labels_gt = label_seg
                obj_labels_gt = np.zeros((CLS_NUM))
                obj_labels_gt[cls] = 1.
                dataxyz = points_batch[:point_num_gt,:3]
                tile_num = math.ceil((sample_num * self.batch_size) / point_num_gt)
                
                indices_shuffle = np.tile(np.arange(point_num_gt), tile_num)[0: sample_num * self.batch_size]
                np.random.shuffle(indices_shuffle)
                batch = self.get_sample(sample_num, points_batch, labels_gt, obj_labels_gt, indices_shuffle)
                self.module.forward(batch, is_train=False)
                probs_2d = self.module.get_outputs()[0].asnumpy().swapaxes(1,2).reshape(sample_num * self.batch_size, -1)
                predictions = [(-1, 0.0)] * point_num_gt
                predictions_vote_acc = [[[0, 0.0] for i in range(SEG_NUM_CLS)]] * point_num_gt
                predictions_vote = [(-1, 0.0)] * point_num_gt
                cat = self.seg_label_to_cat[labels_gt[0]]
                best_voted_label_cnt = [0] *  point_num_gt
                best_voted_label = [-1] *  point_num_gt
                best_voted_label_confidence= [0] *  point_num_gt
                for idx in range(sample_num * self.batch_size):
                    point_idx = indices_shuffle[idx]
                    probs = probs_2d[idx, self.seg_classes[cat]]

                    confidence = np.amax(probs)
                    label = np.argmax(probs) + self.seg_classes[cat][0]
                    if confidence > predictions[point_idx][1]:
                        predictions[point_idx] = [label, confidence]
                        # print(confidence)
                    predictions_vote_acc[point_idx][label][0] +=1    
                    predictions_vote_acc[point_idx][label][1] +=confidence
                    if label != best_voted_label[point_idx] and (
                        predictions_vote_acc[point_idx][label][0] > best_voted_label_cnt[point_idx] or 
                        predictions_vote_acc[point_idx][label][0] == best_voted_label_cnt[point_idx] and 
                        predictions_vote_acc[point_idx][label][1] >  best_voted_label_confidence[point_idx]):
                        best_voted_label_cnt[point_idx] = predictions_vote_acc[point_idx][label][0]
                        best_voted_label_confidence[point_idx] = predictions_vote_acc[point_idx][label][1]
                        best_voted_label[point_idx] = label
                        # print("best_voted_label_cnt", best_voted_label_cnt[point_idx])
                        predictions_vote[point_idx] = [label, best_voted_label_confidence[point_idx]]

                labels_pred[batch_idx, 0:point_num_gt] = np.array([label for label, _ in predictions])
                confidences_pred[batch_idx, 0:point_num_gt] = np.array([confidence for _, confidence in predictions])

                part_ious = [0.0 for _ in range(len(self.seg_classes[cat]))]
                count = 0
                for l in self.seg_classes[cat]:
                    if np.sum((labels_gt == l)) < 40:
                        next;
                    count+=1
                    if np.sum((labels_gt == l) | (labels_pred == l)) == 0:
                        # part is not present in this shape
                        part_ious[l - self.seg_classes[cat][0]] = 1.0
                    else:
                        part_ious[l - self.seg_classes[cat][0]] = float(np.sum((labels_gt == l) & (labels_pred == l))) / float(np.sum((labels_gt == l) | (labels_pred == l)))
                # print(part_ious)
                shape_ious[cat].append(np.sum(part_ious)/count)
                print(cat,shape_ious[cat][-1])



                labels_pred_vote[batch_idx, 0:point_num_gt] = np.array([label if confidence > 0.6 else vlabel for [label, confidence],[vlabel, _] in zip(predictions, predictions_vote)])
                confidences_pred_vote[batch_idx, 0:point_num_gt] = np.array([confidence for _, confidence in predictions_vote])

                count = 0
                part_ious_vote = [0.0 for _ in range(len(self.seg_classes[cat]))]
                for l in self.seg_classes[cat]:
                    if np.sum((labels_gt == l)) < 40:
                        next
                    count+=1
                    if np.sum((labels_gt == l) | (labels_pred_vote == l)) == 0:
                        # part is not present in this shape
                        part_ious_vote[l - self.seg_classes[cat][0]] = 1.0
                    else:
                        part_ious_vote[l - self.seg_classes[cat][0]] = float(np.sum((labels_gt == l) & (labels_pred_vote == l))) / float(np.sum((labels_gt == l) | (labels_pred_vote == l)))
                # print(part_ious_vote)
                shape_ious_vote[cat].append(np.sum(part_ious_vote)/count)
                print(cat,shape_ious_vote[cat][-1])



        
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

        print("all tested!")



        instance_ious_vote = []
        for cat in shape_ious_vote.keys():
            for iou in shape_ious_vote[cat]:
                instance_ious_vote.append(iou)
            shape_ious_vote[cat] = np.mean(shape_ious_vote[cat])
        mean_class_ious_vote = np.mean(list(shape_ious_vote.values()))
        mean_part_ious_vote = np.mean(instance_ious_vote)
        for cat in sorted(shape_ious_vote.keys()):
            print('****** %s: %0.6f'%(cat, shape_ious_vote[cat]))
        print('************vote_vote Class_mIoU: %0.6f' % (mean_class_ious_vote))
        print('************vote Part_mIoU: %0.6f' % (mean_part_ious_vote))

        print("all tested!")



if __name__ == "__main__":
    solver = ShapenetSegSolver()
    solver.inference_create()

    # for f in $(find ./ -name "*" -maxdepth 1); do echo $f; grep 'best iou' $f | tail -1; done# 



