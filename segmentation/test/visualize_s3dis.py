#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# from mayavi.mlab import *
import argparse
import os
import pickle
import numpy as np


gt_label_filenames = []
pred_label_filenames = []

DEFAULT_DATA_DIR = '../../data/s3dis'
NUM_CLASSES = 14

p = argparse.ArgumentParser()
p.add_argument(
    "-d", "--data", dest='data_dir',
    default=DEFAULT_DATA_DIR,
    help="Path to S3DIS data (default is %s)" % DEFAULT_DATA_DIR)

args = p.parse_args()

# object_dict = {
#             'clutter':   0,
#             'ceiling':   1,
#             'floor':     2,
#             'wall':      3,
#             'beam':      4,
#             'column':    5,
#             'door':      6,
#             'window':    7,
#             'table':     8,
#             'chair':     9,
#             'sofa':     10,
#             'bookcase': 11,
#             'board':    12}

# objlst = [
#             'clutter',
#             'ceiling',
#             'floor',
#             'wall',
#             'beam',
#             'column',
#             'door',
#             'window',
#             'table',
#             'chair',
#             'sofa',
#             'bookcase',
#             'board']
# cmap = matplotlib.colors.ListedColormap(objlst, 13)
# viridis = cm.get_cmap('viridis', 13)
# print(viridis.colors)
# # norm=matplotlib.colors.Normalize(vmin=-0.5, vmax=unique.max()+0.5)
# fig, ax = plt.subplots(figsize=(5,5))
# im = ax.imshow(objlst, cmap=cmap)

color_dict={0:[128, 128, 128],
1:[128, 128, 0],
2:[255, 215, 180],
3:[0, 128, 128],
4:[250, 190, 190],
5:[230, 190, 255],
6:[0, 130, 200],
7:[210, 245, 60],
8:[230, 25, 75],
9:[170, 110, 40],
10:[128, 128, 0],
11:[0, 0, 128],
12:[128, 0, 0]}

def plot_meshlab(room_nm):
    xyzfile = os.path.join("../../data/3DIS/prepare_label_rgb_1.5/Area_5",room_nm,"xyzrgb.npy")
    label = os.path.join("../../data/3DIS/prepare_label_rgb_1.5/Area_5",room_nm,"label.npy")
    pred = os.path.join("../../data/3DIS/prepare_label_rgb_1.5/Area_5",room_nm,"pred.npy")

    colors= np.asarray([color_dict[i] for i in range(13)])
    xyzrgb = np.load(xyzfile)
    gtlabel = np.load(label)[:,0].astype(np.int32)
    with open(pred, 'r') as f:
        predlabels = f.readlines()
        predlabel=[line.strip() for line in predlabels]
    predlabel = np.asarray(predlabel).astype(np.int32)
    print("predlabel.shape",predlabel.shape,gtlabel.shape,xyzrgb.shape)
    label_mask=np.where(predlabel!=13)
    xyzrgb = xyzrgb[label_mask]
    gtlabel = gtlabel[label_mask]
    predlabel = predlabel[label_mask]

    gtrgb = colors[gtlabel]
    predrgb = colors[predlabel]

    gtxyzseg = np.concatenate([xyzrgb[:,:3],gtrgb],axis=-1)
    predxyzseg = np.concatenate([xyzrgb[:,:3],predrgb],axis=-1)
    mask = np.where(gtlabel!=1)[0]
    print(mask.shape)
    predxyzseg = predxyzseg[mask,:]
    gtxyzseg = gtxyzseg[mask,:]
    xyzrgb = xyzrgb[mask,:]

    np.savetxt("rooms/{}_input.txt".format(room_nm), xyzrgb, delimiter=";")
    np.savetxt("rooms/{}_gt.txt".format(room_nm), gtxyzseg, delimiter=";")
    np.savetxt("rooms/{}_pred.txt".format(room_nm), predxyzseg, delimiter=";")

file_lst=["conferenceRoom_1",  "hallway_11",  "hallway_2",  "hallway_7",  "office_10",  "office_15",  "office_2",   "office_24",  "office_29",  "office_33",  "office_38",  "office_42",  "office_9",   "storage_4",
"conferenceRoom_2",  "hallway_12",  "hallway_3",  "hallway_8",  "office_11",  "office_16",  "office_20",  "office_25",  "office_3",   "office_34",  "office_39",  "office_5",   "pantry_1",   "WC_1",
"conferenceRoom_3",  "hallway_13",  "hallway_4",  "hallway_9",  "office_12",  "office_17",  "office_21",  "office_26",  "office_30",  "office_35",  "office_4",   "office_6",  "storage_1",  "WC_2", "hallway_1", "hallway_14",  "hallway_5",  "lobby_1",    "office_13",  "office_18",  "office_22",  "office_27",  "office_31",  "office_36",  "office_40",  "office_7",   "storage_2", "hallway_10",  "hallway_15",  "hallway_6",  "office_1",   "office_14",  "office_19",  "office_23",  "office_28",  "office_32",  "office_37",  "office_41",  "office_8",   "storage_3",]
for f in file_lst:
    plot_meshlab(f)

# for area in os.listdir(args.data_dir):
#     path_area = os.path.join(args.data_dir, area)
#     if not os.path.isdir(path_area):
#         continue
#     Rooms = os.listdir(path_area)
#     for room in Rooms:
#         path_room = os.path.join(path_area, room)
#         if not os.path.isdir(path_room):
#             continue
#         path_gt_label = os.path.join(path_room, 'label.npy')
#         if not os.path.exists(path_gt_label):
#             print("gt label: %s does not exist, skipping" % path_gt_label)
#             continue
#         path_pred_label = os.path.join(path_room, 'pred.npy')
#         if not os.path.exists(path_pred_label):
#             print("pred label: %s does not exist, skipping" % path_pred_label)
#             continue
#         pred_label_filenames.append(path_pred_label)
#         gt_label_filenames.append(path_gt_label)

# num_room = len(gt_label_filenames)
# num_preds = len(pred_label_filenames)
# assert num_room == num_preds

# print("Found {} predictions".format(num_room))

# gt_classes = [0] * NUM_CLASSES
# positive_classes = [0] * NUM_CLASSES
# true_positive_classes = [0] * NUM_CLASSES

# gt_classes_ignore13 = [0] * (NUM_CLASSES-1)
# positive_classes_ignore13 = [0] * (NUM_CLASSES-1)
# true_positive_classes_ignore13 = [0] * (NUM_CLASSES-1)

# print("Evaluating predictions:")
# for i in range(num_room):
#     print("  {} ({}/{})".format(pred_label_filenames[i], i + 1, num_room))
#     pred_label = np.loadtxt(pred_label_filenames[i])
#     gt_label = np.load(gt_label_filenames[i])
#     for j in range(gt_label.shape[0]):
#         gt_l = int(gt_label[j])
#         pred_l = int(pred_label[j])
#         gt_classes[gt_l] += 1
#         positive_classes[pred_l] += 1
#         true_positive_classes[gt_l] += int(gt_l==pred_l)
#         if pred_l != 13 and gt_l != 13:
#             gt_classes_ignore13[gt_l] += 1
#             positive_classes_ignore13[pred_l] += 1
#             true_positive_classes_ignore13[gt_l] += int(gt_l==pred_l)

