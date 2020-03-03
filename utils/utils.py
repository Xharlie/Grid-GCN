import mxnet as mx
import numpy as np
import time
import random
# import subprocess
from datetime import datetime
import matplotlib.pyplot as plt


# from mpl_toolkits.mplot3d import Axes3D



def normalize_point_cloud_square(pc, double_axis=None):
    """
    Normalize a point cloud: mean-shift + variance
    Input: Nx3 point cloud
    Return: Nx3 normalized
    """
    centerArray = np.mean(pc, axis=0)
    pc = pc - centerArray
    maxdim = np.max(np.abs(pc), axis=0)
    if double_axis is not None:
        maxdim[double_axis] = maxdim[double_axis] * 0.5
    maxdim = np.max(maxdim)
    pc = pc / np.maximum(1e-5, maxdim)
    return pc

def normalize_point_cloud_square_noy(pc, double_axis=None):
    """
    Normalize a point cloud: mean-shift + variance
    Input: Nx3 point cloud
    Return: Nx3 normalized
    """
    centerArray = np.mean(pc, axis=0)
    pc = pc - centerArray
    maxdim = np.max(np.abs(pc), axis=0)
    if double_axis is not None:
        maxdim[double_axis] = maxdim[double_axis] * 0.5
    maxdim = np.max(maxdim)
    safe_maxdim = np.maximum(1e-5, maxdim)
    pc = pc / safe_maxdim
    pc[:, 1] = pc[:, 1] + centerArray[1] / safe_maxdim
    return pc


def normalize_point_cloud(pc, double_axis=None):
    """
    Normalize a point cloud: mean-shift + variance
    Input: Nx3 point cloud
    Return: Nx3 normalized
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    if double_axis is not None:
        pc[:, double_axis] = pc[:, double_axis] * 0.5
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / np.maximum(1e-5, m)
    if double_axis is not None:
        pc[:, double_axis] = pc[:, double_axis] * 2
    return pc

def normalize_point_cloud_noy(pc):
    """
    Normalize a point cloud: mean-shift + variance
    Input: Nx3 point cloud
    Return: Nx3 normalized
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    safe_m = np.maximum(1e-5, m)
    pc = pc / safe_m
    pc[:,1] = pc[:,1] + centroid[1] / safe_m
    return pc

def normalize_point_cloud_in_scope(pc_batch):
    """
    Normalize a point cloud: mean-shift + variance
    Input: Nx3 point cloud
    Return: Nx3 normalized
    """
    maxdim_batch = np.max(np.abs(pc_batch), axis=(1,2), keepdims=True)
    if np.max(maxdim_batch) > 1:
        pc_batch = pc_batch / np.maximum(1e-5, maxdim_batch)
    return pc_batch

def normalize_point_cloud_batch(pc_batch):
    """
    Normalize a point cloud: mean-shift + variance
    Input: BXNx3 point cloud
    Return: BXNx3 normalized
    """
    centroid_batch = np.mean(pc_batch, axis=1, keepdims=True)
    pc_batch = pc_batch - centroid_batch
    m_batch = np.max(np.sqrt(np.sum(pc_batch ** 2, axis=2, keepdims=True)), axis=1, keepdims=True)
    pc_batch = pc_batch / np.maximum(1e-5, m_batch)
    return pc_batch


def normalize_point_cloud_batch_square(pc_batch):
    """
    Normalize a point cloud: mean-shift + variance
    Input: Nx3 point cloud
    Return: Nx3 normalized
    """
    centerArray_batch = np.mean(pc_batch, axis=1, keepdims=True)
    pc_batch = pc_batch - centerArray_batch
    maxdim_batch = np.max(np.abs(pc_batch), axis=(1,2), keepdims=True)
    pc_batch = pc_batch / np.maximum(1e-5, maxdim_batch)
    return pc_batch


def normalize_point_cloud_batch_noy(pc_batch):
    """
    Normalize a point cloud: mean-shift + variance
    Input: BXNx3 point cloud
    Return: BXNx3 normalized
    """
    centroid_batch = np.mean(pc_batch, axis=1, keepdims=True)
    pc_batch = pc_batch - centroid_batch
    m_batch = np.max(np.sqrt(np.sum(pc_batch ** 2, axis=2, keepdims=True)), axis=1, keepdims=True)
    safe_m = np.maximum(1e-5, m_batch)
    pc_batch = pc_batch / safe_m
    # print(centroid_batch[:,:,2].shape, safe_m.shape)
    pc_batch[:, :, 1] = pc_batch[:, :, 1] + centroid_batch[:,:,1] / safe_m[:,0]
    return pc_batch


def normalize_point_cloud_batch_square_noy(pc_batch):
    """
    Normalize a point cloud: mean-shift + variance
    Input: Nx3 point cloud
    Return: Nx3 normalized
    """
    centerArray_batch = np.mean(pc_batch, axis=1, keepdims=True)
    pc_batch = pc_batch - centerArray_batch
    maxdim_batch = np.max(np.abs(pc_batch), axis=(1,2), keepdims=True)
    safe_maxdim = np.maximum(1e-5, maxdim_batch)
    pc_batch = pc_batch / safe_maxdim
    pc_batch[:, :, 1] = pc_batch[:, :, 1] + centerArray_batch[..., 1] / safe_maxdim[:,0,[0]]
    return pc_batch

def normalize_point_cloud_param(pc):
    """
    Normalize a point cloud: mean-shift + variance
    Input: Nx3 point cloud
    Return: Nx3 normalized
    """
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc, centroid, m


def rotate_point_cloud(batch_data):
    """
    Randomly rotate the point clouds around y axis to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_z(batch_data):
    """
    Randomly rotate the point clouds around z axis to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    center = batch_data.mean(axis=1, keepdims=True)
    batch_data = batch_data - center
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, -sinval, 0],
                                    [sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    rotated_data = rotated_data + center
    return rotated_data


def rotate_point_cloud_y(batch_data):
    """
    Randomly rotate the point clouds around z axis to augument the dataset
    rotation is per shape based along up direction
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    center = batch_data.mean(axis=1, keepdims=True)
    batch_data = batch_data - center
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    rotated_data = rotated_data + center
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """
    Rotate the point cloud along up direction with certain angle.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, :, 0:3]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """
    Rotate the point cloud along up direction with certain angle.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        # rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """
    Randomly perturb the point clouds by small rotations
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k, :, 0:3]
        shape_normal = batch_xyz_normal[k, :, 3:6]
        batch_xyz_normal[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal


def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(angles[0]), -np.sin(angles[0])],
                       [0, np.sin(angles[0]), np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                       [0, 1, 0],
                       [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                       [np.sin(angles[2]), np.cos(angles[2]), 0],
                       [0, 0, 1]])
        R = np.dot(Rz, np.dot(Ry, Rx))
        shape_pc = batch_data[k, :, 0:3]
        shape_normal = batch_data[k, :, 3:6]
        rotated_data[k, :, 0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k, :, 3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """
    Randomly jitter points. jittering is per point.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data


def shift_point_cloud(batch_data, shift_range=0.1):
    """
    Randomly shift point cloud. Shift is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B, 3))
    for batch_index in range(B):
        batch_data[batch_index, :, :] += shifts[batch_index, :]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """
    Randomly scale the point cloud. Scale is per point cloud.
    Input:
      BxNx3 array, original batch of point clouds
    Return:
      BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index, :, :] *= scales[batch_index]
    return batch_data


def shuffle_points(batch_data):
    """
    Shuffle orders of points in each point cloud -- changes FPS behavior.
    Use the same shuffling idx for the entire batch.
    Input:
      BxNxC array
    Output:
      BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:, idx, :]


def random_point_dropout(batch_pc, labels=None, max_dropout_ratio=0.875):
    """
    batch_pc: BxNx3
    labels: [optional] BxN, per point label
    """
    for b in range(batch_pc.shape[0]):
        dropout_ratio = np.random.random() * max_dropout_ratio
        drop_idx = np.where(np.random.random((batch_pc.shape[1])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            batch_pc[b, drop_idx, :] = batch_pc[b, 0, :]  # set to the first point
            if labels is not None:
                labels[b, drop_idx] = labels[b, 0]
    if labels is not None:
        return batch_pc, labels
    else:
        return batch_pc


def point_cloud_label_to_surface_voxel_label(point_cloud, label, res=0.02):
    """
    point cloud to voxel
    Input:
        point_cloud: Nx3
        label: N, or Nx2
    Output:
        uvidx: keep ids when converting to voxel, (M,)
        uvlabel: labels of the kept indices, (M,) or (M,2)
    """
    coordmax = np.max(point_cloud, axis=0)
    coordmin = np.min(point_cloud, axis=0)
    nvox = np.ceil((coordmax - coordmin) / res)
    vidx = np.ceil((point_cloud - coordmin) / res)
    vidx = vidx[:, 0] + vidx[:, 1] * nvox[0] + vidx[:, 2] * nvox[0] * nvox[1]
    uvidx, vpidx = np.unique(vidx, return_index=True)
    uvlabel = label[vpidx]
    return uvidx, uvlabel


def point_cloud_label_to_surface_voxel_label_major(point_cloud, label, res=0.02):
    """
    point cloud to voxel
    Input:
        point_cloud: Nx3
        label: N, or Nx2
    Output:
        uvidx: keep ids when converting to voxel, (M,)
        uvlabel: labels of the kept indices, (M,) or (M,2)
    """
    coordmax = np.max(point_cloud, axis=0)
    coordmin = np.min(point_cloud, axis=0)
    nvox = np.ceil((coordmax - coordmin) / res)
    vidx = np.ceil((point_cloud - coordmin) / res)
    vidx = vidx[:, 0] + vidx[:, 1] * nvox[0] + vidx[:, 2] * nvox[0] * nvox[1]
    vidx_label = np.concatenate((vidx[:, None], label), axis=-1)
    vidx_label = vidx_label[np.argsort(vidx), :]
    label_lst = np.split(vidx_label[:, 1:], np.cumsum(np.unique(vidx_label[:, 0], return_counts=True)[1])[:-1])
    # print(label_lst[0])
    # print(label_lst[1])
    # print(label_lst[2])
    # print(label_lst[3])
    # print(label_lst[4])
    # print(label_lst[5])
    # print(label_lst[6])
    # print(label_lst[7])
    label_lst_maxcount = [[np.unique(labels[:, 0], return_counts=True), np.unique(labels[:, 1], return_counts=True)] for
                          labels in label_lst]
    # print(label_lst_maxcount[0][0][0],label_lst_maxcount[0][0][1] )
    # print(label_lst_maxcount[1][0][0],label_lst_maxcount[1][0][1] )
    # print(label_lst_maxcount[2][0][0],label_lst_maxcount[2][0][1] )
    # print(label_lst_maxcount[3][0][0],label_lst_maxcount[3][0][1] )
    # print(label_lst_maxcount[4][0][0],label_lst_maxcount[4][0][1] )
    # print(label_lst_maxcount[5][0][0],label_lst_maxcount[5][0][1] )
    # print(label_lst_maxcount[5][0][0],label_lst_maxcount[6][0][1] )
    # print(label_lst_maxcount[6][0][0],label_lst_maxcount[7][0][1] )
    uvlabel = np.asarray([[label_amounts[0][0][np.argmax(label_amounts[0][1])],
                           label_amounts[1][0][np.argmax(label_amounts[1][1])]] for label_amounts in
                          label_lst_maxcount], dtype=np.int32)
    # print("shape:",uvlabel.shape)
    # print(uvlabel[:4,:])
    return None, uvlabel


def draw_point_cloud(data, highlight=[], title=''):
    """
    Input: Nx3 numpy array
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 2], data[:, 1], s=6)
    for h in highlight:
        ax.scatter(h[:, 0], h[:, 2], h[:, 1], s=20)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    plt.title(title)
    plt.show()


def _draw_point_cloud_on_axe(ax, data, label):
    all_label = np.unique(label)
    for l in all_label:
        x = data[label == l]
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], label=l)
    ax.legend()


def draw_point_cloud_with_labels(data, label, subsample=None, title=''):
    """
    data: Nx3 numpy array
    label: N, numpy array
    """
    if subsample is not None:
        ids = np.random.choice(data.shape[0], subsample, replace=False)
        data = data[ids]
        label = label[ids]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    _draw_point_cloud_on_axe(ax, data, label)
    plt.title(title)
    plt.show()


def draw_point_cloud_with_labels_compare(data, label1, label2, subsample=None):
    if subsample is not None:
        ids = np.random.choice(data.shape[0], subsample, replace=False)
        data = data[ids]
        label1 = label1[ids]
        label2 = label2[ids]
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    _draw_point_cloud_on_axe(ax, data, label1)
    ax = fig.add_subplot(122, projection='3d')
    _draw_point_cloud_on_axe(ax, data, label2)
    plt.show()


# DEPRECATED, use batch_take instead
def get_index_transformer(shape):
    """
    Get the index transformer symbol
    The returned symbol can be added to `index` for Symbol.pick
    shape: (B, N, M) or (B1, B2, N, M)
    Returns: (B, M), or (B1, B2, M), Symbol
    """
    if len(shape) == 3:
        B, N, M = shape
        i = mx.symbol.arange(B, repeat=M, dtype=np.int32).reshape((B, M))
        return i * N
    elif len(shape) == 4:
        B1, B2, N, M = shape
        i = mx.symbol.arange(B2, repeat=M, dtype=np.int32).tile(B1) * N
        j = mx.symbol.arange(B1, repeat=M * B2, dtype=np.int32) * (N * B2)
        i = (i + j).reshape((B1, B2, M))
        return i
    else:
        raise NotImplementedError


class Timer(object):
    def __init__(self):
        self.reset()

    def tic(self):
        self.start = time.time()

    def toc(self):
        self.time += time.time() - self.start
        self.count += 1

    def get(self):
        return self.time / self.count

    def reset(self):
        self.time = 0
        self.count = 0


# def get_git_hash():
#     return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()

def get_timestamp():
    return datetime.now().strftime('%Y_%m_%d_%H:%M:%S')


def gauss_clip(mu, sigma, clip):
    v = random.gauss(mu, sigma)
    v = max(min(v, mu + clip * sigma), mu - clip * sigma)
    return v


def uniform(bound):
    return bound * (2 * random.random() - 1)


def scaling_factor(scaling_param, method):

    if method == 'g':
        return gauss_clip(1.0, scaling_param, 3)
    elif method == 'u':
        return 1.0 + uniform(scaling_param)
    elif method == 'ig':
        return 1.0 - abs(gauss_clip(0.0, scaling_param, 3))


def rotation_angle(rotation_param, method):

    if method == 'g':
        return gauss_clip(0.0, rotation_param, 3)
    elif method == 'u':
        return uniform(rotation_param)

