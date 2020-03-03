import mxnet as mx
import numpy as np

from configs.configs import configs

use_global_stats = configs['use_global_stats']
if 'relutp' not in configs.keys() or configs['relutp'] == "relu":
    relu = mx.symbol.relu
else:
    print("use leaky relu")
    relu = mx.symbol.LeakyReLU

# DEPRECATED. All previous usages have been replaced by broadcast operations or batch_take.
def expand_and_tile(data, axis, rep, ndim=3):
    """
    expand_dim and tile
    axis: int, axis to expand_dim
    rep: rep to tile
    ndim: ndim of data
    """
    if data is None: return None
    reps = [1] * ndim
    reps.insert(axis, rep)
    return data.expand_dims(axis=axis).tile(reps=reps)


def batch_take(data, index, shape, scope=''):
    """
    Per-batch take operator. Perform on axis 1
    data:  symbol, shape=(B, N, C)
    index: symbol, shape=(B, ...)
    shape: specify shape (B, ..., N, C)
    scope: name scope for this operator
    returns: symbol, shape=(B, ..., C)
    """
    B, N, C = shape[0], shape[-2], shape[-1]
    M = int(np.prod(shape[1:-2]))
    data = mx.symbol.reshape(data, shape=(B*N, C), name=scope+'_pretake_reshape')
    index_transformer = mx.symbol.arange(B, repeat=M, dtype=np.int32).reshape(shape[:-2]) * N
    index = index + index_transformer
    outputs = mx.symbol.take(data, index, name=scope+'_take')
    return outputs

def batch_take_halfc(data, index, shape, scope=''):
    """
    Per-batch take operator. Perform on axis 1
    data:  symbol, shape=(B, N, C)
    index: symbol, shape=(B, ...)
    shape: specify shape (B, ..., N, C)
    scope: name scope for this operator
    returns: symbol, shape=(B, ..., C)
    """
    B, N, C = shape[0], shape[-2], shape[-1]
    M = int(np.prod(shape[1:-2]))
    data = mx.symbol.reshape(data, shape=(B*N, C), name=scope+'_pretake_reshape')
    index_transformer = mx.symbol.arange(B, repeat=M, dtype=np.int32).reshape(shape[:-2]) * N
    index = index + index_transformer
    outputs = mx.symbol.take(data, index, name=scope+'_take')
    return outputs.transpose(axes=(0,3,1,2), name=scope+"/batch_take_c_tpose2")

def batch_take_c(data, index, shape, scope=''):
    """
    Per-batch take operator. Perform on axis 1
    data:  symbol, shape=(B, C, N)
    index: symbol, shape=(B, ...)
    shape: specify shape (B, ..., N, C)
    scope: name scope for this operator
    returns: symbol, shape=(B, C...)
    """
    B, N, C = shape[0], shape[-2], shape[-1]
    M = int(np.prod(shape[1:-2]))
    data = mx.symbol.reshape(data.transpose(axes=(0,2,1), name=scope+"/batch_take_c_tpose1"), shape=(B*N, C), name=scope+'_pretake_reshape')
    index_transformer = mx.symbol.arange(B, repeat=M, dtype=np.int32).reshape(shape[:-2]) * N
    index = index + index_transformer
    outputs = mx.symbol.take(data, index, name=scope+'_take')
    return outputs.transpose(axes=(0,3,1,2), name=scope+"/batch_take_c_tpose2")

def batch_take_g(data, index, shape, scope=''):
    """
    Per-batch take operator. Perform on axis 1
    data:  symbol, shape=(B, N, C)
    index: symbol, shape=(B, ...)
    shape: specify shape (B, ..., N, C)
    scope: name scope for this operator
    returns: symbol, shape=(B, ..., C)
    """
    B, N, C = shape[0], shape[-2], shape[-1]
    M = int(np.prod(shape[1:-2]))
    data = mx.symbol.reshape(data, shape=(B*N, C), name=scope+'_pretake_reshape')
    index_transformer = mx.symbol.arange(B, repeat=M, dtype=np.int32).reshape(shape[:-2]) * N
    index = index + index_transformer
    outputs = mx.symbol.take(data, index, name=scope+'_take')
    return outputs


def batch_take_g_cnt(data, index, shape, scope='', cnt_ratio=1):
    """
    Per-batch take operator. Perform on axis 1
    data:  symbol, shape=(B, N, C)
    index: symbol, shape=(B, ...)
    shape: specify shape (B, ..., N, C)
    scope: name scope for this operator
    returns: symbol, shape=(B, ..., C)
    """
    B, O, P, N, C = shape[0], shape[1], shape[2], shape[3], shape[4]
    M = int(np.prod(shape[1:-2]))
    data = mx.symbol.reshape(data, shape=(B*N, C), name=scope+'_pretake_reshape')
    index_transformer = mx.symbol.arange(B, repeat=M, dtype=np.int32).reshape(shape[:-2]) * N
    index = index + index_transformer
    outputs = mx.symbol.take(data, index, name=scope+'_take')
    if cnt_ratio != 1:
        print("cnt_ratio", cnt_ratio)
        real_index = mx.symbol.arange(0, P, step = cnt_ratio ,dtype=np.int32)
        real_outputs = mx.symbol.take(outputs, real_index, axis=-2)
    else:
        real_outputs = outputs
        outputs = None
    return real_outputs, outputs


def knn(query, data, shape, k=3, scope=''):
    """
    k nearest neighbor
    query: symbol, shape=(B, M, 3), query points
    data: symbol, shape=(B, N, 3), all points
    shape: specify (M, N)
    scope: name scope for this operator
    returns:
        dist, shape=(B, M, k), squared distances to the k nearest points
        ids, shape=(B, M, k), dtype=int32, indices of the k nearest points
    """
    M, N = shape
    query = query.expand_dims(axis=2)  # (B, M, 1, 3)
    data = data.expand_dims(axis=1)  # (B, 1, N, 3)
    diff = mx.symbol.broadcast_sub(query, data)  # (B, M, N, 3)
    all_dist = mx.symbol.sum(diff*diff, axis=3)  # (B, M, N)
    dist, ids = mx.symbol.topk(all_dist, axis=2, k=k, ret_typ='both', is_ascend=True, name='{}_top{}'.format(scope, k))   # (B, M, k)
    return dist, ids.astype('int32')


def conv1d(inputs, num_filter, kernel=(3,), stride=(1,), bn_decay=0.9, layout='NCW', use_bn=True, use_relu=True, attr=None, scope='', bn_axis=1):
    net = mx.symbol.Convolution(data=inputs, num_filter=num_filter, kernel=kernel, stride=stride, layout=layout, attr=attr, name=scope, cudnn_tune="fastest")
    if use_bn:
        net = mx.symbol.BatchNorm(data=net, momentum=bn_decay, axis=bn_axis, fix_gamma=False, use_global_stats=use_global_stats, attr=attr, name=scope+'/bn')
    if use_relu:
        net = relu(data=net, name=scope+'/relu')
    return net

def conv2d(inputs, num_filter, kernel=(3,3), stride=(1,1), bn_decay=0.9, layout='NCHW', use_bn=True, use_relu=True, attr=None, scope='', bn_axis=1, relutyp="relu"):
    net = mx.symbol.Convolution(data=inputs, num_filter=num_filter, kernel=kernel, stride=stride, layout=layout, workspace=4096, attr=attr, name=scope, cudnn_tune="fastest")
    if use_bn:
        net = mx.symbol.BatchNorm(data=net, momentum=bn_decay, axis=bn_axis, fix_gamma=False, use_global_stats=use_global_stats, attr=attr, name=scope+'/bn')
    if use_relu:
        if relutyp == "relu":
            net = relu(data=net, name=scope+'/relu')
        elif relutyp == "leaky":
            net = mx.symbol.LeakyReLU(data=net, name=scope+'/leakyrelu')
    return net

def depthwise_conv2d(inputs, in_channels, depth_multiplier=1, kernel=(3,3), stride=(1,1), bn_decay=0.9, layout='NCHW', use_bn=True, use_relu=True, attr=None, scope=''):
    net = mx.symbol.Convolution(data=inputs, num_filter=in_channels*depth_multiplier, num_group=in_channels, kernel=kernel, stride=stride, layout=layout, attr=attr, name=scope)
    if use_bn:
        axis = 3 if layout == 'NHWC' else 1
        net = mx.symbol.BatchNorm(data=net, momentum=bn_decay, axis=axis, fix_gamma=False, use_global_stats=use_global_stats, attr=attr, name=scope+'/bn')
    if use_relu:
        net = relu(data=net, name=scope+'/relu')
    return net

def separable_conv2d(inputs, in_channels, num_filter, depth_multiplier=1, kernel=(3,3), stride=(1,1), bn_decay=0.9, layout='NCHW', use_bn=True, use_relu=True, attr=None, scope=''):
    net = mx.symbol.Convolution(data=inputs, num_filter=in_channels*depth_multiplier, num_group=in_channels, kernel=kernel, stride=stride, layout=layout, attr=attr, name=scope+'/depth')
    net = mx.symbol.Convolution(data=net, num_filter=num_filter, kernel=(1,1), stride=(1,1), layout=layout, attr=attr, name=scope+'/1x1')
    if use_bn:
        axis = 3 if layout == 'NHWC' else 1
        net = mx.symbol.BatchNorm(data=net, momentum=bn_decay, axis=axis, fix_gamma=False, use_global_stats=use_global_stats, attr=attr, name=scope+'/bn')
    if use_relu:
        net = relu(data=net, name=scope+'/relu')
    return net

def fully_connected_mlp(inputs, mlp, dim = 4, bn_decay=0.9, dropout_ratio=0, flatten=False, use_bn=True, use_relu=True, bn_axis=1, attr=None, scope='', flip=False):
    for i, num_hidden in enumerate(mlp):
        data = fully_connected(inputs, num_hidden, bn_decay=0.9, dropout_ratio=0,
            flatten=flatten, use_bn=use_bn, use_relu=use_relu, bn_axis=bn_axis, attr=None, scope=scope+"/"+str(i), flip=flip)
    return data

def fully_connected_mlp_withbn(inputs, mlp, dim = 4, bn_decay=0.9, dropout_ratio=0, flatten=False, use_bn=True, use_relu=True, bn_axis=1, attr=None, scope='', flip=False):
    for i, num_hidden in enumerate(mlp):
        data = fully_connected(inputs, num_hidden, bn_decay=bn_decay, dropout_ratio=0,
            flatten=flatten, use_bn=use_bn, use_relu=use_relu, bn_axis=bn_axis, attr=None, scope=scope+"/"+str(i), flip=flip)
    return bn_transpose(data, bn_decay=bn_decay, dim=dim, scope=scope+"/lastbn")

def bn_transpose(data, bn_decay=0.9, dim=4, scope=''):
    if dim == 3:
        data = mx.symbol.transpose(data, axes=(0, 2, 1), name=scope + "/bntranspose1")  # change to B C W
    else:
        data = mx.symbol.transpose(data, axes=(0, 3, 1, 2), name=scope + "/bntranspose1")  # change to B C H W
    data = mx.symbol.BatchNorm(data, momentum=bn_decay, axis=1, fix_gamma=False,
                               use_global_stats=use_global_stats, name=scope + '/bn')
    if dim == 3:
        data = mx.symbol.transpose(data, axes=(0, 2, 1), name=scope + "/bntranspose2")  # change to B C W
    else:
        data = mx.symbol.transpose(data, axes=(0, 2, 3, 1), name=scope + "/bntranspose2")
    return data

def fully_connected(inputs, num_hidden, bn_decay=0.9, dropout_ratio=0, flatten=False, use_bn=True, use_relu=True, bn_axis=1, attr=None, scope='', flip=False):
    net = mx.symbol.FullyConnected(data=inputs, num_hidden=num_hidden, flatten=flatten, name=scope)
    if use_bn:
        if flip: net = mx.symbol.transpose(net, axes=(0,3,1,2), name=scope+"/bntranspose1") # change to B C H W
        net = mx.symbol.BatchNorm(net, momentum=bn_decay, axis=bn_axis, fix_gamma=False, use_global_stats=use_global_stats, name=scope+'/bn')
        if flip: net = mx.symbol.transpose(net, axes=(0,2,3,1), name=scope+"/bntranspose2")
    if use_relu:
        net = relu(net, name=scope+'/relu')
    if dropout_ratio > 0:
        net = mx.symbol.Dropout(net, p=dropout_ratio, name=scope+'/dropout')
    return net

def fully_connected_c(inputs, num_hidden, bn_decay=0.9, dropout_ratio=0, flatten=False, use_bn=True, use_relu=True, bn_axis=1, attr=None, scope='', flip=False):
    net = mx.symbol.FullyConnected(data=inputs, num_hidden=num_hidden, flatten=flatten, name=scope)
    if use_bn:
        net = mx.symbol.BatchNorm(net, momentum=bn_decay, axis=bn_axis, fix_gamma=False, use_global_stats=use_global_stats, name=scope+'/bn')
    if use_relu:
        net = relu(net, name=scope+'/relu')
    if dropout_ratio > 0:
        net = mx.symbol.Dropout(net, p=dropout_ratio, name=scope+'/dropout')
    return net

def mlp1d(data, mlp, use_bn=True, bn_decay=0.9, attr=None, scope=''):
    # NWC format is not supported by MXNet. So I change the format to NCW here
    data = mx.symbol.transpose(data, axes=(0,2,1))
    for i, out_size in enumerate(mlp):
        data = conv1d(data, num_filter=out_size, kernel=(1,), stride=(1,),
                use_bn=use_bn, bn_decay=bn_decay, layout='NCW',
                attr=attr, scope='{}/conv{}'.format(scope, i+1))
    data = mx.symbol.transpose(data, axes=(0,2,1))
    return data

def mlp1d_c(data, mlp, use_bn=True, bn_decay=0.9, attr=None, scope=''):
    # NWC format is not supported by MXNet. So I change the format to NCW here
    for i, out_size in enumerate(mlp):
        data = conv1d(data, num_filter=out_size, kernel=(1,), stride=(1,),
                use_bn=use_bn, bn_decay=bn_decay, layout='NCW',
                attr=attr, scope='{}/conv{}'.format(scope, i+1), bn_axis=1)
    return data

def mlp2d(data, mlp, use_bn=True, bn_decay=0.9, attr=None, scope=''):
    # NHWC format may not be well-supported for batchnorm operation. So I change the format to NCHW here
    data = mx.symbol.transpose(data, axes=(0,3,1,2), name=scope+"test1")
    for i, out_size in enumerate(mlp):
        data = conv2d(data, num_filter=out_size, kernel=(1,1), stride=(1,1),
                use_bn=use_bn, bn_decay=bn_decay, layout='NCHW',
                attr=attr, scope='{}/conv{}'.format(scope, i+1))
    data = mx.symbol.transpose(data, axes=(0,2,3,1), name="test2")
    return data

def mlp2d_c(data, mlp, use_bn=True, bn_decay=0.9, attr=None, scope='', relutyp="relu"):
    # NHWC format may not be well-supported for batchnorm operation. So I change the format to NCHW here
    for i, out_size in enumerate(mlp):
        data = conv2d(data, num_filter=out_size, kernel=(1,1), stride=(1,1),
                use_bn=use_bn, bn_decay=bn_decay, layout='NCHW',
                attr=attr, scope='{}/conv{}'.format(scope, i+1), bn_axis=1,relutyp=relutyp)
    return data

if __name__ == "__main__":
    context = mx.cpu()

    '''
    # test index_op for 3d
    input_data = mx.nd.array([[[1,2,3],[3,4,5],[5,6,7],[7,8,9]],[[6,7,8],[8,9,10],[10,11,12],[12,13,14]]])
    index_data = mx.nd.array([[1,2,3],[0,1,3]], dtype=np.int32)
    data = mx.symbol.Variable('data')
    index = mx.symbol.Variable('index')
    output = index_op(data, index, shape=(2,4,3,3), scope='test')
    #t = mx.viz.plot_network(output, shape={'data': input_data.shape, 'index': index_data.shape})
    #t.render()
    mod = mx.mod.Module(output, data_names=['data', 'index'], label_names=[], context=context)
    mod.bind(data_shapes=[mx.io.DataDesc('data', input_data.shape, np.float32), mx.io.DataDesc('index', index_data.shape, np.int32)])
    mod.init_params()
    mod.forward(mx.io.DataBatch([input_data, index_data]))
    print mod.get_outputs()[0].asnumpy()
    '''

    # test knn
    inputs = []
    for i in xrange(3):
        for j in xrange(3):
            for k in xrange(3):
                inputs.append([i,j,k])
    inputs = mx.nd.array(inputs)
    inputs = mx.nd.stack(inputs, inputs*2, axis=0)  # (2, 27, 3)
    query_data = inputs[:, 13:14, :]  # (2, 1, 3)
    query = mx.symbol.Variable('query')
    data = mx.symbol.Variable('data')
    dist, ids = knn(query, data, shape=(1, 27), k=8)
    output = mx.symbol.Group([dist, ids])
    mod = mx.mod.Module(output, data_names=['query', 'data'], label_names=[], context=context)
    mod.bind(data_shapes=[('query', query_data.shape), ('data', inputs.shape)])
    mod.init_params()
    mod.forward(mx.io.DataBatch([query_data, inputs]))
    print(mod.get_outputs()[0].asnumpy())
    print(mod.get_outputs()[1].asnumpy())


