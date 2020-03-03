"""
farthest point sampler
Input: B x N x 3, point clouds
Output: B x npoints, dtype=int32, indices of the result of FPS
"""

import mxnet as mx
import numpy as np

from configs.configs import configs

CC = configs['num_channel']


class FarthestPointSampler(mx.operator.CustomOp):
    def __init__(self, npoints):
        self.npoints = npoints

    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0].asnumpy()
        batch_size, total_points, c = data.shape
        assert c == CC, "need XYZ data or XYZ+normal data"
        assert total_points >= self.npoints, "number of input points should be at least npoints"

        def cal_dist(i, j):
            """ calculate the distance of each point with point j, in batch i """
            diff = data[i,:,:3] - data[i,j::total_points,:3]
            return np.sum(diff*diff, axis=1)

        # results are stored in ids
        ids = np.zeros((batch_size, self.npoints), dtype=np.int32)
        for i in xrange(batch_size):
            # always select the first point
            ids[i,0] = 0
            # min_dist of each point to the selected point set
            min_dist = cal_dist(i, 0)
            for j in xrange(1, self.npoints):
                k = np.argmax(min_dist)
                ids[i,j] = k
                tmp_dist = cal_dist(i, k)
                min_dist = np.minimum(min_dist, tmp_dist)

        self.assign(out_data[0], req[0], mx.nd.array(ids))


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("farthest_point_sampler")
class FarthestPointSamplerProp(mx.operator.CustomOpProp):
    def __init__(self, npoints):
        super(FarthestPointSamplerProp, self).__init__(need_top_grad=False)
        self.npoints = int(npoints)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = (data_shape[0], self.npoints)
        return [data_shape], [output_shape], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype], [np.int32], []

    def create_operator(self, ctx, shapes, dtypes):
        return FarthestPointSampler(self.npoints)


if __name__ == "__main__":
    inputs = mx.nd.random.uniform(0, 1, shape=(2, 100, 3))
    data = mx.symbol.Variable('data')
    output = mx.symbol.Custom(data=data, npoints=5, name='output', op_type='farthest_point_sampler')
    from utils import index_op
    output = index_op(data, output, shape=(2, 100, 5, 3))
    context = mx.cpu()
    mod = mx.mod.Module(output, data_names=['data'], label_names=[], context=context)
    mod.bind(data_shapes=[('data', inputs.shape)])
    mod.init_params()
    mod.forward(mx.io.DataBatch([inputs]))
    result = mod.get_outputs()[0].asnumpy()
    # plot
    import matplotlib.pyplot as plt

    for i in xrange(2):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        data = inputs.asnumpy()[i]
        ax.scatter(data[:,0], data[:,1], data[:,2], s=6)
        ax.scatter(result[i,:,0], result[i,:,1], result[i,:,2], s=20)
        plt.show()

