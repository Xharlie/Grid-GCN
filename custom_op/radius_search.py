"""
Radius search
Input:
    B x M x 3, centeroids
    B x N x 3, all points
Output:
    B x M x npoints, indices of radius search results

If there are less than npoints within the radius, pad with the first value
so that the shape is always the same.
"""

import mxnet as mx
import numpy as np

from configs.configs import configs

CC = configs['num_channel']


class RadiusSearch(mx.operator.CustomOp):
    def __init__(self, radius, npoints):
        self.radius = radius
        self.npoints = npoints

    def forward(self, is_train, req, in_data, out_data, aux):
        centeroids = in_data[0].asnumpy()
        data = in_data[1].asnumpy()
        batch_size, num_centeroids, c = centeroids.shape
        b, total_points, cc = data.shape
        assert batch_size == b, "Both inputs should have the same batch size"
        assert c == cc == CC, "Both inputs should be XYZ data, or XYZ+normal data"

        def cal_dist(i, j):
            """ calculate the distance of each point in `data`, with point j in `centeroids`, in batch i """
            diff = data[i,:,:3] - centeroids[i,j::num_centeroids,:3]
            return np.sqrt(np.sum(diff*diff, axis=1))

        # results are stored in ids
        ids = np.zeros((batch_size, num_centeroids, self.npoints), dtype=np.int32)
        for i in range(batch_size):
            for j in range(num_centeroids):
                dist = cal_dist(i, j)
                valid_ids = np.where(dist < self.radius)[0]
                x = valid_ids.size
                if x == 0:
                    k = np.argmin(dist)  # if no point is within radius, find the closest one
                    ids[i,j,:] = k
                elif x > self.npoints:
                    ids[i,j,:] = valid_ids[:self.npoints]  # keep only the first npoints
                else:
                    ids[i,j,:x] = valid_ids
                    ids[i,j,x:self.npoints] = valid_ids[0]  # pad with valid_ids[0]

        self.assign(out_data[0], req[0], mx.nd.array(ids))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("radius_search")
class RadiusSearchProp(mx.operator.CustomOpProp):
    def __init__(self, radius, npoints):
        super(RadiusSearchProp, self).__init__(need_top_grad=False)
        self.radius = float(radius)
        self.npoints = int(npoints)

    def list_arguments(self):
        return ['centeroids', 'data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        centeroids_shape = in_shape[0]
        output_shape = (centeroids_shape[0], centeroids_shape[1], self.npoints)
        return in_shape, [output_shape], []

    def infer_type(self, in_type):
        return in_type, [np.int32], []

    def create_operator(self, ctx, shapes, dtypes):
        return RadiusSearch(self.radius, self.npoints)


if __name__ == "__main__":
    inputs = mx.nd.random.uniform(0, 1, shape=(2, 100, 3))
    centers_id = np.random.randint(0, 99, size=(2, 2))
    centers = mx.nd.stack(inputs[0, centers_id[0], :], inputs[1, centers_id[1], :], axis=0)
    centeroids = mx.symbol.Variable('centeroids')
    data = mx.symbol.Variable('data')
    output = mx.symbol.Custom(centeroids=centeroids, data=data, radius=0.2, npoints=5, name='output', op_type='radius_search')
    import _init_paths
    from utils.ops import index_op
    from utils.utils import draw_point_cloud
    data = mx.symbol.reshape(data, (2,1,100,3))
    data = mx.symbol.tile(data, reps=(1,2,1,1))
    output = index_op(data, output, shape=(2,2,100,5,3))
    context = mx.cpu()
    mod = mx.mod.Module(output, data_names=['centeroids', 'data'], label_names=[], context=context)
    mod.bind(data_shapes=[('centeroids', centers.shape), ('data', inputs.shape)])
    mod.init_params()
    mod.forward(mx.io.DataBatch([centers, inputs]))
    result = mod.get_outputs()[0].asnumpy()
    print(result)
    # plot
    for i in range(2):
        data = inputs.asnumpy()[i]
        draw_point_cloud(data, [result[i,0], result[i,1]])
