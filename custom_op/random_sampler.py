"""
random sampler (select n numbers from [0,total))
Input: None
Output: B x N, dtype=int32, indices
"""

import mxnet as mx
import numpy as np


class RandomSampler(mx.operator.CustomOp):
    def __init__(self, total, B, N, replace=False):
        self.total = total
        self.B = B
        self.N = N
        self.replace = replace

    def forward(self, is_train, req, in_data, out_data, aux):
        ids = np.zeros((self.B, self.N))
        for i in xrange(self.B):
            ids[i] = np.random.choice(self.total, self.N, replace=self.replace)
        self.assign(out_data[0], req[0], mx.nd.array(ids))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        pass


@mx.operator.register("random_sampler")
class FarthestPointSamplerProp(mx.operator.CustomOpProp):
    def __init__(self, total, B, N, replace=False):
        super(FarthestPointSamplerProp, self).__init__(need_top_grad=False)
        self.total = int(total)
        self.B = int(B)
        self.N = int(N)
        self.replace = bool(replace)

    def list_arguments(self):
        return []

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return [], [(self.B, self.N)], []

    def infer_type(self, in_type):
        return [], [np.int32], []

    def create_operator(self, ctx, shapes, dtypes):
        return RandomSampler(self.total, self.B, self.N, self.replace)


if __name__ == "__main__":
    inputs = mx.nd.random.uniform(0, 1, shape=(2, 100, 3))
    data = mx.symbol.Variable('data')
    ids = mx.symbol.Custom(total=100, B=2, N=5, name='output', op_type='random_sampler')
    import _init_paths
    from utils.ops import index_op
    output = index_op(data, ids, shape=(2, 100, 5, 3))
    output = mx.symbol.Group([ids, output])
    context = mx.cpu()
    mod = mx.mod.Module(output, data_names=['data'], label_names=[], context=context)
    mod.bind(data_shapes=[('data', inputs.shape)])
    mod.init_params()
    mod.forward(mx.io.DataBatch([inputs]))
    ids = mod.get_outputs()[0].asnumpy()
    print(ids)
    result = mod.get_outputs()[1].asnumpy()
    print(result)

