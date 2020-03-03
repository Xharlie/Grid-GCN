"""
Weighted Gradient
Add weight to certain classes in softmax loss calculation
"""
import mxnet as mx
import numpy as np


class WeightedGradient(mx.operator.CustomOp):

    def __init__(self, weight, input_dim=4, axis=1):
        expand_shape = [1] * (input_dim-1)
        expand_shape.insert(axis, -1)
        self.weight = mx.nd.array(weight).reshape(expand_shape)
        self.input_dim = input_dim
        self.axis = axis

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.weight = self.weight.as_in_context(in_data[0].context)
        one_hot = (out_grad[0] < 0) * self.weight
        one_hot = mx.nd.max(one_hot, axis=self.axis, keepdims=True)
        grad = one_hot * out_grad[0]
        self.assign(in_grad[0], req[0], grad)

@mx.operator.register("weighted_gradient")
class WeightedGradientProp(mx.operator.CustomOpProp):

    def __init__(self, weight, input_dim=4, axis=1):
        super(WeightedGradientProp, self).__init__(need_top_grad=True)
        self.weight = eval(weight)
        self.input_dim = int(input_dim)
        self.axis = int(axis)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        assert len(in_shape[0]) == self.input_dim and len(self.weight) == in_shape[0][self.axis]
        return [in_shape[0]], [in_shape[0]], []

    def infer_type(self, in_type):
        dtype = in_type[0]
        return [dtype], [dtype], []

    def create_operator(self, ctx, shapes, dtypes):
        return WeightedGradient(self.weight, self.input_dim, self.axis)

