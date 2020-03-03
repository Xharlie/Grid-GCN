/*!
 * Copyright (c) 2017 by Contributors
 * \file three_nn.cu
 * \brief find three nearest points
 * \author Jianlin Liu
*/
#include "./three_nn-inl.h"
#include "../../common/cuda_utils.h"

namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_ThreeNN)
.set_attr<FCompute>("FCompute<gpu>", ThreeNNForward<gpu>);

}
}