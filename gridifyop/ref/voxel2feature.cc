/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file voxel2feature.cc
 * \author xiangchen.zhao
*/
#include "./voxel2feature-inl.h"


namespace mshadow {
template<typename Dtype>
inline void Voxel2FeatureForward(Tensor<cpu, 4, Dtype> &feature,
                                 const Tensor<cpu, 4, Dtype> &voxels,
                                 const Tensor<cpu, 3, Dtype> &coors,
                                 const Tensor<cpu, 2, Dtype> &num_points_per_voxel,
                                 const Tensor<cpu, 1, Dtype> &actual_voxel_num,
                                 const mxnet::op::Voxel2FeatureParam &param) {
  LOG(FATAL) << "Not Implemented.";
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(Voxel2FeatureParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new Voxel2FeatureOp<cpu, DType>(param);
  });
  return op;
}


Operator *Voxel2FeatureProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                               std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(Voxel2FeatureParam);

MXNET_REGISTER_OP_PROPERTY(Voxel2Feature, Voxel2FeatureProp)
.describe("Voxel2Feature")
.add_argument("voxels","Symbol", "voxels")
.add_argument("coors","Symbol", "coors")
.add_argument("num_points_per_voxel","Symbol", "num points per voxel")
.add_argument("actual_voxel_num", "Symbol", "actual voxel num")
.add_arguments(Voxel2FeatureParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
