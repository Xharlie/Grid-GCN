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
#include "./gridify-inl.h"


namespace mshadow {
    template<typename Dtype>
    inline void GridifyForward(Tensor<cpu, 3, int> &nebidx,
                               Tensor<cpu, 3, Dtype> &nebidxmsk,
                               Tensor<cpu, 3, Dtype> &cent,
                               Tensor<cpu, 2, Dtype> &centmsk,
                               Tensor<cpu, 2, int> &actual_centnum,
                               const Tensor<cpu, 3, Dtype> &data,
                               const Tensor<cpu, 2, int> &actual_numpoints,
                               const mxnet::op::GridifyParam &param) {
        LOG(FATAL) << "Not Implemented.";
    }

}  // namespace mshadow

namespace mxnet {
    namespace op {

        template<>
        Operator *CreateOp<cpu>(GridifyParam param, int dtype) {
            Operator* op = NULL;
            MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
                    op = new GridifyOp<cpu, DType>(param);
            });
            return op;
        }


        Operator *GridifyProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                      std::vector<int> *in_type) const {
            DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
        }

        DMLC_REGISTER_PARAMETER(GridifyParam);

        MXNET_REGISTER_OP_PROPERTY(Gridify, GridifyProp)
        .describe("Gridify")
        .add_argument("data","Symbol", "data")
        .add_argument("actual_numpoints","Symbol", "actual_numpoints")
        .add_arguments(GridifyParam::__FIELDS__());
    }  // namespace op
}  // namespace mxnet
