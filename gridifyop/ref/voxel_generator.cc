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
 * \file voxel_generator.cc
 * \author xiangchen.zhao
*/
#include "./voxel_generator-inl.h"

using namespace std;

namespace mxnet {
namespace op {
std::ostream& operator<<(std::ostream& os, const VoxelGeneratorParam& m) {
  os << "max_points:" << m.max_points
     << "\nmax_voxels:" << m.max_voxels
     // << "\n_batch:" << m.n_batch << ", n_point_per_batch:" << n_point_per_batch
     << "\nheight:" << m.height << ", width:" << m.width
     << "\ncoors_range " << m.coors_range
     << "\nvoxel_size " << m.voxel_size << std::endl;
  return os;
}
}; // op
}; // mxnet

namespace mshadow {
template<typename Dtype>
inline void VoxelGeneratorForward(Tensor<cpu, 4, Dtype> &voxels,
                                  Tensor<cpu, 3, Dtype> &coors,
                                  Tensor<cpu, 2, Dtype> &coor_to_voxelidx,
                                  Tensor<cpu, 2, Dtype> &num_points_per_voxel,
                                  Tensor<cpu, 1, Dtype> &actual_voxel_num,
                                  const Tensor<cpu, 3, Dtype> &data,
                                  const Tensor<cpu, 1, Dtype> &actual_points,
                                  const mxnet::op::VoxelGeneratorParam &param
                                  ) {
  std::cout << "VoxelGeneratorForward<cpu>" << std::endl;
  std::cout << param;
//  const Dtype *in_data = data.dptr_;
//  const Dtype *in_actual_points = actual_points.dptr_;
//  Dtype *out_voxels = voxels.dptr_;
//  Dtype *out_coors = coors.dptr_;
//  Dtype *out_coor_to_voxelidx = coor_to_voxelidx.dptr_;
//  Dtype *out_num_points_per_voxel = num_points_per_voxel.dptr_;
//  Dtype *out_actual_voxel_num = actual_voxel_num.dptr_;
  const int batch_size_ = data.size(0);
  const int ndim = 3; // 3d voxel
  const int data_dim = data.size(2);
  const int coor_dim = coors.size(2);
  int *coor = new int[ndim];
  int *grid_size = new int[ndim];

  for (int k = 0; k < ndim; ++k) {
      grid_size[k] = round((param.coors_range[k+ndim] - param.coors_range[k]) / param.voxel_size[k]);
  }

  for (int idx = 0; idx < batch_size_; ++idx) {
    //points_to_voxel
    int voxel_num = 0;

    for (int i = 0; i < (int)actual_points[idx]; ++i) {
      bool failed = false;

      for (int j = 0; j < ndim; ++j) {
        int c = floor((data[idx][i][j] - param.coors_range[j]) / param.voxel_size[j]);
        if (c < 0 || c >= grid_size[j]) {
          failed = true;
          break;
        }
        coor[ndim-1-j] = c;
      }
      if (failed) continue;
      int coor_index = coor[0] * (param.height * param.width) + coor[1] * param.width + coor[2];
      int voxelidx = coor_to_voxelidx[idx][coor_index];
      if (voxelidx == param.max_voxels) {
        voxelidx = voxel_num;
        if (voxel_num >= param.max_voxels) break;
        voxel_num++;
        coor_to_voxelidx[idx][coor_index] = voxelidx;
        for (int k = 0; k < ndim; ++k){
          coors[idx][voxelidx][k] = coor[k];
        }
      }
      int num = num_points_per_voxel[idx][voxelidx];
      if (num < param.max_points) {
        memcpy(voxels[idx][voxelidx][num].dptr_, data[idx][i].dptr_, data_dim * sizeof(Dtype));
        num_points_per_voxel[idx][voxelidx] += 1;
      }
    }
    actual_voxel_num[idx] = voxel_num;
  }

  delete[] coor;
  delete[] grid_size;
}
}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(VoxelGeneratorParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new VoxelGeneratorOp<cpu, DType>(param);
  });
  return op;
}

// template<>
// Operator *CreateOp<gpu>(VoxelGeneratorParam param, int dtype) {
//   Operator* op = NULL;
//   MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
//     op = new VoxelGeneratorOp<gpu, DType>(param);
//   });
//   return op;
// }


Operator *VoxelGeneratorProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                           std::vector<int> *in_type) const {
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(VoxelGeneratorParam);

MXNET_REGISTER_OP_PROPERTY(VoxelGenerator, VoxelGeneratorProp)
.describe("VoxelGenerator")
.add_argument("data","Symbol", "point cloud data")
.add_argument("actualnum", "Symbol", "number of points")
.add_arguments(VoxelGeneratorParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
