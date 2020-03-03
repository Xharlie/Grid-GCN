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
 * \file voxel2feature.cu
 * \brief voxel to feature operator
 * \author Xiangchen Zhao
*/
#include "./voxel2feature-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include <stdio.h>


#define ndim 3
#define data_ndim 4

namespace mshadow {
namespace cuda {

template<typename Dtype>
__global__ void Voxel2FeatureForwardKernel(Dtype* feature,
                                           const Dtype* voxels,
                                           const Dtype* coors,
                                           const Dtype* num_points_per_voxel,
                                           const Dtype* actual_voxel_num,
                                           const int max_voxels,
                                           const int max_points,
                                           const int channel,
                                           const bool use_intensity,
                                           const Dtype voxel_size_x,
                                           const Dtype voxel_size_y,
                                           const Dtype coors_range_x,
                                           const Dtype coors_range_y,
                                           const int batch_size) {
  Dtype x_offset = (voxel_size_x / 2) + coors_range_x;
  Dtype y_offset = (voxel_size_y / 2) + coors_range_y;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = index / max_voxels;
  int voxel_idx = index % max_voxels;
//  printf("batch idx: %d, voxel idx: %d \n", batch_idx, voxel_idx);
//  __syncthreads();
//  printf("%f \n", actual_voxel_num[batch_idx]);
  if ((voxel_idx >= static_cast<int>(actual_voxel_num[batch_idx])) || (batch_idx >= batch_size)) {return;}

  Dtype* feature_b_v = feature + (batch_idx * (max_voxels * max_points * channel)) + (voxel_idx * (max_points * channel));
  const Dtype* voxels_b_v = voxels + batch_idx * (max_voxels * max_points * data_ndim) + voxel_idx * (max_points * data_ndim);
  const Dtype* coors_b_v = coors + batch_idx * (max_voxels * ndim) + voxel_idx * ndim;
  const Dtype* num_points_per_voxel_b = num_points_per_voxel + batch_idx * max_voxels;

  Dtype* mean = new Dtype[ndim];
  for (int i = 0; i < ndim; ++i) {mean[i] = 0.f;}

  for (int i = 0; i < max_points; ++i) {
    for (int j = 0; j < ndim; ++j) {
      int postion = i * data_ndim + j;
      mean[j] += voxels_b_v[postion];
    }
  }



  for (int i = 0; i < ndim; ++i){
    mean[i] /= num_points_per_voxel_b[voxel_idx];
  }


  for (int i = 0; i < min(static_cast<int>(num_points_per_voxel_b[voxel_idx]), max_points); ++i) {
    // 0, 1, 2, copy x, y, z value
    feature_b_v[i*channel] = voxels_b_v[i*data_ndim];
    feature_b_v[i*channel+1] = voxels_b_v[i*data_ndim+1];
    feature_b_v[i*channel+2] = voxels_b_v[i*data_ndim+2];
    // 3, intensity
    int sub_idx = 3;
    if (use_intensity) {
      feature_b_v[i*channel+3] = voxels_b_v[i*data_ndim+3];
      sub_idx++;
    }
    // sub_idx, sub_idx+1, +2, cluster
    feature_b_v[i*channel+sub_idx] = voxels_b_v[i*4] - mean[0];
    feature_b_v[i*channel+sub_idx+1] = voxels_b_v[i*4+1] - mean[1];
    feature_b_v[i*channel+sub_idx+2] = voxels_b_v[i*4+2] - mean[2];

    //+3, +4 ,center x,y
    feature_b_v[i*channel+sub_idx+3] = voxels_b_v[i*data_ndim] - (coors_b_v[2] * voxel_size_x + x_offset);
    feature_b_v[i*channel+sub_idx+4] = voxels_b_v[i*data_ndim+1] - (coors_b_v[1] * voxel_size_y + y_offset);

//    if (batch_idx == 0 && voxel_idx == 3 && i == 60) {
//      printf("%f, %f, %f\n", mean_x, mean_y, mean_z);
//      printf("%f, %f\n", coors_b_v[2] * voxel_size_x + x_offset, coors_b_v[1] * voxel_size_y + y_offset);
//      printf("%f, %f, %f, %f, %f, %f, %f, %f\n", feature[0], feature[1], feature[2], feature[4], feature[5], feature[6], feature[7], feature[8]);
//    }
  }
//  __syncthreads();
//  if (batch_idx == 0 && voxel_idx == 0) {
//    printf("%f, %f\n", coors_b_v[2] * voxel_size_x + x_offset, coors_b_v[1] * voxel_size_y + y_offset);
//  }
//    printf("%f, %f, %f\n", feature[3*(max_points * channel) + 60*channel],
//    feature[3*(max_points * channel)+60*channel+1],
//    feature[3*(max_points * channel)+60*channel+2]);
//  printf("%d\n", index);

//  if (temp != feature[1]){
//    printf("%f, %f, %d, %d, %d, \n", temp, feature[1], batch_idx, voxel_idx, (batch_idx * (max_voxels * max_points * channel)) + (voxel_idx * (max_points * channel)));
//  }
  delete[] mean;
  return;


} // kernel

template<typename Dtype>
inline void Voxel2FeatureForward(const Tensor<gpu, 4, Dtype> &feature,
                                 const Tensor<gpu, 4, Dtype> &voxels,
                                 const Tensor<gpu, 3, Dtype> &coors,
                                 const Tensor<gpu, 2, Dtype> &num_points_per_voxel,
                                 const Tensor<gpu, 1, Dtype> &actual_voxel_num,
                                 const mxnet::op::Voxel2FeatureParam &param) {
  const int count = feature.size(0) * feature.size(1);
  const int batch_size = feature.size(0);
  const Dtype voxel_size_x = param.voxel_size[0];
  const Dtype voxel_size_y = param.voxel_size[1];
  const Dtype coors_range_x = param.coors_range[0];
  const Dtype coors_range_y = param.coors_range[1];
  const int max_voxels = param.max_voxels;
  const int max_points = param.max_points;
  const bool use_intensity = param.use_intensity;
  const int channel = param.channel;

  Dtype* out_feature = feature.dptr_;
  const Dtype* in_voxels = voxels.dptr_;
  const Dtype* in_coors = coors.dptr_;
  const Dtype* in_num_points_per_voxel = num_points_per_voxel.dptr_;
  const Dtype* in_actual_voxel_num = actual_voxel_num.dptr_;


  const int gridSize = (count + kMaxThreadsPerBlock - 1) / kMaxThreadsPerBlock;
  dim3 dimGrid(gridSize);
  dim3 dimBlock(kMaxThreadsPerBlock);
  // printf("%d \n", kMaxGridDim);
  // printf("%d \n", kMaxThreadsPerBlock);

  CheckLaunchParam(dimGrid, dimBlock, "Voxel2Feature Forward");
  cudaStream_t stream = Stream<gpu>::GetStream(feature.stream_);
  Voxel2FeatureForwardKernel<Dtype><<<dimGrid, dimBlock, 0, stream>>>(
  out_feature, in_voxels, in_coors, in_num_points_per_voxel, in_actual_voxel_num,
  max_voxels, max_points, channel,  use_intensity,
  voxel_size_x, voxel_size_y, coors_range_x, coors_range_y, batch_size);
  MSHADOW_CUDA_POST_KERNEL_CHECK(Voxel2FeatureForwardKernel);

//  cudaDeviceSynchronize();
//  for (int i= 0; i < 1000; ++i){
//    for (int j=0; j < 9; ++j) {
//      Dtype =
//      printf("%f ", 1.f);
//    }
//    printf("\n");
//  }
}

}  // namespace cuda

template<typename Dtype>
inline void Voxel2FeatureForward(const Tensor<gpu, 4, Dtype> &feature,
                                 const Tensor<gpu, 4, Dtype> &voxels,
                                 const Tensor<gpu, 3, Dtype> &coors,
                                 const Tensor<gpu, 2, Dtype> &num_points_per_voxel,
                                 const Tensor<gpu, 1, Dtype> &actual_voxel_num,
                                 const mxnet::op::Voxel2FeatureParam &param) {
  cuda::Voxel2FeatureForward(feature, voxels, coors, num_points_per_voxel, actual_voxel_num, param);
}
}  // namespace mshadow


namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(Voxel2FeatureParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, Dtype, {
    op = new Voxel2FeatureOp<gpu, Dtype>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
