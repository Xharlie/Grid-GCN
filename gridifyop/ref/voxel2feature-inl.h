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
 * \file voxel2feature-inl.h
 * \brief
 * \author Xiangchen Zhao
*/
#ifndef MXNET_OPERATOR_VOXEL2FEATURE_INL_H_
#define MXNET_OPERATOR_VOXEL2FEATURE_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "operator_common.h"
#include "mshadow_op.h"
#include <iostream>

using namespace std; // delete after debug

namespace mxnet {
namespace op {

namespace voxel2feature {
enum VoxelInput {kVoxels, kCoors, kNumPointsPerVoxel, kActualVoxelNum};
enum VoxelOutputs {kFeature};
} // namespace voxel2feature


struct Voxel2FeatureParam : public dmlc::Parameter<Voxel2FeatureParam> {
  index_t max_points;
  index_t max_voxels;
  index_t channel;
  bool use_intensity;
  nnvm::Tuple<float> coors_range;
  nnvm::Tuple<float> voxel_size;

  DMLC_DECLARE_PARAMETER(Voxel2FeatureParam) {
    DMLC_DECLARE_FIELD(max_points).set_default(0).
    describe("max points in one voxels");
    DMLC_DECLARE_FIELD(max_voxels).set_default(0).
    describe("max voxels for the point cloud in each frame");
    DMLC_DECLARE_FIELD(channel).set_default(0).
    describe("channel of feature");
    DMLC_DECLARE_FIELD(use_intensity).set_default(true).
    describe("whether use intensity in feature");
    DMLC_DECLARE_FIELD(coors_range).set_default(nnvm::Tuple<float>()).
    describe("the range of point cloud");
    DMLC_DECLARE_FIELD(voxel_size).set_default(nnvm::Tuple<float>()).
    describe("the size of voxel");
  }
};
template<typename xpu, typename DType>
class Voxel2FeatureOp : public Operator {
public:
  explicit Voxel2FeatureOp(Voxel2FeatureParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 4);
    CHECK_EQ(out_data.size(), 1);

    Stream<xpu> *s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> voxels = in_data[voxel2feature::kVoxels].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> coors = in_data[voxel2feature::kCoors].get<xpu, 3, DType>(s);
    Tensor<xpu, 2, DType> num_points_per_voxel = in_data[voxel2feature::kNumPointsPerVoxel].get<xpu, 2, DType>(s);
    Tensor<xpu, 1, DType> actual_voxel_num = in_data[voxel2feature::kActualVoxelNum].get<xpu, 1, DType>(s);
    Tensor<xpu, 4, DType> feature = out_data[voxel2feature::kFeature].get<xpu, 4, DType>(s);

    mshadow::MapExp<mshadow::sv::saveto>(&feature, mshadow::expr::ScalarExp<DType>(0.0));

    CHECK_EQ(voxels.CheckContiguous(), true);
    CHECK_EQ(coors.CheckContiguous(), true);
    CHECK_EQ(num_points_per_voxel.CheckContiguous(), true);
    CHECK_EQ(actual_voxel_num.CheckContiguous(), true);
    CHECK_EQ(feature.CheckContiguous(), true);


    Voxel2FeatureForward(feature, voxels, coors, num_points_per_voxel, actual_voxel_num, param_);

  }

private:
  Voxel2FeatureParam param_;
}; // class Voxel2FeatureOp

// Declare Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(Voxel2FeatureParam param, int dtype);

#if DMLC_USE_CXX11
class Voxel2FeatureProp : public OperatorProperty {
public:
  std::vector<std::string> ListArguments() const override {
    return {"voxels", "coors", "num_points_per_voxel", "actual_voxel_num"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"feature"};
  }

  int NumOutputs() const override {
    return 1;
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 4U) << "Input:[voxels, coors, num_points_per_voxel, actual_voxel_num]";

    // voxels: [batch_size, max_voxels, max_points, 4]
    TShape vshape = in_shape->at(voxel2feature::kVoxels);
    CHECK_EQ(vshape.ndim(), 4U) << "voxels should be a 4D tensor";

    // coors: [batch_size, max_voxels, 3]
    TShape cshape = in_shape->at(voxel2feature::kCoors);
    CHECK_EQ(cshape.ndim(), 3U) << "coors should be a 3D tensor";

    // num_points_per_voxel: [batch_size,  max_voxels]
    TShape nshape = in_shape->at(voxel2feature::kNumPointsPerVoxel);
    CHECK_EQ(nshape.ndim(), 2U) << "coors should be a 2D tensor";

    // actual_voxel_num: [batch_size]
    TShape ashape = in_shape->at(voxel2feature::kActualVoxelNum);
    CHECK_EQ(ashape.ndim(), 1U) << "coors should be a 1D tensor";



    // feature: [batch_size, max_voxels, max_points, channels]
    out_shape->clear();
    out_shape->push_back(
    Shape4(vshape[0], param_.max_voxels, param_.max_points, param_.channel));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 4U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    Voxel2FeatureProp* sym = new Voxel2FeatureProp();
    sym->param_ = this->param_;
    return sym;
  }

  std::string TypeString() const override {
    return "Voxel2Feature";
  }

  // declare dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(const std::vector<int> &out_grad,
                                             const std::vector<int> &in_data,
                                             const std::vector<int> &out_data) const override {
    return {};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

private:
  Voxel2FeatureParam param_;

};
#endif  // DMLC_USE_CXX11
} // namespace op
} // namespace mxnet
#endif  // MXNET_OPERATOR_VOXEL2Feature_INL_H_
