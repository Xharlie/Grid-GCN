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
 * \file voxel_generator-inl.h
 * \brief
 * \author Xiangchen Zhao
*/
#ifndef MXNET_OPERATOR_VOXEL_GENERATOR_INL_H_
#define MXNET_OPERATOR_VOXEL_GENERATOR_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "operator_common.h"
#include "mshadow_op.h"
#include <time.h>


using namespace std; // delete after debug

namespace mxnet {
    namespace op {

        namespace voxel_generator {
            enum VoxelInput {
                kData, kActualnum
            };
            enum VoxelOutputs {
                kVoxels, kCoors, kCoorsToVoxelidx, kNumPointsPerVoxel, kActualVoxelNum
            };
            enum VoxelResource {
                kTempSpace
            };
        } // namespace voxel_generator


        struct VoxelGeneratorParam : public dmlc::Parameter<VoxelGeneratorParam> {
            index_t max_points;
            index_t max_voxels;
            index_t height;
            index_t width;
            nnvm::Tuple<float> coors_range;
            nnvm::Tuple<float> voxel_size;

            DMLC_DECLARE_PARAMETER(VoxelGeneratorParam) {
                    DMLC_DECLARE_FIELD(max_points).set_default(0).
                            describe("max points in one voxels");
                    DMLC_DECLARE_FIELD(max_voxels).set_default(0).
                    describe("max voxels for the point cloud in each frame");
                    DMLC_DECLARE_FIELD(height).set_default(0).
                    describe("height of canvas");
                    DMLC_DECLARE_FIELD(width).set_default(0).
                    describe("width of canvas");
                    DMLC_DECLARE_FIELD(coors_range).set_default(nnvm::Tuple<float>()).
                    describe("the range of point cloud");
                    DMLC_DECLARE_FIELD(voxel_size).set_default(nnvm::Tuple<float>()).
                    describe("the size of voxel");
            }
        };

        std::ostream &operator<<(std::ostream &os, const VoxelGeneratorParam &m);

        template<typename xpu, typename DType>
        class VoxelGeneratorOp : public Operator {

        public:
            explicit VoxelGeneratorOp(VoxelGeneratorParam param) {
                this->param_ = param;
            }

            virtual void Forward(const OpContext &ctx,
                                 const std::vector <TBlob> &in_data,
                                 const std::vector <OpReqType> &req,
                                 const std::vector <TBlob> &out_data,
                                 const std::vector <TBlob> &aux_states) {
                using namespace mshadow;
                CHECK_EQ(in_data.size(), 2);
                CHECK_EQ(out_data.size(), 5);

//    clock_t start_t, end_t;

                Stream <xpu> *s = ctx.get_stream<xpu>();

                Tensor<xpu, 3, DType> data = in_data[voxel_generator::kData].get<xpu, 3, DType>(s);
                Tensor<xpu, 1, DType> actual_points = in_data[voxel_generator::kActualnum].get<xpu, 1, DType>(s);
                Tensor<xpu, 4, DType> voxels = out_data[voxel_generator::kVoxels].get<xpu, 4, DType>(s);
                Tensor<xpu, 3, DType> coors = out_data[voxel_generator::kCoors].get<xpu, 3, DType>(s);
                Tensor<xpu, 2, DType> coor_to_voxelidx = out_data[voxel_generator::kCoorsToVoxelidx].get<xpu, 2, DType>(
                        s);
                Tensor<xpu, 2, DType> num_points_per_voxel = out_data[voxel_generator::kNumPointsPerVoxel].get<xpu, 2, DType>(
                        s);
                Tensor<xpu, 1, DType> actual_voxel_num = out_data[voxel_generator::kActualVoxelNum].get<xpu, 1, DType>(
                        s);

                // TensorContainer<cpu, 3, DType> cpu_data(data.shape_);
                // TensorContainer<cpu, 1, DType> cpu_actual_points(actual_points.shape_);
                // start_t = clock();
                // Copy(cpu_data, data, s);
                // Copy(cpu_actual_points, actual_points, s);
                // end_t = clock();
                // cout << "clock0: " << (double) (end_t - start_t) * 1000 / CLOCKS_PER_SEC << endl;

//    start_t = clock();
                // if( cpu_voxels.shape_ != voxels.shape_ ) {
                // // if( 1 ) {
                //   cpu_voxels = TensorContainer<cpu, 4, DType>(voxels.shape_, 0.f);
                //   cpu_coors = TensorContainer<cpu, 3, DType>(coors.shape_, 0.f);
                //   cpu_coor_to_voxelidx = TensorContainer<cpu, 2, DType>(coor_to_voxelidx.shape_, (DType)param_.max_voxels);
                mshadow::MapExp<mshadow::sv::saveto>(&voxels, mshadow::expr::ScalarExp<DType>(0.0));
                mshadow::MapExp<mshadow::sv::saveto>(&coors, mshadow::expr::ScalarExp<DType>(0.0));
                mshadow::MapExp<mshadow::sv::saveto>(&coor_to_voxelidx,
                                                     mshadow::expr::ScalarExp<DType>(param_.max_voxels));
                mshadow::MapExp<mshadow::sv::saveto>(&num_points_per_voxel, mshadow::expr::ScalarExp<DType>(0.0));
                mshadow::MapExp<mshadow::sv::saveto>(&actual_voxel_num, mshadow::expr::ScalarExp<DType>(0.0));
                //   cpu_num_points_per_voxel = TensorContainer<cpu, 2, DType>(num_points_per_voxel.shape_, 0.f);
                //   cpu_actual_voxel_num = TensorContainer<cpu, 1, DType>(actual_voxel_num.shape_, 0.f);
                // }
                cudaError err = cudaPeekAtLastError(); \
    CHECK_EQ(err, cudaSuccess) << "Name: " << "MemInit, " << " ErrStr:" << cudaGetErrorString(err);

//    end_t = clock();
//    cout << "clock init vals " << typeid(s).name() << ": " << (double) (end_t - start_t) * 1000 / CLOCKS_PER_SEC << endl;

//    start_t = clock();
                VoxelGeneratorForward(voxels, coors, coor_to_voxelidx, num_points_per_voxel, actual_voxel_num,
                                      data, actual_points, param_);
//    end_t = clock();
//    cout << "\nclock VoxelGeneratorForward<" << typeid(s).name() << ">: " << (double) (end_t - start_t) * 1000 / CLOCKS_PER_SEC << endl;
//    cout << "--------" << std::endl;

                // start_t = clock();
                // Copy(voxels, cpu_voxels, s);
                // Copy(coors, cpu_coors, s);
                // Copy(coor_to_voxelidx, cpu_coor_to_voxelidx, s);
                // Copy(num_points_per_voxel, cpu_num_points_per_voxel, s);
                // Copy(actual_voxel_num, cpu_actual_voxel_num, s);
                // end_t = clock();
                // cout << "clock3:" << (double) (end_t - start_t) * 1000 / CLOCKS_PER_SEC << endl;
            }

//  virtual void Backward(const OpContext &ctx,
//                        const std::vector<TBlob> &out_grad,
//                        const std::vector<TBlob> &in_data,
//                        const std::vector<TBlob> &out_data,
//                        const std::vector<OpReqType> &req,
//                        const std::vector<TBlob> &in_grad,
//                        const std::vector<TBlob> &aux_states){
//    using namespace mshadow;
//    CHECK_EQ(in_data.size(), 2U);
//    CHECK_EQ(out_data.size(), 5U);
//
//    Stream<xpu> *s = ctx.get_stream<xpu>();
//
//    Tensor<xpu, 3, DType> grad_in = in_grad[voxel_generator::kData].get<xpu, 3, DType>(s);
//    Tensor<xpu, 1, DType> grad_Actualnum = in_grad[voxel_generator::kActualnum].get<xpu, 1, DType>(s);
//
//    grad_in = 0.0f;
//    grad_Actualnum = 0.0f;
//  }

        private:
            VoxelGeneratorParam param_;
            // mshadow::TensorContainer<cpu, 4, DType> cpu_voxels; //(voxels.shape_, 0.f);
            // mshadow::TensorContainer<cpu, 3, DType> cpu_coors; //(coors.shape_, 0.f);
            // mshadow::TensorContainer<cpu, 2, DType> cpu_coor_to_voxelidx; //(coor_to_voxelidx.shape_, (DType)param_.max_voxels);
            // mshadow::TensorContainer<cpu, 2, DType> cpu_num_points_per_voxel; //(num_points_per_voxel.shape_, 0.f);
            // mshadow::TensorContainer<cpu, 1, DType> cpu_actual_voxel_num; //(actual_voxel_num.shape_, 0.f);
        }; // class VoxelGeneratorOp

// Declare Factory function, used for dispatch specialization
        template<typename xpu>
        Operator *CreateOp(VoxelGeneratorParam param, int dtype);

#if DMLC_USE_CXX11
        class VoxelGeneratorProp : public OperatorProperty {
          public:
            std::vector<std::string> ListArguments() const override {
              return {"data", "actualnum"};
            }

            std::vector<std::string> ListOutputs() const override {
              return {"voxels", "coors", "coor_to_voxelidx", "num_points_per_voxel", "actual_voxel_num"};
            }

          int NumOutputs() const override {
            return 5;
          }

          int NumVisibleOutputs() const override {
            return 5;
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
              CHECK_EQ(in_shape->size(), 2U) << "Input:[data, Actualnum]";

              // data: [batch_size, N, C]
              TShape dshape = in_shape->at(voxel_generator::kData);
              CHECK_EQ(dshape.ndim(), 3U) << "data should be a 3D tensor";

              // Actualnum: [batch_size]
              TShape ashape = in_shape->at(voxel_generator::kActualnum);
              CHECK_EQ(ashape.ndim(), 1U) << "actualnum should be a 1D tensor";

              // voxels: [batch_size, max_voxels, max_points]
              // coors: [batch_size, max_voxels, 3]
              // coor_to_voxelidx: [batch_size, height * width]
              // num_points_per_voxel: [batch_size,  max_voxels]
              // actual_voxel_num: [batch_size]
              out_shape->clear();
              out_shape->push_back(Shape4(dshape[0], param_.max_voxels, param_.max_points, 4));
              out_shape->push_back(Shape3(dshape[0], param_.max_voxels, 3));
              out_shape->push_back(Shape2(dshape[0], param_.height * param_.width));
              out_shape->push_back(Shape2(dshape[0], param_.max_voxels));
              out_shape->push_back(Shape1(dshape[0]));
              return true;
            }

            bool InferType(std::vector<int> *in_type,
                           std::vector<int> *out_type,
                           std::vector<int> *aux_type) const override {
              CHECK_EQ(in_type->size(), 2U);
              int dtype = (*in_type)[0];
              CHECK_NE(dtype, -1) << "Input must have specified type";

              out_type->clear();
              out_type->push_back(dtype);
              out_type->push_back(dtype);
              out_type->push_back(dtype);
              out_type->push_back(dtype);
              out_type->push_back(dtype);
              return true;
            }

            OperatorProperty* Copy() const override {
              VoxelGeneratorProp* sym = new VoxelGeneratorProp();
              sym->param_ = this->param_;
              return sym;
            }

            std::string TypeString() const override {
              return "VoxelGenerator";
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
          VoxelGeneratorParam param_;

        };
#endif  // DMLC_USE_CXX11
    } // namespace op
} // namespace mxnet
#endif  // MXNET_OPERATOR_VOXEL_GENERATOR_INL_H_
