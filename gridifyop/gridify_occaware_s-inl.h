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
 * \file gridify_occaware_s-inl.h
 * \brief
 * \author Qiangeng Xu
*/
#ifndef MXNET_OPERATOR_GRIDIFY_occaware_s_INL_H_
#define MXNET_OPERATOR_GRIDIFY_occaware_s_INL_H_

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
        namespace gridify_occaware_s {
            enum gridifyInput_occaware_s {
                kData, kActualnum
            };
            enum gridifyOutputs_occaware_s {
                kNeighborsB, kNeighborsMaskB, kCentsB, kCentsMaskB, kCentsNumB
            };
            enum gridifyResource_occaware_s {
                kTempSpace
            };
        } // namespace gridifier


        struct GridifyParam_occaware_s : public dmlc::Parameter<GridifyParam_occaware_s> {
//    width, height, depth from birdview
            index_t max_p_grid;
            index_t max_o_grid;
            index_t kernel_size;
            index_t stride;
            index_t loc;
            nnvm::Tuple<float> coord_shift;
            nnvm::Tuple<float> voxel_size;
            nnvm::Tuple<int> grid_size;

            DMLC_DECLARE_PARAMETER(GridifyParam_occaware_s) {
                    DMLC_DECLARE_FIELD(max_p_grid).set_default(0).
                        describe("max points in one voxels");
                    DMLC_DECLARE_FIELD(max_o_grid).set_default(0).
                        describe("max voxels for the point cloud in each frame");
                    DMLC_DECLARE_FIELD(kernel_size).set_default(0).
                        describe("single kernal size for each x,y,z axis");
                    DMLC_DECLARE_FIELD(stride).set_default(0).
                        describe("single stride size for each x,y,z axis");
                    DMLC_DECLARE_FIELD(loc).set_default(0).
                        describe("whether to loc center within the center");
                    DMLC_DECLARE_FIELD(coord_shift).set_default(nnvm::Tuple<float>()).
                        describe("the starting coordinate of gridification for point cloud");
                    DMLC_DECLARE_FIELD(voxel_size).set_default(nnvm::Tuple<float>()).
                        describe("the size of voxel, [0.04, 0.04, 0.02]");
                    DMLC_DECLARE_FIELD(grid_size).set_default(nnvm::Tuple<int>()).
                        describe("the size of grid_size, [50, 50, 100]");
            }
        };

        std::ostream &operator<<(std::ostream &os, const GridifyParam_occaware_s &m);

        template<typename xpu, typename DType>
        class GridifyOp_occaware_s : public Operator {

        public:
            explicit GridifyOp_occaware_s(GridifyParam_occaware_s param) {
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

                Stream <xpu> *s = ctx.get_stream<xpu>();
                Tensor<xpu, 3, DType> data = in_data[gridify_occaware_s::kData].get<xpu, 3, DType>(s); // B * N * 4
                Tensor<xpu, 2, int> actual_numpoints = in_data[gridify_occaware_s::kActualnum].get<xpu, 2, int>(s); // B * 1
                Tensor<xpu, 3, int> nebidx = out_data[gridify_occaware_s::kNeighborsB].get<xpu, 3, int>(s); // B * O * P
                Tensor<xpu, 3, DType> nebidxmsk = out_data[gridify_occaware_s::kNeighborsMaskB].get<xpu, 3, DType>(s); // B * O * P
                Tensor<xpu, 3, DType> cent = out_data[gridify_occaware_s::kCentsB].get<xpu, 3, DType>(s); // B * O * 4
                Tensor<xpu, 2, DType> centmsk = out_data[gridify_occaware_s::kCentsMaskB].get<xpu, 2, DType>(s); // B * O
                Tensor<xpu, 2, int> actual_centnum = out_data[gridify_occaware_s::kCentsNumB].get<xpu, 2, int>(s); // B * 1

                mshadow::MapExp<mshadow::sv::saveto>(&nebidx, mshadow::expr::ScalarExp<int>(0));
                mshadow::MapExp<mshadow::sv::saveto>(&nebidxmsk, mshadow::expr::ScalarExp<DType>(0.0));
                mshadow::MapExp<mshadow::sv::saveto>(&cent, mshadow::expr::ScalarExp<DType>(1.0));
                mshadow::MapExp<mshadow::sv::saveto>(&centmsk, mshadow::expr::ScalarExp<DType>(0.0));
                mshadow::MapExp<mshadow::sv::saveto>(&actual_centnum, mshadow::expr::ScalarExp<int>(0));

                cudaError err = cudaPeekAtLastError(); \
    CHECK_EQ(err, cudaSuccess) << "Name: " << "MemInit, " << " ErrStr:" << cudaGetErrorString(err);

                GridifyForward_occaware_s(nebidx, nebidxmsk, cent, centmsk, actual_centnum,
                               data, actual_numpoints, param_);
            }

        private:
            GridifyParam_occaware_s param_;
            // mshadow::TensorContainer<cpu, 4, DType> cpu_voxels; //(voxels.shape_, 0.f);
            // mshadow::TensorContainer<cpu, 3, DType> cpu_coors; //(coors.shape_, 0.f);
            // mshadow::TensorContainer<cpu, 2, DType> cpu_coor_to_voxelidx; //(coor_to_voxelidx.shape_, (DType)param_.max_voxels);
            // mshadow::TensorContainer<cpu, 2, DType> cpu_num_points_per_voxel; //(num_points_per_voxel.shape_, 0.f);
            // mshadow::TensorContainer<cpu, 1, DType> cpu_actual_voxel_num; //(actual_voxel_num.shape_, 0.f);
        }; // class VoxelGeneratorOp

// Declare Factory function, used for dispatch specialization
        template<typename xpu>
        Operator *CreateOp(GridifyParam_occaware_s param, int dtype);

#if DMLC_USE_CXX11
        class GridifyProp_occaware_s : public OperatorProperty {
          public:
            std::vector<std::string> ListArguments() const override {
              return {"data", "actual_numpoints"};
            }

            std::vector<std::string> ListOutputs() const override {
              return {"nebidx", "nebidxmsk", "cent", "centmsk", "actual_centnum"};
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
              TShape dshape = in_shape->at(gridify_occaware_s::kData);
              CHECK_EQ(dshape.ndim(), 3U) << "data should be a 3D tensor";

              // Actualnum: [batch_size]
              TShape ashape = in_shape->at(gridify_occaware_s::kActualnum);
              CHECK_EQ(ashape.ndim(), 2U) << "actualnum should be a 2D tensor";

//                Tensor<xpu, 3, DType> nebidx = out_data[gridify::kNeighborsB].get<xpu, 3, DType>(s); // B * O * P
//                Tensor<xpu, 3, DType> nebidxmsk = out_data[gridify::kNeighborsMaskB].get<xpu, 3, DType>(s); // B * O * P
//                Tensor<xpu, 3, DType> cent = out_data[gridify::kCentsB].get<xpu, 3, DType>(s); // B * O * 4
//                Tensor<xpu, 2, DType> centmsk = out_data[gridify::kCentsMaskB].get<xpu, 2, DType>(s); // B * O
//                Tensor<xpu, 1, DType> actual_centnum = out_data[gridify::kCentsNumB].get<xpu, 1, DType>(s); // B

              out_shape->clear();
              out_shape->push_back(Shape3(dshape[0], param_.max_o_grid, param_.max_p_grid));
              out_shape->push_back(Shape3(dshape[0], param_.max_o_grid, param_.max_p_grid));
              out_shape->push_back(Shape3(dshape[0], param_.max_o_grid, 4));
              out_shape->push_back(Shape2(dshape[0], param_.max_o_grid));
              out_shape->push_back(Shape2(dshape[0], 1));
              return true;
            }

            bool InferType(std::vector<int> *in_type,
                           std::vector<int> *out_type,
                           std::vector<int> *aux_type) const override {
              CHECK_EQ(in_type->size(), 2U);
              int dtype = (*in_type)[0];
              int intdtype = (*in_type)[1];
              CHECK_NE(dtype, -1) << "Input must have specified type";

              out_type->clear();
              out_type->push_back(intdtype);
              out_type->push_back(dtype);
              out_type->push_back(dtype);
              out_type->push_back(dtype);
              out_type->push_back(intdtype);
              return true;
            }

            OperatorProperty* Copy() const override {
              GridifyProp_occaware_s* sym = new GridifyProp_occaware_s();
              sym->param_ = this->param_;
              return sym;
            }

            std::string TypeString() const override {
              return "Gridify_occaware_s";
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
          GridifyParam_occaware_s param_;

        };
#endif  // DMLC_USE_CXX11
    } // namespace op
} // namespace mxnet
#endif  // MXNET_OPERATOR_GRIDIFY_occaware_s_INL_H_
