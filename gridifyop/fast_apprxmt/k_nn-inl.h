/*!
 * Copyright (c) 2017 by Contributors
 * \file k_nn-inl.h
 * \brief find three nearest points
 * \author Jianlin Liu
*/
#ifndef MXNET_OPERATOR_CONTRIB_K_NN_INL_H_
#define MXNET_OPERATOR_CONTRIB_K_NN_INL_H_

#include <vector>
#include <utility>
#include <math.h>
#include <limits>
#include <algorithm>
#include <mxnet/operator_util.h>
//#include "../mxnet_op.h"
//#include "../elemwise_op_common.h"
//#include "../mshadow_op.h"
//#include "../tensor/init_op.h"
//#include "../operator_common.h"
#include "mxnet_op.h"
#include "elemwise_op_common.h"
#include "mshadow_op.h"
#include "tensor/init_op.h"
#include "operator_common.h"
namespace mxnet {
    typedef std::vector <mxnet::TShape> ShapeVector;
    namespace op {

        struct KNNParam : public dmlc::Parameter<KNNParam> {
//  float radius;
//  int nsample;
            index_t k;
            DMLC_DECLARE_PARAMETER(KNNParam) {
                    DMLC_DECLARE_FIELD(k).set_default(3).
                            describe("numbers of neighbors");
            }
        };

        struct KNNKernel {
            template<typename DType>
            MSHADOW_XINLINE static void Map(int i, const int n, const int m, const int topk,
                const DType *unknown, const DType *known, const int *downnum, const int *upnum,
                int *idx) {
//                printf("has k,%d \n", k);
                int b = i / n;
                int downnum_val = downnum[b];
                int upnum_val = upnum[b];
                if(i % n >= upnum_val){
                    return;
                }
                known += b * m * 3;
                unknown += i * 3;
                idx += i * topk;

                float ux = unknown[0];
                float uy = unknown[1];
                float uz = unknown[2];

                float* best = new float[topk];
                int* besti = new int[topk];
//                float best[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
//                int besti[3];
                for (int l = 0; l < topk; l++){
                    best[l] = FLT_MAX;
                }
//                int reallast;
                for (int k = 0; k < downnum_val; ++k) {
                    float x = known[k * 3 + 0];
                    float y = known[k * 3 + 1];
                    float z = known[k * 3 + 2];
                    float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
//                    reallast = topk-1 < k ? topk-1 : k;
                    for (int l = 0; l < topk; l++){
                        if (d < best[l]){
                            for (int j = topk-1; j > l ; j--){
                                best[j] = best[j-1];
                                besti[j] = besti[j-1];
                            }
                            best[l] = d;
                            besti[l] = k;
                            break;
                        }
                    }
                }
                for (int l = 0; l < topk; l++){
                    idx[l] = besti[l];
                }
                delete [] best;
                delete [] besti;
            }
        };

        template<typename xpu>
        void KNNForward(const nnvm::NodeAttrs &attrs, const OpContext &ctx,
                        const std::vector <TBlob> &in_data,
                        const std::vector <OpReqType> &req,
                        const std::vector <TBlob> &out_data) {
            using namespace mshadow;
            CHECK_EQ(in_data.size(), 4U);
            CHECK_EQ(out_data.size(), 1U);

            const int batch_size = in_data[0].size(0);
            const int n = in_data[0].size(1); // unknown
            const int m = in_data[1].size(1); // known
            mshadow::Stream <xpu> *s = ctx.get_stream<xpu>();
            const KNNParam &param = nnvm::get<KNNParam>(attrs.parsed);
            MSHADOW_TYPE_SWITCH(in_data[0].type_flag_, DType, {
                    mxnet_op::Kernel<KNNKernel, xpu>::Launch(
                            s, batch_size * n, n, m, param.k, in_data[0].dptr<DType>(), in_data[1].dptr<DType>(),
                            in_data[2].dptr<int>(), in_data[3].dptr<int>(), out_data[0].dptr<int>());
            });
        }

    }
}

#endif  // MXNET_OPERATOR_CONTRIB_K_NN_INL_H_
