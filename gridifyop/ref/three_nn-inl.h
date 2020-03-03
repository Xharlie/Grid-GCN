/*!
 * Copyright (c) 2017 by Contributors
 * \file three_nn-inl.h
 * \brief find three nearest points
 * \author Jianlin Liu
*/
#ifndef MXNET_OPERATOR_CONTRIB_THREE_NN_INL_H_
#define MXNET_OPERATOR_CONTRIB_THREE_NN_INL_H_

#include <vector>
#include <utility>
#include <math.h>
#include <limits>
#include <mxnet/operator_util.h>
#include "../mxnet_op.h"
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../tensor/init_op.h"
#include "../operator_common.h"

namespace mxnet {
  typedef std::vector<mxnet::TShape> ShapeVector;
namespace op {

struct ThreeNNParam : public dmlc::Parameter<ThreeNNParam> {
//  float radius;
//  int nsample;
  DMLC_DECLARE_PARAMETER(ThreeNNParam) {
//    DMLC_DECLARE_FIELD(radius)
//      .describe("Search radius.");
//    DMLC_DECLARE_FIELD(nsample)
//      .describe("Number of samples ball within radius to be returned.");
  }
};

struct ThreeNNKernel {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, const int n, const int m,
                                  const DType* unknown, const DType* known,
                                  float* dist, int* idx) {
    int b = i / n;
    known += b * m * 3;
    unknown += i * 3;
    dist += i * 3;
    idx += i * 3;

    float ux = unknown[0];
    float uy = unknown[1];
    float uz = unknown[2];

    float best1 = FLT_MAX, best2 = FLT_MAX, best3 = FLT_MAX;
    int besti1 = 0, besti2 = 0, besti3 = 0;
    for (int k = 0; k < m; ++k) {
        float x = known[k * 3 + 0];
        float y = known[k * 3 + 1];
        float z = known[k * 3 + 2];
        float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
        if (d < best1) {
            best3 = best2; besti3 = besti2;
            best2 = best1; besti2 = besti1;
            best1 = d; besti1 = k;
        }
        else if (d < best2) {
            best3 = best2; besti3 = besti2;
            best2 = d; besti2 = k;
        }
        else if (d < best3) {
            best3 = d; besti3 = k;
        }
    }
    dist[0] = sqrt(best1); dist[1] = sqrt(best2); dist[2] = sqrt(best3);
    if (dist[0] == DType(0.0)){
      dist[0] = DType(1.0);
      dist[1] = DType(0.0);
      dist[2] = DType(0.0);
    }
    else{
      dist[0] = DType(1.0) / dist[0];
      dist[1] = DType(1.0) / dist[1];
      dist[2] = DType(1.0) / dist[2];
      DType norm = dist[0] + dist[1] + dist[2];
      dist[0] /= norm;
      dist[1] /= norm;
      dist[2] /= norm;
    }
    idx[0] = besti1; idx[1] = besti2; idx[2] = besti3;

  }
};

template <typename xpu>
void ThreeNNForward(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                         const std::vector<TBlob>& in_data,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& out_data) {
  using namespace mshadow;
  CHECK_EQ(in_data.size(), 2U);
  CHECK_EQ(out_data.size(), 2U);

  const int batch_size = in_data[0].size(0);
  const int n = in_data[0].size(1); // unknown
  const int m = in_data[1].size(1); // known

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
//  const ThreeNNParam& param = nnvm::get<ThreeNNParam>(attrs.parsed);
  MSHADOW_TYPE_SWITCH(in_data[0].type_flag_, DType, {
     mxnet_op::Kernel<ThreeNNKernel, xpu>::Launch(
       s, batch_size*n, n, m, in_data[0].dptr<DType>(), in_data[1].dptr<DType>(),
       out_data[0].dptr<float>(), out_data[1].dptr<int>());
  });
}

}
}

#endif  // MXNET_OPERATOR_CONTRIB_THREE_NN_INL_H_
