/*!
 * Copyright (c) 2017 by Contributors
 * \file three_nn.cc
 * \brief find three nearest points
 * \author Jianlin Liu
*/
#include "./k_nn-inl.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(KNNParam);

NNVM_REGISTER_OP(_contrib_KNN)
.describe("KNN foward.")
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr_parser(ParamParser<KNNParam>)
.set_attr<nnvm::FNumVisibleOutputs>("FNumVisibleOutputs",
    [](const NodeAttrs& attrs) {
  return 1;
})
.set_attr<nnvm::FListInputNames>("FListInputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"unknown", "known", "downnum", "upnum"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames", [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"idx"};
})
.set_attr<mxnet::FInferShape>("FInferShape", [](const nnvm::NodeAttrs& attrs,
      mxnet::ShapeVector *in_shape, mxnet::ShapeVector *out_shape){
  using namespace mshadow;
  const KNNParam param = nnvm::get<KNNParam>(attrs.parsed);
  CHECK_EQ(in_shape->size(), 4) << "Input:[unknown, known, downnum, upnum]";

  mxnet::TShape unknown_shape = in_shape->at(0);
  CHECK_EQ(unknown_shape.ndim(), 3) << "Unknown should be of shape (b, n, 3)";
  CHECK_EQ(unknown_shape[2], 3) << "Last dim of unknown should be 3";

  mxnet::TShape known_shape = in_shape->at(1);
  CHECK_EQ(known_shape.ndim(), 3) << "Known should be of shape (b, m, 3)";
  CHECK_EQ(known_shape[2], 3) << "Last dim of known should be 3";

  out_shape->clear();
  out_shape->push_back(Shape3(unknown_shape[0], unknown_shape[1], param.k));
//  out_shape->push_back(Shape3(unknown_shape[0], unknown_shape[1], 3));
  return true;
})
.set_attr<nnvm::FInferType>("FInferType", [](const nnvm::NodeAttrs& attrs,
      std::vector<int> *in_type, std::vector<int> *out_type) {
  CHECK_EQ(in_type->size(), 4);
  int dtype = (*in_type)[0];
  CHECK_EQ(dtype, (*in_type)[1]);
  CHECK_NE(dtype, -1) << "Input must have specified type";

  out_type->clear();
  out_type->push_back(mshadow::kInt32);
  return true;
})
.set_attr<FCompute>("FCompute<cpu>", KNNForward<cpu>)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.add_argument("unknown", "NDArray-or-Symbol", "Query points xyz, 3D tensor")
.add_argument("known", "NDArray-or-Symbol", "Reference point xyz, 3D tensor")
.add_argument("downnum", "NDArray-or-Symbol", "num of real know")
.add_argument("upnum", "NDArray-or-Symbol", "num of real unknow")
.add_arguments(KNNParam::__FIELDS__());

}
}