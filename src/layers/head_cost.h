#pragma once

#include "layers/loss.h"

namespace marian {

static inline RationalLoss headEntropyCost(Ptr<ExpressionGraph> graph,
                                               Ptr<data::CorpusBatch> batch,
                                               Ptr<Options> options,
                                               Expr weights,
					       float alpha) {
  float epsilon = 1e-6f;

  // auto headEntropyLoss = -sum((weights * log(weights + epsilon))) * alpha;
  auto headEntropyLoss = -sum((weights * log(weights + epsilon)), -1);
 
  // LOG(info, "headEntropyLoss {}", headEntropyLoss->shape());
  size_t numLabels = headEntropyLoss->shape().elements();
 
  std::vector<int> axes_ = {0, 1, 2, 3};
  Expr lossSum = headEntropyLoss;
    for(int i = 0; i < axes_.size(); ++i)
      lossSum = sum(lossSum, axes_[i]);
  // LOG(info, "headEntropyLoss Sum {}", lossSum->shape());
  // LOG(info, "headEntropyLoss numLabels {}", numLabels);
  // debug(lossSum, "headEntropySum function"); 
  return RationalLoss(lossSum, numLabels);
}

}  // namespace marian
