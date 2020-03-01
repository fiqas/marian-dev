#pragma once

#include "layers/loss.h"

namespace marian {

static inline RationalLoss AttentionCost(Ptr<ExpressionGraph> graph,
                                               Ptr<data::Batch> batch,
                                               Ptr<Options> options,
                                               std::vector<Expr> penalties,
					       float alpha = 1.0) {
  // for (int i = 0; i < penalty.size(); i++) {
          // LOG(info, "type = {}, i = {} penalty = {} ", type, i, penalty[i]->shape());
  // }

  size_t numLabels = penalties.size(); 
  std::vector<int> axes = {0, 1, 2, 3};
  
  Expr penaltySum = graph->constant({1, 1, 1, 1}, inits::zeros());
  for (auto p : penalties) {
    penaltySum = penaltySum + p;
    // debug(p, "penalty");
    // numLabels += p->shape().elements();
  }

  // debug(penaltySum, "penaltySum");
  return RationalLoss(penaltySum * alpha, numLabels);
}

}  // namespace marian
