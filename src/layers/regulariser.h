#pragma once

#include "layers/loss.h"
#include "common/logging.h"

namespace marian {

static inline RationalLoss regulariserCost(Ptr<ExpressionGraph> graph,
                                               Ptr<data::CorpusBatch> batch,
                                               Ptr<Options> options,
                                               std::vector<Expr> encoderCosts,
					       std::vector<Expr> decoderCosts,
                                               Expr labels
					       ) { // [beam depth=1, max src length, batch size, tgt length]

  float regulariserScalar = options->get<float>("group-lasso-regulariser");

  // int layerCount = decoderCosts.size() + encoderCosts.size();
  int layerCount = 1.0;
  // float epsilon = 1e-6f;
  // Expr regulariserLoss = encoderCosts[0]; // sum up loss over all regularisation costs
  Expr regulariserLoss = graph->constant({1, 1}, inits::zeros()); // sum up loss over all regularisation costs
  
  // LOG(info, "encoder cost length: {}", encoderCosts.size());
  // LOG(info, "decoder cost length: {}", decoderCosts.size());
  // int count = 0;
  for(auto c : encoderCosts) {
    // LOG(info, "inside for encoder");
    // if (count == 0) {
      // count++;
      // continue;
    // }
    // else
      regulariserLoss = regulariserLoss + c;
    // debug(regulariserLoss);
  }

  for(auto c : decoderCosts) {
    // LOG(info, "inside for decoder");
    regulariserLoss = regulariserLoss + c;
    //debug(regulariserLoss);
  }



  return RationalLoss(regulariserScalar * regulariserLoss, layerCount);
  // return RationalLoss(regulariserScalar * regulariserLoss, layerCount);
}

}  // namespace marian
