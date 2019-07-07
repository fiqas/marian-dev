#pragma once

#include "layers/loss.h"

namespace marian {

static inline RationalLoss headEntropyCost(Ptr<ExpressionGraph> graph,
                                               Ptr<data::CorpusBatch> batch,
                                               Ptr<Options> options,
                                               std::vector<Expr> weights,
					       std::string type,
					       float alpha = 1.0) {
  // for (int i = 0; i < weights.size(); i++) {
          // LOG(info, "type = {}, i = {} weights = {} ", type, i, weights[i]->shape());
  // }

  float epsilon = 1e-6f;
  size_t numLabels; 
  int numHeads = options->get<int>("transformer-heads");
  std::vector<int> axes = {0, 1, 2, 3};
  Expr weightsConcat;
  Expr lossSum;

  if (type == "encoder") {
    auto weightsConcat = concatenate(weights, /*axis */ -3);
    // debug(weightsConcat, "weightsConcat " + type);
  
    auto headLog = log(weightsConcat + epsilon);
    // debug(headLog, "log(weights + epsilon) " + type);
  
    auto headEntropyLoss = -sum((weightsConcat * headLog), -1);
    // debug(headEntropyLoss, "-sum(weights * log(weights + eps))" + type);

    numLabels = weights.size() * batch->words() * numHeads;
    lossSum = headEntropyLoss;
    for(int i = 0; i < axes.size(); ++i)
      lossSum = sum(lossSum, axes[i]);
  }
  else { // If decoder
    std::vector<Expr> weightsSelf;
    std::vector<Expr> weightsContext;

    bool toggle = false;
    std::partition_copy(weights.begin(),
                        weights.end(),
                        std::back_inserter(weightsSelf),
                        std::back_inserter(weightsContext),
                        [&toggle](Expr) { return toggle = !toggle; });		

    auto weightsSelfConcat = concatenate(weightsSelf, /*axis */ -3);
    auto weightsContextConcat = concatenate(weightsContext, /*axis */ -3);
    
    // debug(weightsSelfConcat, "weightsSelfConcat " + type);
    // debug(weightsContextConcat, "weightsContextConcat " + type);

    auto headSelfLog = log(weightsSelfConcat + epsilon);
    auto headContextLog = log(weightsContextConcat + epsilon);
    
    // debug(headSelfLog, "log(weightsSelf + epsilon) " + type);
    // debug(headContextLog, "log(weightsContext + epsilon) " + type);
    
    auto headEntropySelfLoss = -sum((weightsSelfConcat * headSelfLog), -1);
    auto headEntropyContextLoss = -sum((weightsContextConcat * headContextLog), -1);

    // debug(headEntropySelfLoss, "-sum(weightsSelf * log(weightsSelf + eps))" + type);
    // debug(headEntropyContextLoss, "-sum(weightsContext * log(weightsContext + eps))" + type);

    numLabels = weights.size() * batch->wordsTrg() * numHeads;
    Expr lossSelfSum = headEntropySelfLoss;
    Expr lossContextSum = headEntropyContextLoss;
    for(int i = 0; i < axes.size(); ++i) {
      lossSelfSum = sum(lossSelfSum, axes[i]);
      lossContextSum = sum(lossContextSum, axes[i]);
    }


    // debug(lossSelfSum, "lossSelfSum " + type);
    // debug(lossContextSum, "lossContextSum " + type);
    lossSum = lossSelfSum + lossContextSum;
  }

  // debug(lossSum, "lossSum " + type);
  return RationalLoss(lossSum * alpha, numLabels);
}

}  // namespace marian
