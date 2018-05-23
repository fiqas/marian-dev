#pragma once

// #include "marian.h"
// #include "common/definitions.h"
// #include "common/logging.h"
#include "common/options.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "graph/node_initializers.h"
// #include "common/definitions.h"
#include "data/corpus.h"
// #include "layers/constructors.h"
// #include "layers/generic.h"
// #include "layers/guided_alignment.h"
// #include "layers/factory.h"
// #include "layers/param_initializers.h"
//

namespace marian {

class WeightingBase {
public:
  WeightingBase(){};
  virtual Expr getWeights(Ptr<ExpressionGraph> graph,
                          Ptr<data::CorpusBatch> batch)
      = 0;
  virtual void debugWeighting(std::vector<float> weightedMask,
                              std::vector<float> freqMask,
                              Ptr<data::CorpusBatch> batch){};
};

class DataWeighting : public WeightingBase {
protected:
  std::string weightingType_;

public:
  DataWeighting(std::string weightingType)
      : WeightingBase(), weightingType_(weightingType){};
  Expr getWeights(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch);
};

class DynamicWeighting : public WeightingBase {
protected:
  std::vector<int64_t> wordFreqs_;

  void updateWordFrequencies(Ptr<data::CorpusBatch> batch);
  float weightFrequency(int64_t freq);
  std::vector<float> weightWords(Ptr<data::CorpusBatch> batch);
  std::vector<float> mapWords(Ptr<data::CorpusBatch> batch);

public:
  DynamicWeighting() : WeightingBase() { wordFreqs_.resize(1000); };
  Expr getWeights(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch);
  void debugWeighting(std::vector<float> weightedMask,
                      std::vector<float> freqMask,
                      Ptr<data::CorpusBatch> batch) override;
};
Ptr<WeightingBase> WeightingFactory(Ptr<Options> options);
}
