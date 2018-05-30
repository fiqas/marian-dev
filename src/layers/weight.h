#pragma once

#include "common/options.h"
#include "data/corpus.h"
#include "graph/expression_graph.h"
#include "graph/expression_operators.h"
#include "graph/node_initializers.h"

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
  std::vector<float> params_;

  void updateWordFrequencies(Ptr<data::CorpusBatch> batch);
  virtual float weightFrequency(int64_t freq);
  std::vector<float> weightWords(Ptr<data::CorpusBatch> batch);
  std::vector<float> mapWords(Ptr<data::CorpusBatch> batch);

public:
  DynamicWeighting(int vocabSize, std::vector<float> params) : WeightingBase() {
    wordFreqs_.resize(vocabSize + 1);
    params_ = params;
  };
  Expr getWeights(Ptr<ExpressionGraph> graph, Ptr<data::CorpusBatch> batch);
  void debugWeighting(std::vector<float> weightedMask,
                      std::vector<float> freqMask,
                      Ptr<data::CorpusBatch> batch) override;
};

class ExponentialWeighting : public DynamicWeighting {
protected:
  float weightFrequency(int64_t freq);

public:
  ExponentialWeighting(int vocabSize, std::vector<float> params)
      : DynamicWeighting(vocabSize, params) {
    ABORT_IF(params.size() < 6,
             "Too few hyperparameters for exponential weighting!");
  };
};

class InvSqrtWeighting : public DynamicWeighting {
protected:
  float weightFrequency(int64_t freq);

public:
  InvSqrtWeighting(int vocabSize, std::vector<float> params)
      : DynamicWeighting(vocabSize, params) {
    ABORT_IF(params.size() < 4,
             "Too few hyperparameters for exponential weighting!");
  };
};

class InvSqrtBoundWeighting : public DynamicWeighting {
protected:
  float weightFrequency(int64_t freq);

public:
  InvSqrtBoundWeighting(int vocabSize, std::vector<float> params)
      : DynamicWeighting(vocabSize, params) {
    ABORT_IF(params.size() < 5,
             "Too few hyperparameters for exponential weighting!");
  };
};

Ptr<WeightingBase> WeightingFactory(Ptr<Options> options);
}
