#include "layers/weight.h"

namespace marian {

Ptr<WeightingBase> WeightingFactory(Ptr<Options> options) {
  if(options->has("data-weighting"))
    return New<DataWeighting>(options->get<std::string>("data-weighting-type"));
  else if(options->has("dynamic-weighting")
          && options->get<bool>("dynamic-weighting")) {
    auto vocabSize = options->get<std::vector<int>>("dim-vocabs").back();
    auto params = options->get<std::vector<float>>("dynamic-weighting-params");
    std::cerr << " Params before passing: ";
    for(size_t i = 0; i < params.size(); i++) {
      std::cerr << " " << params[i];
    }
    std::cerr << std::endl;
    auto type = options->get<std::string>("dynamic-weighting-type");
    std::cerr << "Vocabulary size: " << vocabSize << std::endl;
    if(type == "exp") {
      std::cerr << "Weighting type: exp" << std::endl;
      return New<ExponentialWeighting>(vocabSize, params);
    } else if(type == "inv-sqrt") {
      std::cerr << "Weighting type: inv_sqrt" << std::endl;
      return New<InvSqrtWeighting>(vocabSize, params);
    } else
      return New<DynamicWeighting>(vocabSize, params);
    std::cerr << "Weighting type: NONE, weighting set to 1" << std::endl;
  }
}

Expr DataWeighting::getWeights(Ptr<ExpressionGraph> graph,
                               Ptr<data::CorpusBatch> batch) {
  ABORT_IF(batch->getDataWeights().empty(),
           "Vector of weights is unexpectedly empty!");
  bool sentenceWeighting = weightingType_ == "sentence";
  int dimBatch = batch->size();
  int dimWords = sentenceWeighting ? 1 : batch->back()->batchWidth();
  auto weights = graph->constant({1, dimWords, dimBatch, 1},
                                 inits::from_vector(batch->getDataWeights()));
  return weights;
}

void DynamicWeighting::updateWordFrequencies(Ptr<data::CorpusBatch> batch) {
  auto sb = batch->back();
  for(size_t i = 0; i < sb->data().size(); i++) {
    Word w = sb->data()[i];
    // if(wordFreqs_.size() < w && w != 0) {
    // std::cerr << "RESIZEEEEE" << std::endl;
    // wordFreqs_.resize(w);
    // }
    wordFreqs_[w]++;
  }
}

float DynamicWeighting::weightFrequency(int64_t freq) {
  float floatFreq = static_cast<float>(freq);
  auto result = 1.0f;
  return result;
}

// float DynamicWeighting::weightFrequency(float freq) {
// auto result = 1.0f;
// return result;
// }

std::vector<float> DynamicWeighting::mapWords(Ptr<data::CorpusBatch> batch) {
  auto sb = batch->back();
  int dimBatch = batch->size();
  int dimWords = batch->back()->batchWidth();
  size_t batchLength = sb->batchWidth() * sb->batchSize();
  std::vector<float> weightedMask(batchLength);
  for(size_t i = 0; i < sb->data().size(); i++) {
    Word w = sb->data()[i];
    weightedMask[i] = wordFreqs_[w];
  }
  return weightedMask;
}

std::vector<float> DynamicWeighting::weightWords(Ptr<data::CorpusBatch> batch) {
  auto sb = batch->back();
  int dimBatch = batch->size();
  int dimWords = batch->back()->batchWidth();
  size_t batchLength = sb->batchWidth() * sb->batchSize();
  size_t batchLength2 = sb->data().size();

  std::vector<float> weightedMask(batchLength);
  for(size_t i = 0; i < batchLength; i++) {
    Word w = sb->data()[i];
    weightedMask[i] = weightFrequency(wordFreqs_[w]);
  }
  return weightedMask;
}

Expr DynamicWeighting::getWeights(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) {
  auto sb = batch->back();
  int dimBatch = batch->size();
  int dimWords = batch->back()->batchWidth();
  size_t batchLength = sb->batchWidth() * sb->batchSize();

  updateWordFrequencies(batch);
  auto weightsVector = weightWords(batch);
  // std::cerr << "before debug" << std::endl;
  // debugWeighting(weightsVector, freqVector, batch);

  Expr weights = graph->constant({1, dimWords, dimBatch, 1},
                                 inits::from_vector(weightsVector));
  return weights;
}

void DynamicWeighting::debugWeighting(std::vector<float> weightedMask,
                                      std::vector<float> freqMask,
                                      Ptr<data::CorpusBatch> batch) {
  std::cerr << "hohohoho weights" << std::endl;
  std::cerr << "original" << std::endl;
  auto sb = batch->back();
  for(size_t i = 0; i < sb->batchWidth(); i++) {
    std::cerr << "\t w: ";
    for(size_t j = 0; j < sb->batchSize(); j++) {
      size_t idx = i * sb->batchSize() + j;
      Word w = sb->data()[idx];
      std::cerr << w << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << "freqs" << std::endl;
  for(size_t i = 0; i < sb->batchWidth(); i++) {
    std::cerr << "\t w: ";
    for(size_t j = 0; j < sb->batchSize(); j++) {
      size_t idx = i * sb->batchSize() + j;
      float w = freqMask[idx];
      std::cerr << w << " ";
    }
    std::cerr << std::endl;
  }
  std::cerr << "weights" << std::endl;
  for(size_t i = 0; i < sb->batchWidth(); i++) {
    std::cerr << "\t w: ";
    for(size_t j = 0; j < sb->batchSize(); j++) {
      size_t idx = i * sb->batchSize() + j;
      float w = weightedMask[idx];
      std::cerr << w << " ";
    }
    std::cerr << std::endl;
  }
}

float ExponentialWeighting::weightFrequency(int64_t freq) {
  // def exponential_equations_fixed(x, a, b, c, d):
  // lrate = up * a * np.exp(b * x)
  // if x > freq:
  // return c * x + d
  // return lrate

  float result;
  float floatFreq = static_cast<float>(freq);
  // std::cerr << "if floatFreq > " << params_[0] << std::endl;
  if(floatFreq > params_[0]) {
    // std::cerr << params_[4] <<  " * floatFreq + " << params_[5] << std::endl;
    result = params_[4] * floatFreq + params_[5];
  } else {
    // std::cerr << params_[1] <<  " * " << params_[2] << " exp(floatFreq * " << params_[3] << ")" << std::endl;
    result = params_[1] * params_[2] * std::exp(params_[3] * floatFreq);
  }
  return result;
}

float InvSqrtWeighting::weightFrequency(int64_t freq) {
// def inv_sqrt_equations(x, a, b, c, d):
// return a / (c * (x**b) + d)

  float result;
  float floatFreq = static_cast<float>(freq);
  result = params_[0] / (params_[2] * (floatFreq, params_[1]) + params_[3]);
  return result;
}
}
