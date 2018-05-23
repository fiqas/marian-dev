#include "layers/weight.h"

namespace marian {

Ptr<WeightingBase> WeightingFactory(Ptr<Options> options) {
  if(options->has("data-weighting"))
    return New<DataWeighting>(options->get<std::string>("data-weighting-type"));
  else if(options->has("dynamic-weighting") && options->get<bool>("dynamic-weighting")) {
    auto vocabSize = options->get<std::vector<int>>("dim-vocabs").back();
    std::cerr << "vocabSize"  << vocabSize << std::endl;
    return New<DynamicWeighting>(vocabSize);
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
      // wordFreqs_.resize(w);
    // }
    wordFreqs_[w]++;
  }
}

float DynamicWeighting::weightFrequency(int64_t freq) {
  float a = 10.0f;
  float b = 1.0f / 4.0f;
  float c = 1.0f;

  float floatFreq = static_cast<float>(freq);

  if(freq != 0.0f) {
      auto result = a / (std::pow(floatFreq, b) * c);
    return result;
  } else
    return 0.0f;
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
  // debugWeighting(weightsVector, freqVector, batch);

  Expr weights
      = graph->constant({1, dimWords, dimBatch, 1},
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

}
