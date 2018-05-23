#include "layers/weight.h"

namespace marian {

Ptr<WeightingBase> WeightingFactory(Ptr<Options> options) {
  if(options->has("data-weighting"))
    return New<DataWeighting>(options->get<std::string>("data-weighting-type"));
  // else if(options->get<bool>("dynamic-weighting"))
  else if(options->has("dynamic-weighting") && options->get<bool>("dynamic-weighting"))
    return New<DynamicWeighting>();
}

Expr DataWeighting::getWeights(Ptr<ExpressionGraph> graph,
                               Ptr<data::CorpusBatch> batch) {
  ABORT_IF(batch->getDataWeights().empty(),
           "Vector of weights is unexpectedly empty!");
  bool sentenceWeighting = weightingType_ == "sentence";
  int dimBatch = batch->size();
  int dimWords = sentenceWeighting ? 1 : batch->back()->batchWidth();
  auto weights = graph->constant(
      {1, dimWords, dimBatch, 1},
      inits::from_vector(batch->getDataWeights()));
  return weights;
}

void DynamicWeighting::updateWordFrequencies(Ptr<data::CorpusBatch> batch) {
  // std::cerr << "wordFreq size before " << wordFreqs_.size() << std::endl;
  auto sb = batch->back();
  for(size_t i = 0; i < sb->data().size(); i++) {
    Word w = sb->data()[i];
    // std::cerr << "word id " << w << std::endl;
    if(wordFreqs_.size() < w && w != 0) {
      wordFreqs_.resize(w);
      // std::cerr << "wordFreq size after resizing " << wordFreqs_.size()
      // << std::endl;
    }
    wordFreqs_[w]++;
  }
}

float DynamicWeighting::weightFrequency(int64_t freq) {
  // float a = -0.07593015;
  // float b = 0.9990619;
  // float c = 2.09397751;
  float a = 10.0f;
  float b = 1.0f / 4.0f;
  float c = 1.0f;

  // std::cerr << "before static_cast " << freq << std::endl;

  float floatFreq = static_cast<float>(freq);
  // std::cerr << "after static cast " << floatFreq << std::endl;

  if(freq != 0.0f) {
    // auto result = a * std::log(b * floatFreq) + c;
      auto result = a / (std::pow(floatFreq, b) * c);
    // if(!std::isfinite(result))
    // std::cerr << "NAN " << freq << " " << result << std::endl;
    // else
    // std::cerr << "NORMAL " << freq << " " << result << std::endl;
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

  // std::cerr << "batch Length " << batchLength << " " << batchLength2
  // << std::endl;
  std::vector<float> weightedMask(batchLength);
  for(size_t i = 0; i < batchLength; i++) {
    // std::cerr << "trying to get element " << i << std::endl;
    Word w = sb->data()[i];
    // std::cerr << "trying to weight element " << i << std::endl;
    // std::cerr << "weightedMask size " << weightedMask.size() << std::endl;
    // std::cerr << "wordFreqs size " << wordFreqs_.size() << w << std::endl;
    weightedMask[i] = weightFrequency(wordFreqs_[w]);
    // std::cerr << "after weightFrequency " << i << std::endl;
  }
  return weightedMask;
}

Expr DynamicWeighting::getWeights(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) {
  auto sb = batch->back();
  int dimBatch = batch->size();
  int dimWords = batch->back()->batchWidth();
  size_t batchLength = sb->batchWidth() * sb->batchSize();

  // std::cerr << "before update freqs" << std::endl;
  updateWordFrequencies(batch);
  // std::cerr << "before mapWords" << std::endl;
  // auto freqVector = mapWords(batch);
  // std::cerr << "before weightWords" << std::endl;
  auto weightsVector = weightWords(batch);
  // std::cerr << "after weightWords" << std::endl;

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
