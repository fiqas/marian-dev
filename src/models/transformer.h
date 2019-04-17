// TODO: This is really a .CPP file now. I kept the .H name to minimize confusing git, until this is code-reviewed.
// This is meant to speed-up builds, and to support Ctrl-F7 to rebuild.

#pragma once

#include "marian.h"

#include "layers/constructors.h"
#include "layers/factory.h"
#include "models/decoder.h"
#include "models/encoder.h"
#include "models/states.h"
#include "models/transformer_factory.h"
#include "rnn/constructors.h"

namespace marian {

// clang-format off

// shared base class for transformer-based EncoderTransformer and DecoderTransformer
// Both classes share a lot of code. This template adds that shared code into their
// base while still deriving from EncoderBase and DecoderBase, respectively.
template<class EncoderOrDecoderBase>
class Transformer : public EncoderOrDecoderBase {
  typedef EncoderOrDecoderBase Base;

protected:
  using Base::options_; using Base::inference_; using Base::batchIndex_;
  std::unordered_map<std::string, Expr> cache_;

  // attention weights produced by step()
  // If enabled, it is set once per batch during training, and once per step during translation.
  // It can be accessed by getAlignments(). @TODO: move into a state or return-value object
  std::vector<Expr> alignments_; // [max tgt len or 1][beam depth, max src length, batch size, 1]

  template <typename T> T opt(const std::string& key) const { Ptr<Options> options = options_; return options->get<T>(key); }  // need to duplicate, since somehow using Base::opt is not working
  // FIXME: that separate options assignment is weird

  template <typename T> T opt(const std::string& key, const T& def) const { Ptr<Options> options = options_; if (options->has(key)) return options->get<T>(key); else return def; }

  Ptr<ExpressionGraph> graph_;

public:
  Transformer(Ptr<Options> options)
    : EncoderOrDecoderBase(options) {
  }

  static Expr transposeTimeBatch(Expr input) { return transpose(input, {0, 2, 1, 3}); }

  Expr addPositionalEmbeddings(Expr input, int start = 0, bool trainPosEmbeddings = false) const {
    int dimEmb   = input->shape()[-1];
    int dimWords = input->shape()[-3];

    Expr embeddings = input;

    if(trainPosEmbeddings) {
      int maxLength = opt<int>("max-length");

      // Hack for translating with length longer than trained embeddings
      // We check if the embedding matrix "Wpos" already exist so we can
      // check the number of positions in that loaded parameter.
      // We then have to restict the maximum length to the maximum positon
      // and positions beyond this will be the maximum position.
      Expr seenEmb = graph_->get("Wpos");
      int numPos = seenEmb ? seenEmb->shape()[-2] : maxLength;

      auto embeddingLayer = embedding()
                            ("prefix", "Wpos") // share positional embeddings across all encoders/decorders
                            ("dimVocab", numPos)
                            ("dimEmb", dimEmb)
                            .construct(graph_);

      // fill with increasing numbers until current length or maxPos
      std::vector<IndexType> positions(dimWords, numPos - 1);
      for(int i = 0; i < std::min(dimWords, numPos); ++i)
        positions[i] = i;

      auto signal = embeddingLayer->apply(positions, {dimWords, 1, dimEmb});
      embeddings = embeddings + signal;
    } else {
      // @TODO : test if embeddings should be scaled when trainable
      // according to paper embeddings are scaled up by \sqrt(d_m)
      embeddings = std::sqrt((float)dimEmb) * embeddings;

      auto signal = graph_->constant({dimWords, 1, dimEmb},
                                     inits::sinusoidalPositionEmbeddings(start));
      embeddings = embeddings + signal;
    }

    return embeddings;
  }

  virtual Expr addSpecialEmbeddings(Expr input, int start = 0, Ptr<data::CorpusBatch> /*batch*/ = nullptr) const {
    bool trainPosEmbeddings = opt<bool>("transformer-train-positions", false);
    return addPositionalEmbeddings(input, start, trainPosEmbeddings);
  }

  Expr triangleMask(int length) const {
    // fill triangle mask
    std::vector<float> vMask(length * length, 0);
    for(int i = 0; i < length; ++i)
      for(int j = 0; j <= i; ++j)
        vMask[i * length + j] = 1.f;
    return graph_->constant({1, length, length}, inits::from_vector(vMask));
  }

  // convert multiplicative 1/0 mask to additive 0/-inf log mask, and transpose to match result of bdot() op in Attention()
  static Expr transposedLogMask(Expr mask) { // mask: [-4: beam depth=1, -3: batch size, -2: vector dim=1, -1: max length]
    auto ms = mask->shape();
    mask = (1 - mask) * -99999999.f;
    return reshape(mask, {ms[-3], 1, ms[-2], ms[-1]}); // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
  }

  static Expr SplitHeads(Expr input, int dimHeads) {
    int dimModel = input->shape()[-1];
    int dimSteps = input->shape()[-2];
    int dimBatch = input->shape()[-3];
    int dimBeam  = input->shape()[-4];

    int dimDepth = dimModel / dimHeads;

    auto output
        = reshape(input, {dimBatch * dimBeam, dimSteps, dimHeads, dimDepth});

    return transpose(output, {0, 2, 1, 3}); // [dimBatch*dimBeam, dimHeads, dimSteps, dimDepth]
  }

  static Expr JoinHeads(Expr input, int dimBeam = 1) {
    int dimDepth = input->shape()[-1];
    int dimSteps = input->shape()[-2];
    int dimHeads = input->shape()[-3];
    int dimBatchBeam = input->shape()[-4];

    int dimModel = dimHeads * dimDepth;
    int dimBatch = dimBatchBeam / dimBeam;

    auto output = transpose(input, {0, 2, 1, 3});

    return reshape(output, {dimBeam, dimBatch, dimSteps, dimModel});
  }

  // like affine() but with built-in parameters, activation, and dropout
  static inline
  Expr dense(Expr x, std::string prefix, std::string suffix, int outDim, const std::function<Expr(Expr)>& actFn = nullptr, float dropProb = 0.0f)
  {
    auto graph = x->graph();

    auto W = graph->param(prefix + "_W" + suffix, { x->shape()[-1], outDim }, inits::glorot_uniform);
    auto b = graph->param(prefix + "_b" + suffix, { 1,              outDim }, inits::zeros);

    x = affine(x, W, b);
    if (actFn)
      x = actFn(x);
    if (dropProb)
      x = dropout(x, dropProb);
    return x;
  }

  Expr layerNorm(Expr x, std::string prefix, std::string suffix = std::string()) const {
    int dimModel = x->shape()[-1];
    auto scale = graph_->param(prefix + "_ln_scale" + suffix, { 1, dimModel }, inits::ones);
    auto bias  = graph_->param(prefix + "_ln_bias"  + suffix, { 1, dimModel }, inits::zeros);
    return marian::layerNorm(x, scale, bias, 1e-6f);
  }

  Expr preProcess(std::string prefix, std::string ops, Expr input, float dropProb = 0.0f) const {
    auto output = input;
    for(auto op : ops) {
      // dropout
      if (op == 'd')
        output = dropout(output, dropProb);
      // layer normalization
      else if (op == 'n')
        output = layerNorm(output, prefix, "_pre");
      else
        ABORT("Unknown pre-processing operation '{}'", op);
    }
    return output;
  }

  Expr postProcess(std::string prefix, std::string ops, Expr input, Expr prevInput, float dropProb = 0.0f) const {
    auto output = input;
    for(auto op : ops) {
      // dropout
      if(op == 'd')
        output = dropout(output, dropProb);
      // skip connection
      else if(op == 'a')
        output = output + prevInput;
      // highway connection
      else if(op == 'h') {
        int dimModel = input->shape()[-1];
        auto t = dense(prevInput, prefix, /*suffix=*/"h", dimModel);
        output = highway(output, prevInput, t);
      }
      // layer normalization
      else if(op == 'n')
        output = layerNorm(output, prefix);
      else
        ABORT("Unknown pre-processing operation '{}'", op);
    }
    return output;
  }

  void collectOneHead(Expr weights, int dimBeam) {
    // select first head, this is arbitrary as the choice does not really matter
    auto head0 = slice(weights, -3, 0);

    int dimBatchBeam = head0->shape()[-4];
    int srcWords = head0->shape()[-1];
    int trgWords = head0->shape()[-2];
    int dimBatch = dimBatchBeam / dimBeam;

    // reshape and transpose to match the format guided_alignment expects
    head0 = reshape(head0, {dimBeam, dimBatch, trgWords, srcWords});
    head0 = transpose(head0, {0, 3, 1, 2}); // [-4: beam depth, -3: max src length, -2: batch size, -1: max tgt length]

    // save only last alignment set. For training this will be all alignments,
    // for translation only the last one. Also split alignments by target words.
    // @TODO: make splitting obsolete
    alignments_.clear();
    for(int i = 0; i < trgWords; ++i) {
      alignments_.push_back(slice(head0, -1, i)); // [tgt index][-4: beam depth, -3: max src length, -2: batch size, -1: 1]
    }
  }

  // determine the multiplicative-attention probability and performs the associative lookup as well
  // q, k, and v have already been split into multiple heads, undergone any desired linear transform.
  Expr Attention(std::string  /*prefix*/,
                 Expr q,              // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]
                 Expr k,              // [-4: batch size, -3: num heads, -2: max src length, -1: split vector dim]
                 Expr v,              // [-4: batch size, -3: num heads, -2: max src length, -1: split vector dim]
                 Expr mask = nullptr, // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                 bool saveAttentionWeights = false,
                 int dimBeam = 1,
                 bool reshapeMask = false) {
    int dk = k->shape()[-1];

    // softmax over batched dot product of query and keys (applied over all
    // time steps and batch entries), also add mask for illegal connections
    // // hoho LOG(info, "Attention Q = {}", q->shape());
    // // hoho LOG(info, "Attention K = {}", k->shape());
    // // hoho LOG(info, "Attention V = {}", v->shape());

    // multiplicative attention with flattened softmax
    float scale = 1.0f / std::sqrt((float)dk); // scaling to avoid extreme values due to matrix multiplication
    auto z = bdot(q, k, false, true, scale); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]
    // hoho debug(z, "bdot(Q,K)");

    // hoho LOG(info, "Attention Q * K.T shape = {}", z->shape());
    // hoho LOG(info, "Attention mask.shape = {}", mask->shape());

    auto maskShape = mask->shape();
    bool firstStep = std::all_of(maskShape.begin(), maskShape.end(), [](int i) { return i==1; });
    // hoho LOG(info, "Attention mask firstStep = {}", firstStep);
    
    if (mask && reshapeMask && !firstStep) {
      int beamSize = q->shape()[-5];
      int batchSize = q->shape()[-4];
      auto maskReshape = reshape(mask, {beamSize, batchSize, mask->shape()[-3], mask->shape()[-2], mask->shape()[-1]});
      // hoho LOG(info, "Attention mask.shape AFTER RESHAPE = {}", maskReshape->shape());
      // hoho debug(mask, "MASK BASELINE");
      // hoho debug(maskReshape, "MASK RESHAPE");
      // mask out garbage beyond end of sequences
      // z = z + mask;
      z = z + maskReshape;
    }
    else {
      // hoho debug(mask, "MASK BASELINE");
      z = z + mask;
    }

    // take softmax along src sequence axis (-1)
    auto weights = softmax(z); // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: max src length]
    // hoho debug(weights, "softmax(bdot(Q,K))");
    // hoho LOG(info, "softmax(Q * K.T) shape = {}", weights->shape());

    if(saveAttentionWeights)
      collectOneHead(weights, dimBeam);

    // optional dropout for attention weights
    float dropProb
        = inference_ ? 0 : opt<float>("transformer-dropout-attention");
    weights = dropout(weights, dropProb);

    // apply attention weights to values
    auto output = bdot(weights, v);   // [-4: beam depth * batch size, -3: num heads, -2: max tgt length, -1: split vector dim]
    //HYDRA // hoho debug(output, "bdot(softmax,K)");
    // hoho LOG(info, "Attention output softmax(Q * K.T) * V= {}", output->shape());
    return output;
  }

  Expr SoftmaxGatingNetwork(std::string prefix,
                         Expr input,
                         int dimHeads,
                         int dimModel) {
  
    auto Wg = graph_->param(prefix + "_Wg", {dimModel, dimHeads}, inits::glorot_uniform);
  
    auto WgMult = mean(bdot(input, Wg), -2);
    // hoho debug(input, "input");
    auto gatingSoftmax = softmax(WgMult);
    // hoho debug(gatingSoftmax, "gatingSoftmax in SoftmaxGatingNetwork");
    return gatingSoftmax;
  
  }

  //This one translates foreeeeever and overflows GPU memory
  Expr TopKGatingNetwork(std::string prefix,
                     Expr input,
                     int dimHeads,
                     int K,
                     int dimModel) {


        
    // hoho debug(input, "input_" + prefix);
    // hoho LOG(info, "Entering TopKGatingNetwork");
    auto Wg = graph_->param(prefix + "_Wg", {dimModel, dimHeads}, inits::glorot_uniform);
    auto bg = graph_->param(prefix + "_bg", {1, dimHeads}, inits::zeros);
    
    auto Wnoise = graph_->param(prefix + "_Wnoise", {dimModel, dimHeads}, inits::glorot_uniform);
    auto bnoise = graph_->param(prefix + "_bnoise", {1, dimHeads}, inits::zeros);
    
    auto WgMult = bdot(input, Wg) + bg;
    // hoho LOG(info, "bdot(input, Wg) = {}", WgMult->shape());


    auto WnoiseMult = bdot(input, Wnoise) + bnoise;
    // hoho LOG(info, "bdot(input, Wnoise) = {}", WnoiseMult->shape());
    // auto gatingAvg = mean(gatingMult, -2);
    // // hoho LOG(info, "avg(gatingMult) = {}", gatingAvg->shape());


    auto softplusOut = log(1 + exp(WnoiseMult));
    // auto gatingAvg = mean(gatingMult, -2);
    // // hoho LOG(info, "avg(gatingMult) = {}", gatingAvg->shape());

    std::random_device rd{};
    std::mt19937 gen{rd()};
    // std::mt19937 gen(1811);
    std::normal_distribution<float> d(0, 1);

    // result = Wg_multi + np.random.standard_normal() * softplus_out

    auto gatingResult = mean(WgMult + d(gen) * softplusOut, -2);
    // std::vector<float> haha;


    // gatingResult->val()->get(haha);

    // std::stringstream ss;
    // for(size_t i = 0; i < haha.size(); ++i)
    // {
        // if(i != 0)
              // ss << ",";
          // ss << haha[i];
    // }
    // std::string gatingResultString = ss.str();

    // hoho LOG(info, "gatingResult = {}", gatingResult->shape());
    // // hoho LOG(info, "gatingResultString = {}", gatingResultString);

    // hoho debug(gatingResult, "gatingResult_" + prefix);


    // MASKING MESSES THINGS UP
    // int K = 2; // select K heads for each sentence
    

    // graph_->setInference(true);
    auto gatingResultMasked = gatingResult;
    // auto currentMax = max(gatingResultMasked, -1);
    // graph_->setInference(true);
    auto currentMax = max(stopGradient(gatingResult), -1);
    // graph_->setInference(false);
    Expr mask;
    

    for(int i = 0; i < K - 1; i++) {

      // // // hoho LOG(info, "K = {}", i);
      // // hoho debug(gatingResultMasked, "gatingResultMasked before");
      // // currentMax = max(gatingResultMasked, -1);
      // // hoho debug(currentMax, "currentMax" + std::to_string(i));
      // hoho debug(currentMax, "currentmax");

      // // // if(i < K-1) {
        mask = lt(stopGradient(gatingResultMasked), currentMax);
        // // // // hoho LOG(info, "maaaaask = {}", mask->shape());
        // hoho debug(mask, "mask_" + prefix);
        auto mask2 = (1 - mask) * -99999999.f;
        // // hoho debug(mask, "mask -inf");

        gatingResultMasked = gatingResultMasked + mask2;
        // hoho debug(gatingResultMasked, "gatingResultMasked after " + prefix);
      // // // // }
      // // // //
      currentMax = max(stopGradient(gatingResultMasked), -1);
        // hoho LOG(info, "currentMax = {}", currentMax->shape());
    }

    // std::vector<float> ugh({0, 1, 0, 1, 0, 1, 0, 1});

    auto topKMask = ge(stopGradient(gatingResult), currentMax);
    // graph_->setInference(false);
    // auto gatingOutput = graph_->constant({beamSize * batchSize, 1, dimHeads}, inits::from_vector(gatingInit));
    // auto topKMask = graph_->constant({gatingResult->shape()[0], gatingResult->shape()[1],gatingResult->shape()[2],gatingResult->shape()[3]}, inits::from_vector(ugh));
    // hoho debug(topKMask, "topKMask binary" + prefix);
    // topKMask = (1 - topKMask) * -99999999.f;
    // // hoho LOG(info, "topKMask = {}", topKMask->shape());
    //HYDRA // hoho debug(topKMask, "topKMask");

    // auto maskedGatingOutput = gatingResult + topKMask;
    // // hoho debug(maskedGatingOutput, "maskedGatingOutput");

    // graph_->setInference(false);
    // auto gatingSoftmax = softmax(maskedGatingOutput);
    auto gatingSoftmax = softmax(gatingResult, topKMask);
    //HYDRA // hoho debug(gatingSoftmax, "gatingSoftmax" + prefix);
    // gatingSoftmax = gatingSoftmax;
    // hoho debug(gatingSoftmax, "gatingSoftmax after masking");

    // hoho LOG(info, "gatingSoftmax = {}", gatingSoftmax->shape());
    // hoho LOG(info, "Exiting TopKGatingNetwork");
    return gatingSoftmax;
  }


  Expr Hydra(std::string prefix,
                 int dimOut,
                 int dimHeads,
                 int dimK,
                 int dimHeadSize,
                 Expr q,             // [-4: beam depth * batch size, -3: num heads, -2: max q length, -1: split vector dim]
                 const Expr &keys,   // [-4: beam depth, -3: batch size, -2: max kv length, -1: vector dim]
                 const Expr &values, // [-4: beam depth, -3: batch size, -2: max kv length, -1: vector dim]
                 const Expr &mask,   // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                 bool cache = false,
                 bool saveAttentionWeights = false) {

    int dimModel = q->shape()[-1];
    int maxLengthQuery = q->shape()[-2];
    int maxLengthKeys = keys->shape()[-2];
    int maxLengthValues = values->shape()[-2];
    int batchSize = keys->shape()[-3];
    int beamSize = q->shape()[-4];

    int dimAtt = dimHeads * dimHeadSize;
    
    //STEP 1 - Initialize Wq, Wk, Wv so that every row is a single head
    
    //HYDRA // hoho LOG(info, "mask.shape in FingerPuppet {}", mask->shape()); 
    //HYDRA // hoho LOG(info, "input shape = {}", q->shape());
    //HYDRA // hoho LOG(info, "dimHeads {} dimHeadSize {}", dimHeads, dimHeadSize);
    auto Wq = graph_->param(prefix + "_Wq", {dimHeads, dimModel * dimHeadSize}, inits::glorot_uniform);
    auto bq = graph_->param(prefix + "_bq", {dimHeads, dimHeadSize}, inits::zeros);
   
    auto Wk = graph_->param(prefix + "_Wk", {dimHeads, dimModel * dimHeadSize}, inits::glorot_uniform);
    auto bk = graph_->param(prefix + "_bk", {dimHeads, dimHeadSize}, inits::zeros);
    
    auto Wv = graph_->param(prefix + "_Wv", {dimHeads, dimModel * dimHeadSize}, inits::glorot_uniform);
    auto bv = graph_->param(prefix + "_bv", {dimHeads, dimHeadSize}, inits::zeros);


    //HYDRA // hoho debug(Wq, "graph param Wq_" + prefix);

    // STEP 2 - Initialize gating (constant for now with random binary, seed is set so should return the same vector every time it runs)
    
    // auto gatingOutput = SoftmaxGatingNetwork(prefix, q, dimHeads, dimModel);
    // auto gatingOutput = TopKGatingNetwork(prefix, q, dimHeads, dimK, dimModel);
    //HYDRA // hoho LOG(info, "gatingOutput before reshape= {}", gatingOutput->shape());
    // gatingOutput = reshape(gatingOutput, {beamSize, batchSize, 1, dimHeads});
    // // hoho LOG(info, "gatingOutput after reshape= {}", gatingOutput->shape());
    std::vector<float> gatingInit (beamSize * batchSize * 1 * dimHeads, 1.0); 
    auto gatingOutput = graph_->constant({beamSize, batchSize, 1, dimHeads}, inits::from_vector(gatingInit));
    auto gatingOutputMask = gt(gatingOutput, 0);



    // auto gatingOutputScalars = transpose(reshape(gatingOutput, {beamSize, batchSize, 1, 1, dimHeads}), {0, 1, 4, 2, 3});
    auto gatingOutputScalars = reshape(gatingOutput, {beamSize, batchSize, 1, 1, dimHeads});

    //HYDRA // hoho LOG(info, "gatingOutput after reshape= {}", gatingOutput->shape());
    //HYDRA // hoho LOG(info, "gatingOutputMask = {}", gatingOutputMask->shape());
    //HYDRA // hoho LOG(info, "gatingOutputScalars = {}", gatingOutputScalars->shape());
    //HYDRA // hoho debug(gatingOutput, "gatingOutput" + prefix);
    //HYDRA // hoho debug(gatingOutputScalars, "gatingOutputScalars" + prefix);

    // auto hoho = gt(gatingOutput, 0.0);
    // // hoho debug(hoho);
    
    // STEP 3 - Unionize the selected heads for all sentences
    
    // auto gatingSum = sum(gatingOutput, -3);
    // // hoho LOG(info, "gatingSum.shape = {}", gatingSum->shape());
    // // hoho debug(gatingSum);

    // auto gatingIndices = reshape(gt(gatingSum, 0), {1, dimHeads});
    // // hoho debug(gatingIndices, "gatingIndices");
    // // hoho LOG(info, "gatingIndices.shape = {}", gatingIndices->shape());


    // STEP 4 - Mask heads for each sentence with the gate's output
    // Wq
    auto gatingTransposed = transpose(gatingOutputMask, {0, 1, 3, 2});
    //HYDRA // hoho debug(gatingTransposed, "gatingTransposed");
    //HYDRA // hoho LOG(info, "gatingTransposed = {}", gatingTransposed->shape());
    auto WqMasked = Wq * gatingTransposed;
    //HYDRA // hoho LOG(info, "WqMasked.shape = {}", WqMasked->shape());
    auto bqMasked = reshape(bq * gatingTransposed, {batchSize, dimHeads, 1, dimHeadSize});
    //HYDRA // hoho LOG(info, "bqMasked.shape = {}", bqMasked->shape());
    //HYDRA // hoho debug(WqMasked, "WqMasked_" + prefix);
  

    // Wk
    auto WkMasked = Wk * gatingTransposed;
    auto bkMasked = reshape(bk * gatingTransposed, {batchSize, dimHeads, 1, dimHeadSize});
    
    // Wv
    auto WvMasked = Wv * gatingTransposed;
    auto bvMasked = reshape(bv * gatingTransposed, {batchSize, dimHeads, 1, dimHeadSize});
    
    // STEP 5 - Slice only those heads that are in the selected union
    //
    // TODO Welp, needs to be implemented, because slicing doesnt take Expr and can't iterate 
    // Change gatingOutput to vector and get indices manually???
    

    // STEP 6 - Reshape back into individual heads
    
    auto WqSelected = reshape(WqMasked, {batchSize, 1, dimHeads * dimHeadSize, dimModel});
    auto bqSelected = reshape(bqMasked, {batchSize, 1, dimHeads * dimHeadSize, 1});
    //HYDRA // hoho debug(WqSelected, "WqSelected_" + prefix);
    //HYDRA // hoho debug(bqSelected, "bqSelected_" + prefix);

    //HYDRA // hoho LOG(info, "WqSelected.shape = {}", WqSelected->shape());
    //HYDRA // hoho LOG(info, "bqSelected.shape = {}", bqSelected->shape());

    auto WkSelected = reshape(WkMasked, {batchSize, 1, dimHeads * dimHeadSize, dimModel});
    auto bkSelected = reshape(bkMasked, {batchSize, 1, dimHeads * dimHeadSize, 1});

    auto WvSelected = reshape(WvMasked, {batchSize, 1, dimHeads * dimHeadSize, dimModel});
    auto bvSelected = reshape(bvMasked, {batchSize, 1, dimHeads * dimHeadSize, 1});

    // TODO: Fix?
    //
    
    auto WqSelectedReshaped = transpose(WqSelected, {0, 1, 3, 2});
    auto bqSelectedReshaped = transpose(bqSelected, {0, 1, 3, 2});
    
    auto WkSelectedReshaped = transpose(WkSelected, {0, 1, 3, 2});
    auto bkSelectedReshaped = transpose(bkSelected, {0, 1, 3, 2});

    auto WvSelectedReshaped = transpose(WvSelected, {0, 1, 3, 2});
    auto bvSelectedReshaped = transpose(bvSelected, {0, 1, 3, 2});
    //HYDRA // hoho LOG(info, "WqTransposed.shape = {}", WqSelectedReshaped->shape());
    //HYDRA // hoho LOG(info, "bqTransposed.shape = {}", bqSelectedReshaped->shape());
    //HYDRA // hoho debug(WqSelectedReshaped, "WqSelectedReshaped_" + prefix);
    //HYDRA // hoho debug(bqSelectedReshaped, "bqSelectedReshaped_" + prefix);
    
    // STEP 8 - Calculate Q, K, V
    // Reshape input queries, keys and values to have 1 more dimension to multiply them with my Q, K, V

    auto inputReshape = reshape(q, {beamSize, batchSize, 1, maxLengthQuery, dimModel});
    //HYDRA // hoho debug(inputReshape, "inputReshape");
    //HYDRA // hoho LOG(info, "inputReshape.shape = {}", inputReshape->shape()); 

    auto keysReshape = reshape(keys, {beamSize, batchSize, 1, maxLengthKeys, dimModel});
    //HYDRA // hoho debug(keysReshape, "keysReshape_" + prefix);
    //HYDRA // hoho LOG(info, "keysReshape.shape = {}", keysReshape->shape()); 

    auto valuesReshape = reshape(values, {beamSize, batchSize, 1, maxLengthValues, dimModel});
    //HYDRA // hoho debug(valuesReshape, "valuesReshape_" + prefix);
    //HYDRA // hoho LOG(info, "valuesReshape.shape = {}", valuesReshape->shape()); 
    // auto WqFiltered = index_select(WqMasked, 1, gatingIndices);
    // // hoho LOG(info, "WqFiltered.shape = {}", WqFiltered->shape()); 

    
    auto Q = bdot(inputReshape, WqSelectedReshaped) + bqSelectedReshaped;
    //HYDRA // hoho debug(Q, "Q_" + prefix);
    //HYDRA // hoho LOG(info, "Q before split.shape = {}", Q->shape());

    auto K = bdot(keysReshape, WkSelectedReshaped) + bkSelectedReshaped;
    //HYDRA // hoho debug(K, "K_" + prefix);
    //HYDRA // hoho LOG(info, "K before split.shape = {}", K->shape());
    
    auto V = bdot(valuesReshape, WvSelectedReshaped) + bvSelectedReshaped;
    //HYDRA // hoho debug(V, "V_" + prefix);
    //HYDRA // hoho LOG(info, "V before split.shape = {}", V->shape());
    //HYDRA // hoho LOG(info, "prefix = {}", prefix);
    
    // UNROLL BEAM
    auto QReshape = reshape(Q, {beamSize * batchSize, 1, maxLengthQuery, dimHeads * dimHeadSize});
    //HYDRA // hoho debug(QReshape, "QReshape_" + prefix);
    // hoho LOG(info, "QReshape.shape = {}", QReshape->shape());
    auto KReshape = reshape(K, {beamSize * batchSize, 1, maxLengthKeys, dimHeads * dimHeadSize});
    //HYDRA // hoho debug(KReshape, "KReshape_" + prefix);
    // hoho LOG(info, "KReshape.shape = {}", KReshape->shape());
    auto VReshape = reshape(V, {beamSize * batchSize, 1,  maxLengthValues, dimHeads * dimHeadSize});
    //HYDRA // hoho debug(VReshape, "VReshape_" + prefix);

    // apply multi-head attention to downscaled inputs
    auto output
        = Attention(prefix, QReshape, KReshape, VReshape, mask, saveAttentionWeights, beamSize); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]

    //HYDRA // hoho LOG(info, "output.shape = {}", output->shape());
    //HYDRA // hoho debug(output, "Attention output_" + prefix);
    auto outputReshape = reshape(output, {beamSize, batchSize, maxLengthQuery, dimHeads, dimHeadSize});
    //HYDRA // hoho debug(outputReshape, "Attention reshape first output_" + prefix);
    // auto outputReshapeSecond = reshape(outputReshapeFirst, {beamSize, batchSize, maxLengthQuery, dimHeads, dimHeadSize});
    // // hoho debug(outputReshapeSecond, "Attention reshape first output_" + prefix);
    auto outputTransposed = transpose(outputReshape, {0, 1, 3, 2, 4});
    //HYDRA // hoho debug(outputTransposed, "outputTransposed_" + prefix);
    // hoho LOG(info, "outputReshape.shape = {}", outputReshape->shape());
    auto outputScaled = outputTransposed * gatingOutputScalars;
    //HYDRA // hoho LOG(info, "outputScaled.shape = {}", outputScaled->shape());
    //HYDRA // hoho debug(outputScaled, "outputScaled_" + prefix);
    
    auto outputScaledTransposed = transpose(outputScaled, {0, 1, 3, 2, 4});
    //HYDRA // hoho debug(outputScaledTransposed, "outputScaledTransposed_" + prefix);
    
    auto outputScaledReshape = reshape(outputScaledTransposed, {beamSize, batchSize, maxLengthQuery, dimHeads * dimHeadSize});

    // hoho LOG(info, "outputScaled.shape = {}", outputScaled->shape());
    // auto outputScaledReshape = reshape(outputScaled, {beamSize, batchSize, maxLengthQuery, dimHeads, dimHeadSize});
    // outputScaledReshape = reshape(outputScaledReshape, {beamSize, batchSize, maxLengthQuery, dimHeads * dimHeadSize});
    // outputScaledReshape = reshape(outputScaledReshape, {beamSize * batchSize, maxLengthQuery, dimHeads * dimHeadSize});
    //HYDRA // hoho debug(outputScaledReshape, "outputScaledReshape_" + prefix);
    //HYDRA // hoho LOG(info, "outputScaledReshape.shape = {}", outputScaledReshape->shape());

    // bool project = !opt<bool>("transformer-no-projection");
    // if(project || dimAtt != dimOut) {
    auto Wo
      = graph_->param(prefix + "_Wo", {dimAtt, dimOut}, inits::glorot_uniform);

    //HYDRA // hoho LOG(info, "Wo.shape = {}", Wo->shape());
    auto bo = graph_->param(prefix + "_bo", {1, dimOut}, inits::zeros);
    //HYDRA // hoho LOG(info, "bo.shape = {}", bo->shape());
    auto outputFinal = affine(outputScaledReshape, Wo, bo);
    // auto outputFinal = affine(outputScaled, Wo, bo);
    //HYDRA // hoho LOG(info, "outputFinal.shape = {}", outputFinal->shape());

    //HYDRA // hoho debug(outputFinal, "outputFinal_" + prefix);
    return outputFinal;
  }

  Expr FingerPuppet(std::string prefix,
                 int dimOut,
                 int dimHeads,
                 int dimHeadSize,
                 Expr q,             // [-4: beam depth * batch size, -3: num heads, -2: max q length, -1: split vector dim]
                 const Expr &keys,   // [-4: beam depth, -3: batch size, -2: max kv length, -1: vector dim]
                 const Expr &values, // [-4: beam depth, -3: batch size, -2: max kv length, -1: vector dim]
                 const Expr &mask,   // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                 bool cache = false,
                 bool saveAttentionWeights = false,
                 bool loadBase = true) {
    
    int dimModel = q->shape()[-1];
    int maxLengthQuery = q->shape()[-2];
    int maxLengthKeys = keys->shape()[-2];
    int maxLengthValues = values->shape()[-2];
    int batchSize = keys->shape()[-3];
    int beamSize = q->shape()[-4];

    // hoho debug(q, "input"); 
    // hoho LOG(info, "mask.shape in FingerPuppet {}", mask->shape()); 
    // hoho LOG(info, "input shape = {}", q->shape());
    // hoho LOG(info, "dimHeads {} dimHeadSize {}", dimHeads, dimHeadSize);
    // hoho LOG(info, "prefix = {}", prefix);
    int dimAtt = dimHeads * dimHeadSize;


    // INITIALIZE GATING NETWORK
    
    // auto gatingOutput = SoftmaxGatingNetwork(prefix, q, dimHeads, dimModel);
    auto gatingOutput = TopKGatingNetwork(prefix, q, dimHeads, 4, dimModel);
    // gatingOutput = reshape(gatingOutput, {beamSize * batchSize, 1, dimHeads});
    //

    // std::vector<float> ugh(dimHeads * batchSize)
    // std::vector<float> gatingInit (beamSize * batchSize * 1 * dimHeads) =
      // {1, 1, 1, 1, 0, 0, 0, 0,
       // 1, 1, 1, 0, 0, 1, 0, 0,
       // 0, 1, 1, 1, 0, 1, 0, 0}; 
    // auto gatingOutput = graph_->constant({beamSize * batchSize, 1, dimHeads}, inits::from_vector(gatingInit));
    // auto gatingOutput = graph_->constant({beamSize * batchSize, dimHeads}, inits::from_vector(gatingInit));
    auto gatingOutputMask = gt(gatingOutput, 0);


    // auto gatingVectorExpr = reshape(gatingOutputMask, {dimHeads});

   // // hoho LOG(info, "gatingVectorExpr shape = {}", gatingVectorExpr->shape()); 
    // std::vector<float> gatingVectorize;

    // gatingOutput->val()->get(gatingVectorize);

    // std::stringstream ss;
    // for(size_t i = 0; i < gatingVectorize.size(); ++i)
    // {
        // if(i != 0)
              // ss << ",";
          // ss << gatingVectorize[i];
    // }
    // std::string s = ss.str();

    // // hoho LOG(info, "vector length = {}", gatingVectorize.size());
    // // hoho LOG(info, "vectorized gatingOutputMask = {}", s);

    // auto gatingOutputScalars = transpose(reshape(gatingOutput, {beamSize, batchSize, 1, dimHeads, 1}), {0, 1, 4, 2, 3});
    auto gatingOutputScalars = reshape(gatingOutput, {beamSize, batchSize, dimHeads, 1, 1});

    // hoho LOG(info, "gatingOutput after reshape= {}", gatingOutput->shape());
    // hoho LOG(info, "gatingOutputMask = {}", gatingOutputMask->shape());
    // hoho LOG(info, "gatingOutputScalars = {}", gatingOutputScalars->shape());
    // hoho debug(gatingOutputScalars, "gatingOutputScalars_" + prefix);
    auto gatingTransposed = reshape(gatingOutputMask, {beamSize * batchSize, 1, dimHeads, 1});

    // hoho debug(gatingTransposed, "gatingTransposed");
    // hoho LOG(info, "gatingTransposed = {}", gatingTransposed->shape());


    ////////////////////////////////////////////////////////////////////////////////////////////////////


    // CALCULATE Q (which we should do always because query is new)
   
    // Initialize Wq, bq 
    auto Wq = graph_->param(prefix + "_Wq", {dimModel, dimHeads * dimHeadSize}, inits::glorot_uniform);
    auto bq = graph_->param(prefix + "_bq", {       1, dimHeads * dimHeadSize}, inits::zeros);
    // hoho debug(Wq, "Wq_" + prefix);
    // hoho debug(bq, "Wq_" + prefix);

    // Mask Wq by gating network output
    auto WqReshape = reshape(Wq, {dimModel, dimHeads, dimHeadSize});
    auto bqReshape = reshape(bq, {1, dimHeads, dimHeadSize});

    // auto bqSlice = index_select(bqReshape, 1, gatingVectorExpr);
    // // hoho LOG(info, "bqSlice shape = {}", bqSlice);

    // hoho LOG(info, "WqReshape = {}", WqReshape->shape());
    // hoho LOG(info, "bqReshape = {}", bqReshape->shape());
    auto WqMasked = WqReshape * gatingTransposed;
    auto bqMasked = reshape(bqReshape * gatingTransposed, {batchSize, dimHeads, 1, dimHeadSize});
    
    // hoho LOG(info, "WqMasked.shape = {}", WqMasked->shape());
    // hoho LOG(info, "bqMasked.shape = {}", bqMasked->shape());
    // hoho debug(WqMasked, "WqMasked_" + prefix);
    // hoho debug(bqMasked, "bqMasked_" + prefix);
    

    // Reshape back into individual heads, prepare dimensions of Wq for bdot later
    auto WqSelectedReshaped = transpose(WqMasked, {0, 2, 1, 3});
    auto WqBeam = reshape(WqSelectedReshaped, {1, batchSize, dimHeads, dimModel, dimHeadSize});
    auto WqRepeat = repeat(WqBeam, beamSize, -5);
    // auto bqSelectedReshaped = transpose(bqMasked, {0, 2, 1, 3});
    auto bqSelectedReshaped = bqMasked;

    // hoho debug(WqSelectedReshaped, "WqSelectedReshaped_" + prefix);
    // // hoho debug(WqBeam, "WqBeam_" + prefix);
    // // hoho debug(WqRepeat, "WqRepeat_" + prefix);
    // hoho debug(WqSelectedReshaped, "WqSelectedReshaped_" + prefix);
    // hoho debug(bqSelectedReshaped, "bqSelectedReshaped_" + prefix);
    // hoho LOG(info, "WqSelectedReshaped.shape = {}", WqSelectedReshaped->shape());
    // hoho LOG(info, "WqBeam.shape = {}", WqBeam->shape());
    // hoho LOG(info, "WqRepeat.shape = {}", WqRepeat->shape());
    // hoho LOG(info, "bqSelectedReshaped.shape = {}", bqSelectedReshaped->shape());
    

    // Prepare input dimensions for bdot later
    auto inputReshape = reshape(q, {beamSize, batchSize, 1, maxLengthQuery, dimModel});
    // // hoho debug(inputReshape, "inputReshape");
    // hoho LOG(info, "inputReshape.shape = {}", inputReshape->shape()); 
    
    auto inputRepeat = repeat(inputReshape, 8, -3);
    // hoho LOG(info, "inputRepeat = {}", inputRepeat->shape());

    
    // Calculate Q itself
    
    auto Q = bdot(inputRepeat, WqRepeat) + bqSelectedReshaped;
    // auto Q = affine(inputRepeat, WqSelectedReshaped, bqSelectedReshaped);
    // hoho debug(Q, "Q_" + prefix);
    // hoho LOG(info, "Q.shape = {}", Q->shape());


    ////////////////////////////////////////////////////////////////////////////////////////////////////


    // Calculate K and V or take them from cache

    Expr K;
    // Caching transformation of the encoder that should not be created again.
    // @TODO: set this automatically by memoizing encoder context and
    // memoization propagation (short-term)
    if (!cache || (cache && cache_.count(prefix + "_keys") == 0)) {
      
      // hoho LOG(info, "K calculating...");
      // Initialize Wk, bk
      auto Wk = graph_->param(prefix + "_Wk", {dimModel, dimHeads * dimHeadSize}, inits::glorot_uniform);
      auto bk = graph_->param(prefix + "_bk", {       1, dimHeads * dimHeadSize}, inits::zeros);
      
      
      // Mask Wk by gating network output
      auto WkReshape = reshape(Wk, {dimModel, dimHeads, dimHeadSize});
      auto bkReshape = reshape(bk, {1, dimHeads, dimHeadSize});

      auto WkMasked = WkReshape * gatingTransposed;
      auto bkMasked = reshape(bkReshape * gatingTransposed, {batchSize, dimHeads, 1, dimHeadSize});
      // hoho LOG(info, "WkMasked.shape = {}", WkMasked->shape());
      // hoho LOG(info, "bkMasked.shape = {}", bkMasked->shape());
      // // hoho debug(WkMasked, "WkMasked_" + prefix);
      // // hoho debug(bkMasked, "bkMasked_" + prefix);
    
      // Reshape back into individual heads, prepare dimensions of Wk for bdot later
      auto WkSelectedReshaped = transpose(WkMasked, {0, 2, 1, 3});
      auto WkBeam = reshape(WkSelectedReshaped, {1, batchSize, dimHeads, dimModel, dimHeadSize});
      auto WkRepeat = repeat(WkBeam, beamSize, -5);
      auto bkSelectedReshaped = bkMasked;
      // auto bkSelectedReshaped = transpose(bkMasked, {0, 2, 1, 3});
      
      
      // Prepare input dimensions for bdot later
      auto keysReshape = reshape(keys, {beamSize, batchSize, 1, maxLengthKeys, dimModel});
      auto keysRepeat = repeat(keysReshape, 8, -3);
      // hoho LOG(info, "keysReshape.shape = {}", keysReshape->shape()); 

      
      // Calculate K itself
      K = bdot(keysRepeat, WkRepeat) + bkSelectedReshaped;
      // auto K = affine(keysRepeat, WkSelectedReshaped, bkSelectedReshaped);
      // hoho debug(K, "K_" + prefix);
      // hoho LOG(info, "K.shape = {}", K->shape());
      
      cache_[prefix + "_keys"] = K;
      // hoho debug(K, "K CALCULATED");
    }
    else {
      // hoho LOG(info, "K from cache");
      K = cache_[prefix + "_keys"];
      // hoho debug(K, "K CACHE");
    }

    // // hoho LOG(info, "Finished K = {}", K->shape());

    Expr V;
    if (!cache || (cache && cache_.count(prefix + "_values") == 0)) {
      // hoho LOG(info, "K calculating...");
      
      // Initialize Wv, bv
      auto Wv = graph_->param(prefix + "_Wv", {dimModel, dimHeads * dimHeadSize}, inits::glorot_uniform);
      auto bv = graph_->param(prefix + "_bv", {       1, dimHeads * dimHeadSize}, inits::zeros);
      
      
      // Mask Wv by gating network output
      auto WvReshape = reshape(Wv, {dimModel, dimHeads, dimHeadSize});
      auto bvReshape = reshape(bv, {1, dimHeads, dimHeadSize});
      
      auto WvMasked = WvReshape * gatingTransposed;
      auto bvMasked = reshape(bvReshape * gatingTransposed, {batchSize, dimHeads, 1, dimHeadSize});
      // hoho LOG(info, "WvMasked.shape = {}", WvMasked->shape());
      // hoho LOG(info, "bvMasked.shape = {}", bvMasked->shape());
      // // hoho debug(WvMasked, "WvMasked_" + prefix);
      // // hoho debug(bvMasked, "bvMasked_" + prefix);
     

      // Reshape back into individual heads, prepare dimensions of Wv for bdot later
      auto WvSelectedReshaped = transpose(WvMasked, {0, 2, 1, 3});
      auto WvBeam = reshape(WvSelectedReshaped, {1, batchSize, dimHeads, dimModel, dimHeadSize});
      auto WvRepeat = repeat(WvBeam, beamSize, -5);
      auto bvSelectedReshaped = bvMasked;
      // auto bvSelectedReshaped = transpose(bvMasked, {0, 2, 1, 3});


      // Prepare input dimensions for bdot later

      auto valuesReshape = reshape(values, {beamSize, batchSize, 1, maxLengthValues, dimModel});
      auto valuesRepeat = repeat(valuesReshape, 8, -3);
      // hoho LOG(info, "valuesReshape.shape = {}", valuesReshape->shape()); 


      // Calculate V itself
      V = bdot(valuesRepeat, WvRepeat) + bvSelectedReshaped;
      // auto V = affine(valuesRepeat, WvSelectedReshaped, bvSelectedReshaped);
      // hoho debug(V, "V_" + prefix);
      // hoho LOG(info, "V.shape = {}", V->shape());

      cache_[prefix + "_values"] = V;
      // hoho debug(V, "V CALCULATED");
      // hoho LOG(info, "CACHE V PASSED");
    } else {
      // hoho LOG(info, "V from cache");
      V = cache_[prefix + "_values"];
      // hoho debug(V, "V CACHE");
    }

    // hoho LOG(info, "Finished V = {}", V->shape());
    // // hoho LOG(info, "Finished Q, K, V = {}", Q->shape(), K->shape(), V->shape());
    // Apply multi-head attention
    auto output
        = Attention(prefix, Q, K, V, mask, saveAttentionWeights, beamSize, true); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]
    // hoho debug(output, "Attention output_" + prefix);
    // hoho LOG(info, "output.shape = {}", output->shape());
    output = output * gatingOutputScalars;
    // hoho LOG(info, "output * gatingOutputScalars.shape = {}", output->shape());
    // hoho debug(output, "output * gatingOutputScalars_" + prefix);

    // Basically JoinHeads here
    auto outputTransposed = transpose(output, {0, 1, 3, 2, 4});
    // hoho LOG(info, "outputTransposed.shape = {}", outputTransposed->shape());
    auto outputConcat = reshape(outputTransposed, {beamSize, batchSize, maxLengthQuery, dimHeadSize * dimHeads});
    // hoho LOG(info, "outputConcat.shape = {}", outputConcat->shape());

    Expr outputFinal;
    bool project = !opt<bool>("transformer-no-projection");
    if(project || dimAtt != dimOut) {

      // hoho LOG(info, "DOING PROJECTION WITH Wo");
      auto Wo
        = graph_->param(prefix + "_Wo", {dimAtt, dimOut}, inits::glorot_uniform);
      auto bo = graph_->param(prefix + "_bo", {1, dimOut}, inits::zeros);

      // hoho LOG(info, "Wo.shape = {}", Wo->shape());
      // hoho LOG(info, "bo.shape = {}", bo->shape());
      outputFinal = affine(outputConcat, Wo, bo);
      // hoho LOG(info, "outputFinal.shape = {}", outputFinal->shape());
    }
    else {
      outputFinal = outputConcat;
      // hoho LOG(info, "outputFinal.shape = {}", outputFinal->shape());
    }

      // hoho debug(outputFinal, "outputFinal_" + prefix);
      return outputFinal;
  }

  Expr MultiHead(std::string prefix,
                 int dimOut,
                 int dimHeads,
                 int dimHeadSize,
                 Expr q,             // [-4: beam depth * batch size, -3: num heads, -2: max q length, -1: split vector dim]
                 const Expr &keys,   // [-4: beam depth, -3: batch size, -2: max kv length, -1: vector dim]
                 const Expr &values, // [-4: beam depth, -3: batch size, -2: max kv length, -1: vector dim]
                 const Expr &mask,   // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                 bool cache = false,
                 bool saveAttentionWeights = false) {
    int dimModel = q->shape()[-1];
    // @TODO: good opportunity to implement auto-batching here or do something manually?
    auto Wq = graph_->param(prefix + "_Wq", {dimModel, dimHeads * dimHeadSize}, inits::glorot_uniform);
    auto bq = graph_->param(prefix + "_bq", {       1, dimHeads * dimHeadSize}, inits::zeros);
    auto qh = affine(q, Wq, bq);
    // hoho debug(q, "input");
    // hoho LOG(info, "qh shape = {}", qh->shape());
    // hoho debug(Wq, "Wq_" + prefix);
    // hoho debug(bq, "bq_" + prefix);
    // hoho debug(qh, "Q before splitHeads_" + prefix);
    qh = SplitHeads(qh, dimHeads); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]
    // hoho LOG(info, "Q shape = {}", qh->shape());
    // hoho debug(qh, "Q after splitHeads_" + prefix);

    Expr kh;
    // Caching transformation of the encoder that should not be created again.
    // @TODO: set this automatically by memoizing encoder context and
    // memoization propagation (short-term)
    if (!cache || (cache && cache_.count(prefix + "_keys") == 0)) {
      auto Wk = graph_->param(prefix + "_Wk", {dimModel, dimHeads * dimHeadSize}, inits::glorot_uniform);
      auto bk = graph_->param(prefix + "_bk", {1,        dimHeads * dimHeadSize}, inits::zeros);

      kh = affine(keys, Wk, bk);     // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
      kh = SplitHeads(kh, dimHeads); // [-4: batch size, -3: num heads, -2: max length, -1: split vector dim]
      cache_[prefix + "_keys"] = kh;
      // hoho debug(kh, "K CALCULATED");
    }
    else {
      kh = cache_[prefix + "_keys"] * 1;
      // hoho debug(kh, "K CACHE");
    }

    Expr vh;
    if (!cache || (cache && cache_.count(prefix + "_values") == 0)) {
      auto Wv = graph_->param(prefix + "_Wv", {dimModel, dimHeads * dimHeadSize}, inits::glorot_uniform);
      auto bv = graph_->param(prefix + "_bv", {1,        dimHeads * dimHeadSize}, inits::zeros);

      vh = affine(values, Wv, bv); // [-4: batch size, -3: num heads, -2: max length, -1: split vector dim]
      vh = SplitHeads(vh, dimHeads);
      cache_[prefix + "_values"] = vh;
      // hoho debug(vh, "V CALCULATED");
    } else {
      vh = cache_[prefix + "_values"] * 1;
      // hoho debug(vh, "V CACHE");
    }

    int dimBeam = q->shape()[-4];

    // apply multi-head attention to downscaled inputs
    auto output
        = Attention(prefix, qh, kh, vh, mask, saveAttentionWeights, dimBeam); // [-4: beam depth * batch size, -3: num heads, -2: max length, -1: split vector dim]
    // hoho LOG(info, "output.shape = {}", output->shape());

    // hoho debug(output, "Attention output_" + prefix);
    
    output = JoinHeads(output, dimBeam); // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]

    // hoho debug(output, "Attention output after JoinHeads_" + prefix);
    // hoho LOG(info, "output after joinHeads shape = {}", output->shape());
    int dimAtt = output->shape()[-1];

    bool project = !opt<bool>("transformer-no-projection");
    if(project || dimAtt != dimOut) {
      auto Wo
        = graph_->param(prefix + "_Wo", {dimAtt, dimOut}, inits::glorot_uniform);
      auto bo = graph_->param(prefix + "_bo", {1, dimOut}, inits::zeros);
      output = affine(output, Wo, bo);
    }

    // hoho LOG(info, "projected output affine = {}", output->shape());
    // hoho debug(output, "outputFinal_" + prefix);
    return output;
  }

  Expr LayerAttention(std::string prefix,
                      Expr input,         // [-4: beam depth, -3: batch size, -2: max length, -1: vector dim]
                      const Expr& keys,   // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
                      const Expr& values, // ...?
                      const Expr& mask,   // [-4: batch size, -3: num heads broadcast=1, -2: max length broadcast=1, -1: max length]
                      bool cache = false,
                      bool saveAttentionWeights = false) {
    int dimModel = input->shape()[-1];

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    auto opsPre = opt<std::string>("transformer-preprocess");
    auto output = preProcess(prefix + "_Wo", opsPre, input, dropProb);

    // Number of heads
    auto heads = opt<int>("transformer-heads");
    auto headDim = opt<int>("transformer-head-dim");
    auto K = opt<int>("transformer-select-heads");
    auto type = opt<std::string>("transformer-attention");

    auto load = opt<bool>("transformer-load-base");

    // hoho LOG(info, "input just before FingerPuppet = {}", output->shape());
    // multi-head self-attention over previous input
    if (type == "finger-puppet") {
      output = FingerPuppet(prefix, dimModel, heads, headDim, output, keys, values, mask, cache, saveAttentionWeights, load);
    }
    else if (type == "hydra") {
      output = Hydra(prefix, dimModel, heads, K, headDim, output, keys, values, mask, cache, saveAttentionWeights);
    }
    else {
      output = MultiHead(prefix, dimModel, heads, headDim, output, keys, values, mask, cache, saveAttentionWeights);
    }

    auto opsPost = opt<std::string>("transformer-postprocess");
    output = postProcess(prefix + "_Wo", opsPost, output, input, dropProb);

    return output;
  }

  Expr DecoderLayerSelfAttention(rnn::State& decoderLayerState,
                                 const rnn::State& prevdecoderLayerState,
                                 std::string prefix,
                                 Expr input,
                                 Expr selfMask,
                                 int startPos) {
    selfMask = transposedLogMask(selfMask);

    auto values = input;
    if(startPos > 0) {
      values = concatenate({prevdecoderLayerState.output, input}, /*axis=*/-2);
    }
    decoderLayerState.output = values;

    return LayerAttention(prefix, input, values, values, selfMask,
                          /*cache=*/false);
  }

  static inline
  std::function<Expr(Expr)> activationByName(const std::string& actName)
  {
    if (actName == "relu")
      return (ActivationFunction*)relu;
    else if (actName == "swish")
      return (ActivationFunction*)swish;
    else if (actName == "gelu")
      return (ActivationFunction*)gelu;
    ABORT("Invalid activation name '{}'", actName);
  }

  Expr LayerFFN(std::string prefix, Expr input) const {
    int dimModel = input->shape()[-1];

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    auto opsPre = opt<std::string>("transformer-preprocess");
    auto output = preProcess(prefix + "_ffn", opsPre, input, dropProb);

    int dimFfn = opt<int>("transformer-dim-ffn");
    int depthFfn = opt<int>("transformer-ffn-depth");
    auto actFn = activationByName(opt<std::string>("transformer-ffn-activation"));
    float ffnDropProb
      = inference_ ? 0 : opt<float>("transformer-dropout-ffn");

    ABORT_IF(depthFfn < 1, "Filter depth {} is smaller than 1", depthFfn);

    // the stack of FF layers
    for(int i = 1; i < depthFfn; ++i)
      output = dense(output, prefix, /*suffix=*/std::to_string(i), dimFfn, actFn, ffnDropProb);
    output = dense(output, prefix, /*suffix=*/std::to_string(depthFfn), dimModel);

    auto opsPost = opt<std::string>("transformer-postprocess");
    output
      = postProcess(prefix + "_ffn", opsPost, output, input, dropProb);

    return output;
  }

  // Implementation of Average Attention Network Layer (AAN) from
  // https://arxiv.org/pdf/1805.00631.pdf
  Expr LayerAAN(std::string prefix, Expr x, Expr y) const {
    int dimModel = x->shape()[-1];

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    auto opsPre = opt<std::string>("transformer-preprocess");

    y = preProcess(prefix + "_ffn", opsPre, y, dropProb);

    // FFN
    int dimAan   = opt<int>("transformer-dim-aan");
    int depthAan = opt<int>("transformer-aan-depth");
    auto actFn = activationByName(opt<std::string>("transformer-aan-activation"));
    float aanDropProb = inference_ ? 0 : opt<float>("transformer-dropout-ffn");

    // the stack of AAN layers
    for(int i = 1; i < depthAan; ++i)
      y = dense(y, prefix, /*suffix=*/std::to_string(i), dimAan, actFn, aanDropProb);
    if(y->shape()[-1] != dimModel) // bring it back to the desired dimension if needed
      y = dense(y, prefix, std::to_string(depthAan), dimModel);

    bool noGate = opt<bool>("transformer-aan-nogate");
    if(!noGate) {
      auto gi = dense(x, prefix, /*suffix=*/"i", dimModel, (ActivationFunction*)sigmoid);
      auto gf = dense(y, prefix, /*suffix=*/"f", dimModel, (ActivationFunction*)sigmoid);
      y = gi * x + gf * y;
    }

    auto opsPost = opt<std::string>("transformer-postprocess");
    y = postProcess(prefix + "_ffn", opsPost, y, x, dropProb);

    return y;
  }

  // Implementation of Average Attention Network Layer (AAN) from
  // https://arxiv.org/pdf/1805.00631.pdf
  // Function wrapper using decoderState as input.
  Expr DecoderLayerAAN(rnn::State& decoderState,
                       const rnn::State& prevDecoderState,
                       std::string prefix,
                       Expr input,
                       Expr selfMask,
                       int startPos) const {
    auto output = input;
    if(startPos > 0) {
      // we are decoding at a position after 0
      output = (prevDecoderState.output * (float)startPos + input) / float(startPos + 1);
    }
    else if(startPos == 0 && output->shape()[-2] > 1) {
      // we are training or scoring, because there is no history and
      // the context is larger than a single time step. We do not need
      // to average batch with only single words.
      selfMask = selfMask / sum(selfMask, /*axis=*/-1);
      output = bdot(selfMask, output);
    }
    decoderState.output = output; // BUGBUG: mutable?

    return LayerAAN(prefix, input, output);
  }

  Expr DecoderLayerRNN(rnn::State& decoderState,
                       const rnn::State& prevDecoderState,
                       std::string prefix,
                       Expr input,
                       Expr /*selfMask*/,
                       int /*startPos*/) const {
    float dropoutRnn = inference_ ? 0.f : opt<float>("dropout-rnn");

    auto rnn = rnn::rnn()                                          //
        ("type", opt<std::string>("dec-cell"))                     //
        ("prefix", prefix)                                         //
        ("dimInput", opt<int>("dim-emb"))                          //
        ("dimState", opt<int>("dim-emb"))                          //
        ("dropout", dropoutRnn)                                    //
        ("layer-normalization", opt<bool>("layer-normalization"))  //
        .push_back(rnn::cell())                                    //
        .construct(graph_);

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    auto opsPre = opt<std::string>("transformer-preprocess");
    auto output = preProcess(prefix, opsPre, input, dropProb);

    output = transposeTimeBatch(output);
    output = rnn->transduce(output, prevDecoderState);
    decoderState = rnn->lastCellStates()[0];
    output = transposeTimeBatch(output);

    auto opsPost = opt<std::string>("transformer-postprocess");
    output = postProcess(prefix + "_ffn", opsPost, output, input, dropProb);

    return output;
  }
};

class EncoderTransformer : public Transformer<EncoderBase> {
public:
  EncoderTransformer(Ptr<Options> options) : Transformer(options) {}
  virtual ~EncoderTransformer() {}

  // returns the embedding matrix based on options
  // and based on batchIndex_.

  Ptr<IEmbeddingLayer> createULREmbeddingLayer() const {
    // standard encoder word embeddings
    int dimSrcVoc = opt<std::vector<int>>("dim-vocabs")[0];  //ULR multi-lingual src
    int dimTgtVoc = opt<std::vector<int>>("dim-vocabs")[1];  //ULR monon tgt
    int dimEmb = opt<int>("dim-emb");
    int dimUlrEmb = opt<int>("ulr-dim-emb");
    auto embFactory = ulr_embedding()("dimSrcVoc", dimSrcVoc)("dimTgtVoc", dimTgtVoc)
                                     ("dimUlrEmb", dimUlrEmb)("dimEmb", dimEmb)
                                     ("ulrTrainTransform", opt<bool>("ulr-trainable-transformation"))
                                     ("ulrQueryFile", opt<std::string>("ulr-query-vectors"))
                                     ("ulrKeysFile", opt<std::string>("ulr-keys-vectors"));
    return embFactory.construct(graph_);
  }

  Ptr<IEmbeddingLayer> createWordEmbeddingLayer(size_t subBatchIndex) const {
    // standard encoder word embeddings
    int dimVoc = opt<std::vector<int>>("dim-vocabs")[subBatchIndex];
    int dimEmb = opt<int>("dim-emb");
    auto embFactory = embedding()("dimVocab", dimVoc)("dimEmb", dimEmb);
    if(opt<bool>("tied-embeddings-src") || opt<bool>("tied-embeddings-all"))
      embFactory("prefix", "Wemb");
    else
      embFactory("prefix", prefix_ + "_Wemb");
    if(options_->has("embedding-fix-src"))
      embFactory("fixed", opt<bool>("embedding-fix-src"));
    if(options_->hasAndNotEmpty("embedding-vectors")) {
      auto embFiles = opt<std::vector<std::string>>("embedding-vectors");
      embFactory("embFile", embFiles[subBatchIndex])
                ("normalization", opt<bool>("embedding-normalization"));
    }
    return embFactory.construct(graph_);
  }

  virtual Ptr<EncoderState> build(Ptr<ExpressionGraph> graph,
                                  Ptr<data::CorpusBatch> batch) override {
    graph_ = graph;
    // hoho LOG(info, "batch size = {}", batch->size());
    return apply(batch);
  }

  virtual Ptr<EncoderState> apply(Ptr<data::CorpusBatch> batch) {
    int dimBatch = (int)batch->size();
    int dimSrcWords = (int)(*batch)[batchIndex_]->batchWidth();
    // create the embedding matrix, considering tying and some other options
    // embed the source words in the batch
    Expr batchEmbeddings, batchMask;
    Ptr<IEmbeddingLayer> embedding;
    if (options_->has("ulr") && options_->get<bool>("ulr") == true)
      embedding = createULREmbeddingLayer(); // embedding uses ULR
    else
      embedding = createWordEmbeddingLayer(batchIndex_);
    std::tie(batchEmbeddings, batchMask) = embedding->apply((*batch)[batchIndex_]);

    // apply dropout over source words
    float dropoutSrc = inference_ ? 0 : opt<float>("dropout-src");
    if(dropoutSrc) {
      int srcWords = batchEmbeddings->shape()[-3];
      batchEmbeddings = dropout(batchEmbeddings, dropoutSrc, {srcWords, 1, 1});
    }

    batchEmbeddings = addSpecialEmbeddings(batchEmbeddings, /*start=*/0, batch);

    // reorganize batch and timestep
    batchEmbeddings = atleast_nd(batchEmbeddings, 4);
    batchMask = atleast_nd(batchMask, 4);
    auto layer = transposeTimeBatch(batchEmbeddings); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
    auto layerMask
      = reshape(transposeTimeBatch(batchMask), {1, dimBatch, 1, dimSrcWords}); // [-4: beam depth=1, -3: batch size, -2: vector dim=1, -1: max length]

    auto opsEmb = opt<std::string>("transformer-postprocess-emb");

    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");
    layer = preProcess(prefix_ + "_emb", opsEmb, layer, dropProb);

    layerMask = transposedLogMask(layerMask); // [-4: batch size, -3: 1, -2: vector dim=1, -1: max length]

    // apply encoder layers
    auto encDepth = opt<int>("enc-depth");
    for(int i = 1; i <= encDepth; ++i) {
      layer = LayerAttention(prefix_ + "_l" + std::to_string(i) + "_self",
                             layer, // query
                             layer, // keys
                             layer, // values
                             layerMask);

      layer = LayerFFN(prefix_ + "_l" + std::to_string(i) + "_ffn", layer);
      // hoho LOG(info, "EncoderLayer {}", i);
    }

    // restore organization of batch and time steps. This is currently required
    // to make RNN-based decoders and beam search work with this. We are looking
    // into making this more natural.
    auto context = transposeTimeBatch(layer); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]

    return New<EncoderState>(context, batchMask, batch);
  }

  virtual void clear() override {}
};

class TransformerState : public DecoderState {
public:
  TransformerState(const rnn::States& states,
                   Expr logProbs,
                   const std::vector<Ptr<EncoderState>>& encStates,
                   Ptr<data::CorpusBatch> batch)
      : DecoderState(states, logProbs, encStates, batch) {}

  virtual Ptr<DecoderState> select(const std::vector<IndexType>& selIdx,
                                   int beamSize) const override {
    // Create hypothesis-selected state based on current state and hyp indices
    auto selectedState = New<TransformerState>(states_.select(selIdx, beamSize, /*isBatchMajor=*/true), logProbs_, encStates_, batch_);

    // Set the same target token position as the current state
    // @TODO: This is the same as in base function.
    selectedState->setPosition(getPosition());
    return selectedState;
  }
};

class DecoderTransformer : public Transformer<DecoderBase> {
private:
  Ptr<mlp::Output> output_;

private:
  void lazyCreateOutputLayer()
  {
    if(output_) // create it lazily
      return;

    int dimTrgVoc = opt<std::vector<int>>("dim-vocabs")[batchIndex_];

    auto outputFactory = mlp::output()         //
        ("prefix", prefix_ + "_ff_logit_out")  //
        ("dim", dimTrgVoc);

    if(opt<bool>("tied-embeddings") || opt<bool>("tied-embeddings-all")) {
      std::string tiedPrefix = prefix_ + "_Wemb";
      if(opt<bool>("tied-embeddings-all") || opt<bool>("tied-embeddings-src"))
        tiedPrefix = "Wemb";
      outputFactory.tieTransposed(tiedPrefix);
    }

    output_ = std::dynamic_pointer_cast<mlp::Output>(outputFactory.construct(graph_)); // (construct() returns only the underlying interface)
  }

public:
  DecoderTransformer(Ptr<Options> options) : Transformer(options) {}

  virtual Ptr<DecoderState> startState(
      Ptr<ExpressionGraph> graph,
      Ptr<data::CorpusBatch> batch,
      std::vector<Ptr<EncoderState>>& encStates) override {
    graph_ = graph;

    std::string layerType = opt<std::string>("transformer-decoder-autoreg", "self-attention");
    if (layerType == "rnn") {
      int dimBatch = (int)batch->size();
      int dim = opt<int>("dim-emb");

      auto start = graph->constant({1, 1, dimBatch, dim}, inits::zeros);
      rnn::States startStates(opt<size_t>("dec-depth"), {start, start});

      // don't use TransformerState for RNN layers
      return New<DecoderState>(startStates, nullptr, encStates, batch);
    }
    else {
      rnn::States startStates;
      return New<TransformerState>(startStates, nullptr, encStates, batch);
    }
  }

  virtual Ptr<DecoderState> step(Ptr<ExpressionGraph> graph,
                                 Ptr<DecoderState> state) override {
    ABORT_IF(graph != graph_, "An inconsistent graph parameter was passed to step()");
    lazyCreateOutputLayer();
    return step(state);
  }

  Ptr<DecoderState> step(Ptr<DecoderState> state) {
    auto embeddings  = state->getTargetEmbeddings(); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]
    auto decoderMask = state->getTargetMask();       // [max length, batch size, 1]  --this is a hypothesis

    // // hoho LOG(info, "decoderMask = {}", decoderMask->shape());

    // dropout target words
    float dropoutTrg = inference_ ? 0 : opt<float>("dropout-trg");
    if(dropoutTrg) {
      int trgWords = embeddings->shape()[-3];
      embeddings = dropout(embeddings, dropoutTrg, {trgWords, 1, 1});
    }

    //************************************************************************//

    int dimBeam = 1;
    if(embeddings->shape().size() > 3)
      dimBeam = embeddings->shape()[-4];

    // set current target token position during decoding or training. At training
    // this should be 0. During translation the current length of the translation.
    // Used for position embeddings and creating new decoder states.
    int startPos = (int)state->getPosition();

    auto scaledEmbeddings = addSpecialEmbeddings(embeddings, startPos);
    scaledEmbeddings = atleast_nd(scaledEmbeddings, 4);

    // reorganize batch and timestep
    auto query = transposeTimeBatch(scaledEmbeddings); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]

    auto opsEmb = opt<std::string>("transformer-postprocess-emb");
    float dropProb = inference_ ? 0 : opt<float>("transformer-dropout");

    query = preProcess(prefix_ + "_emb", opsEmb, query, dropProb);

    int dimTrgWords = query->shape()[-2];
    int dimBatch    = query->shape()[-3];
    auto selfMask = triangleMask(dimTrgWords);  // [ (1,) 1, max length, max length]
    // // hoho LOG(info, "self.mask in decoder = {}", selfMask->shape());
    if(decoderMask) {
      decoderMask = atleast_nd(decoderMask, 4);             // [ 1, max length, batch size, 1 ]
      // // hoho LOG(info, "decoderMask atleast_nd = {}", decoderMask->shape());
      decoderMask = reshape(transposeTimeBatch(decoderMask),// [ 1, batch size, max length, 1 ]
                            {1, dimBatch, 1, dimTrgWords}); // [ 1, batch size, 1, max length ]
      // // hoho LOG(info, "decoderMask reshape = {}", decoderMask->shape());
      selfMask = selfMask * decoderMask;
    }

    std::vector<Expr> encoderContexts;
    std::vector<Expr> encoderMasks;

    for(auto encoderState : state->getEncoderStates()) {
      auto encoderContext = encoderState->getContext();
      auto encoderMask = encoderState->getMask();

      encoderContext = transposeTimeBatch(encoderContext); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]

      int dimSrcWords = encoderContext->shape()[-2];

      //int dims = encoderMask->shape().size();
      encoderMask = atleast_nd(encoderMask, 4);
      encoderMask = reshape(transposeTimeBatch(encoderMask),
                            {1, dimBatch, 1, dimSrcWords});
      encoderMask = transposedLogMask(encoderMask);
      if(dimBeam > 1)
        encoderMask = repeat(encoderMask, dimBeam, /*axis=*/ -4);

      encoderContexts.push_back(encoderContext);
      encoderMasks.push_back(encoderMask);
    }

    rnn::States prevDecoderStates = state->getStates();
    rnn::States decoderStates;
    // apply decoder layers
    auto decDepth = opt<int>("dec-depth");
    std::vector<size_t> tiedLayers = opt<std::vector<size_t>>("transformer-tied-layers",
                                                              std::vector<size_t>());
    ABORT_IF(!tiedLayers.empty() && tiedLayers.size() != decDepth,
             "Specified layer tying for {} layers, but decoder has {} layers",
             tiedLayers.size(),
             decDepth);

    for(int i = 0; i < decDepth; ++i) {
      std::string layerNo = std::to_string(i + 1);
      // hoho LOG(info, "DecoderLayer {}", i);
      if (!tiedLayers.empty())
        layerNo = std::to_string(tiedLayers[i]);

      rnn::State prevDecoderState;
      if(prevDecoderStates.size() > 0)
        prevDecoderState = prevDecoderStates[i];

      // self-attention
      std::string layerType = opt<std::string>("transformer-decoder-autoreg", "self-attention");
      rnn::State decoderState;
      if(layerType == "self-attention")
        query = DecoderLayerSelfAttention(decoderState, prevDecoderState, prefix_ + "_l" + layerNo + "_self", query, selfMask, startPos);
      else if(layerType == "average-attention")
        query = DecoderLayerAAN(decoderState, prevDecoderState, prefix_ + "_l" + layerNo + "_aan", query, selfMask, startPos);
      else if(layerType == "rnn")
        query = DecoderLayerRNN(decoderState, prevDecoderState, prefix_ + "_l" + layerNo + "_rnn", query, selfMask, startPos);
      else
        ABORT("Unknown auto-regressive layer type in transformer decoder {}",
              layerType);

      // source-target attention
      // Iterate over multiple encoders and simply stack the attention blocks
      if(encoderContexts.size() > 0) {
        for(size_t j = 0; j < encoderContexts.size(); ++j) { // multiple encoders are applied one after another
          // hoho LOG(info, "encoderContexts j = {}", j);
          // hoho LOG(info, "encoderContexts size = {}", encoderContexts.size());
          std::string prefix
            = prefix_ + "_l" + layerNo + "_context";
          if(j > 0)
            prefix += "_enc" + std::to_string(j + 1);

          // if training is performed with guided_alignment or if alignment is requested during
          // decoding or scoring return the attention weights of one head of the last layer.
          // @TODO: maybe allow to return average or max over all heads?
          bool saveAttentionWeights = false;
          if(j == 0 && (options_->get("guided-alignment", std::string("none")) != "none" || options_->hasAndNotEmpty("alignment"))) {
            size_t attLayer = decDepth - 1;
            std::string gaStr = options_->get<std::string>("transformer-guided-alignment-layer", "last");
            if(gaStr != "last")
              attLayer = std::stoull(gaStr) - 1;

            ABORT_IF(attLayer >= decDepth,
                     "Chosen layer for guided attention ({}) larger than number of layers ({})",
                     attLayer + 1, decDepth);

            saveAttentionWeights = i == attLayer;
          }
          
          // hoho LOG(info, "query shape = {}", query->shape());
          // hoho LOG(info, "keys shape = {}", encoderContexts[j]->shape());
          // hoho LOG(info, "values shape = {}", encoderContexts[j]->shape());
          // hoho LOG(info, "mask shape = {}", encoderMasks[j]->shape());
          // hoho LOG(info, "query shape = {}", query->shape());
          query = LayerAttention(prefix,
                                 query,
                                 encoderContexts[j], // keys
                                 encoderContexts[j], // values
                                 encoderMasks[j],
                                 /*cache=*/true,
                                 saveAttentionWeights);
        }
      }

      // remember decoder state
      decoderStates.push_back(decoderState);

      query = LayerFFN(prefix_ + "_l" + layerNo + "_ffn", query); // [-4: beam depth=1, -3: batch size, -2: max length, -1: vector dim]
    }

    auto decoderContext = transposeTimeBatch(query); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vector dim]

    //************************************************************************//

    // final feed-forward layer (output)
    if(shortlist_)
      output_->setShortlist(shortlist_);
    Expr logits = output_->apply(decoderContext); // [-4: beam depth=1, -3: max length, -2: batch size, -1: vocab or shortlist dim]

    // return unormalized(!) probabilities
    Ptr<DecoderState> nextState;
    if (opt<std::string>("transformer-decoder-autoreg", "self-attention") == "rnn") {
      nextState = New<DecoderState>(
          decoderStates, logits, state->getEncoderStates(), state->getBatch());
    } else {
      nextState = New<TransformerState>(
          decoderStates, logits, state->getEncoderStates(), state->getBatch());
    }
    nextState->setPosition(state->getPosition() + 1);
    return nextState;
  }

  // helper function for guided alignment
  // @TODO: const vector<> seems wrong. Either make it non-const or a const& (more efficient but dangerous)
  virtual const std::vector<Expr> getAlignments(int /*i*/ = 0) override {
    return alignments_;
  }

  void clear() override {
    if (output_)
      output_->clear();
    cache_.clear();
    alignments_.clear();
  }
};

// factory functions
Ptr<EncoderBase> NewEncoderTransformer(Ptr<Options> options)
{
  return New<EncoderTransformer>(options);
}

Ptr<DecoderBase> NewDecoderTransformer(Ptr<Options> options)
{
  return New<DecoderTransformer>(options);
}

// clang-format on

}  // namespace marian
