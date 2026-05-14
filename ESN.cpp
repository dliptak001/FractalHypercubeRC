#include "ESN.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>

template <size_t DEPTH>
ESN<DEPTH>::ESN(const FractalReservoirConfig<DEPTH>& fractal_cfg)
    : fractal_(FractalReservoir<DEPTH>::Create(fractal_cfg)),
      fractal_cfg_(fractal_cfg),
      num_inputs_(fractal_cfg.num_inputs)
{
}

template <size_t DEPTH>
void ESN<DEPTH>::Warmup(std::span<const float> inputs)
{
    const size_t K = num_inputs_;
    if (inputs.size() % K != 0)
        throw std::invalid_argument(
            "ESN::Warmup: inputs.size() must be a multiple of cfg.num_inputs");
    const size_t num_steps = inputs.size() / K;
    for (size_t s = 0; s < num_steps; ++s)
        fractal_->UpdateState(inputs.subspan(s * K, K));
}

template <size_t DEPTH>
void ESN<DEPTH>::Run(std::span<const float> inputs)
{
    const size_t K = num_inputs_;
    if (inputs.size() % K != 0)
        throw std::invalid_argument(
            "ESN::Run: inputs.size() must be a multiple of cfg.num_inputs");
    const size_t num_steps = inputs.size() / K;
    states_.resize((num_collected_ + num_steps) * OUT);
    for (size_t s = 0; s < num_steps; ++s)
    {
        fractal_->UpdateState(inputs.subspan(s * K, K));
        memcpy(states_.data() + (num_collected_ + s) * OUT,
               fractal_->Leaves(), OUT * sizeof(float));
    }
    num_collected_ += num_steps;
}

template <size_t DEPTH>
void ESN<DEPTH>::ClearStates()
{
    states_.clear();
    states_.shrink_to_fit();
    num_collected_ = 0;
}

// Derive train_size from the targets span and config layout. Regression
// expects train_size * config.num_outputs floats; classification expects
// one float-as-int label per timestep.
static inline size_t derive_train_size(size_t span_size, const ReadoutConfig& cfg, const char* fn)
{
    const size_t per_step =
        (cfg.task == ReadoutTask::Classification)
            ? size_t{1}
            : static_cast<size_t>(cfg.num_outputs);
    if (per_step == 0 || span_size % per_step != 0)
        throw std::invalid_argument(
            std::string(fn) +
            ": targets.size() must be a multiple of config.num_outputs (regression) "
            "or equal to train_size (classification)");
    return span_size / per_step;
}

template <size_t DEPTH>
void ESN<DEPTH>::Train(std::span<const float> targets,
                       const ReadoutConfig& config)
{
    const size_t train_size = derive_train_size(targets.size(), config, "ESN::Train");
    auto sub = ReadoutStates(0, train_size);
    readout_.Train(sub.data(), targets.data(), train_size, EffectiveDIM(), config);
}

template <size_t DEPTH>
void ESN<DEPTH>::Train(std::span<const float> targets,
                       const ReadoutConfig& config,
                       CNNTrainHooks& hooks)
{
    const size_t train_size = derive_train_size(targets.size(), config, "ESN::Train");
    auto sub = ReadoutStates(0, train_size);
    readout_.Train(sub.data(), targets.data(), train_size, EffectiveDIM(), config, hooks);
}

template <size_t DEPTH>
void ESN<DEPTH>::InitOnline(const float* warmup_inputs, size_t warmup_count,
                          const ReadoutConfig& config)
{
    Run(std::span<const float>(warmup_inputs, warmup_count * num_inputs_));
    auto sub = ReadoutStates(0, warmup_count);
    readout_.InitOnline(sub.data(), warmup_count, EffectiveDIM(), config);
    ClearStates();
}

template <size_t DEPTH>
void ESN<DEPTH>::TrainLiveStep(float target_class, float lr, float weight_decay)
{
    readout_.TrainOnlineStep(fractal_->Leaves(), static_cast<int>(target_class), lr, weight_decay);
}

template <size_t DEPTH>
void ESN<DEPTH>::CopyLiveState(float* out) const
{
    std::memcpy(out, fractal_->Leaves(), OUT * sizeof(float));
}

template <size_t DEPTH>
void ESN<DEPTH>::TrainLiveBatch(const float* states, const int* targets,
                              size_t count, float lr, float weight_decay)
{
    readout_.TrainOnlineBatch(states, targets, count, lr, weight_decay);
}

template <size_t DEPTH>
void ESN<DEPTH>::TrainLiveStepRegression(const float* target, float lr,
                                       float weight_decay)
{
    readout_.TrainOnlineStepRegression(fractal_->Leaves(), target, lr, weight_decay);
}

template <size_t DEPTH>
void ESN<DEPTH>::TrainLiveBatchRegression(const float* states, const float* targets,
                                        size_t count, float lr, float weight_decay)
{
    readout_.TrainOnlineBatchRegression(states, targets, count, lr, weight_decay);
}

template <size_t DEPTH>
void ESN<DEPTH>::ComputeTargetCentering(const float* targets, size_t num_samples)
{
    readout_.ComputeTargetCentering(targets, num_samples);
}

template <size_t DEPTH>
float ESN<DEPTH>::PredictRaw(size_t timestep) const
{
    if (timestep >= num_collected_)
        throw std::out_of_range("ESN::PredictRaw: timestep out of range");
    return readout_.PredictRaw(ReadoutInput(timestep));
}

template <size_t DEPTH>
void ESN<DEPTH>::PredictRaw(size_t timestep, float* output) const
{
    if (timestep >= num_collected_)
        throw std::out_of_range("ESN::PredictRaw: timestep out of range");
    readout_.PredictRaw(ReadoutInput(timestep), output);
}

template <size_t DEPTH>
float ESN<DEPTH>::PredictLiveRaw() const
{
    return readout_.PredictRaw(fractal_->Leaves());
}

template <size_t DEPTH>
void ESN<DEPTH>::PredictLiveRaw(float* output) const
{
    readout_.PredictRaw(fractal_->Leaves(), output);
}

template <size_t DEPTH>
double ESN<DEPTH>::R2(std::span<const float> targets, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range("ESN::R2: start + count exceeds num_collected_");
    const size_t K = readout_.NumOutputs();
    if (targets.size() < (start + count) * K)
        throw std::out_of_range(
            "ESN::R2: targets.size() < (start + count) * num_outputs");
    auto sub = ReadoutStates(start, count);
    return readout_.R2(sub.data(), targets.data() + start * K, count);
}

template <size_t DEPTH>
double ESN<DEPTH>::NRMSE(std::span<const float> targets, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range("ESN::NRMSE: start + count exceeds num_collected_");
    if (count == 0) return 0.0;

    const size_t K = readout_.NumOutputs();
    if (targets.size() < (start + count) * K)
        throw std::out_of_range(
            "ESN::NRMSE: targets.size() < (start + count) * num_outputs");
    const float* tgt = targets.data() + start * K;

    std::vector<float> preds(count * K);
    for (size_t s = 0; s < count; ++s)
        readout_.PredictRaw(ReadoutInput(start + s), preds.data() + s * K);

    double nrmse_sum = 0.0;
    for (size_t k = 0; k < K; ++k) {
        double mean = 0.0;
        for (size_t s = 0; s < count; ++s)
            mean += tgt[s * K + k];
        mean /= static_cast<double>(count);

        double var = 0.0, mse_k = 0.0;
        for (size_t s = 0; s < count; ++s) {
            double y  = tgt[s * K + k];
            double yh = preds[s * K + k];
            var += (y - mean) * (y - mean);
            mse_k += (y - yh) * (y - yh);
        }
        if (var < 1e-12)
            nrmse_sum += std::numeric_limits<double>::infinity();
        else
            nrmse_sum += std::sqrt(mse_k / count) / std::sqrt(var / count);
    }
    return nrmse_sum / static_cast<double>(K);
}

template <size_t DEPTH>
double ESN<DEPTH>::Accuracy(std::span<const float> labels, size_t start, size_t count) const
{
    if (start + count > num_collected_)
        throw std::out_of_range("ESN::Accuracy: start + count exceeds num_collected_");
    if (labels.size() < start + count)
        throw std::out_of_range("ESN::Accuracy: labels.size() < start + count");
    auto sub = ReadoutStates(start, count);
    return readout_.Accuracy(sub.data(), labels.data() + start, count);
}

template <size_t DEPTH>
size_t ESN<DEPTH>::NumOutputs() const
{
    return readout_.NumOutputs();
}

template <size_t DEPTH>
FractalReservoirConfig<DEPTH> ESN<DEPTH>::GetConfig() const
{
    return fractal_cfg_;
}

template <size_t DEPTH>
typename ESN<DEPTH>::ReadoutState ESN<DEPTH>::GetReadoutState() const
{
    ReadoutState s;
    s.is_trained = readout_.NumFeatures() > 0;
    s.bias = static_cast<double>(readout_.Bias());
    s.feature_mean = readout_.FeatureMean();
    s.feature_scale = readout_.FeatureScale();
    const auto& w = readout_.Weights();
    s.weights.assign(w.begin(), w.end());
    s.target_mean = readout_.TargetMean();
    return s;
}

template <size_t DEPTH>
void ESN<DEPTH>::SetReadoutState(const ReadoutState& state)
{
    if (!state.is_trained) return;
    readout_.SetState(state.weights, state.bias,
                      state.feature_mean, state.feature_scale,
                      state.target_mean);
}

template <size_t DEPTH>
void ESN<DEPTH>::SetCNNConfig(const ReadoutConfig& cfg)
{
    readout_.SetConfig(cfg);
}

// ---------------------------------------------------------------
//  Readout feature accessors (leaf-only readout, no subsampling)
// ---------------------------------------------------------------

template <size_t DEPTH>
const float* ESN<DEPTH>::ReadoutInput(size_t timestep) const
{
    return states_.data() + timestep * OUT;
}

template <size_t DEPTH>
std::vector<float> ESN<DEPTH>::ReadoutStates(size_t start, size_t count) const
{
    std::vector<float> buf(count * OUT);
    std::memcpy(buf.data(), states_.data() + start * OUT, count * OUT * sizeof(float));
    return buf;
}

template <size_t DEPTH>
std::vector<float> ESN<DEPTH>::SelectedStates() const
{
    return ReadoutStates(0, num_collected_);
}

template class ESN<1>;
template class ESN<2>;
template class ESN<3>;
template class ESN<4>;
template class ESN<5>;
