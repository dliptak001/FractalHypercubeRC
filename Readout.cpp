#include "Readout.h"
#include "HCNN.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <numbers>
#include <numeric>
#include <stdexcept>


Readout::Readout() = default;
Readout::~Readout() = default;
Readout::Readout(Readout&&) noexcept = default;
Readout& Readout::operator=(Readout&&) noexcept = default;

// ---------------------------------------------------------------------------
//  Input standardization
// ---------------------------------------------------------------------------

void Readout::compute_standardization(const float* states,
                                         size_t num_samples, size_t n)
{
    input_mean_.resize(n);
    input_scale_.resize(n);

    // Per-vertex mean (double accumulation for precision at large N).
    std::vector<double> mean_acc(n, 0.0);
    for (size_t s = 0; s < num_samples; ++s) {
        const float* x = states + s * n;
        for (size_t v = 0; v < n; ++v)
            mean_acc[v] += x[v];
    }
    const double inv_n = 1.0 / static_cast<double>(num_samples);
    for (size_t v = 0; v < n; ++v)
        input_mean_[v] = static_cast<float>(mean_acc[v] * inv_n);

    // Per-vertex variance (double accumulation).
    std::vector<double> var_acc(n, 0.0);
    for (size_t s = 0; s < num_samples; ++s) {
        const float* x = states + s * n;
        for (size_t v = 0; v < n; ++v) {
            double d = x[v] - input_mean_[v];
            var_acc[v] += d * d;
        }
    }
    for (size_t v = 0; v < n; ++v) {
        float std_v = static_cast<float>(std::sqrt(var_acc[v] * inv_n));
        input_scale_[v] = (std_v > 1e-8f) ? (1.0f / std_v) : 1.0f;
    }
}

void Readout::standardize(const float* in, float* out, size_t n) const
{
    for (size_t v = 0; v < n; ++v)
        out[v] = (in[v] - input_mean_[v]) * input_scale_[v];
}

// ---------------------------------------------------------------------------
//  Architecture
// ---------------------------------------------------------------------------

void Readout::build_architecture()
{
    if (dim_ < 3)
        throw std::invalid_argument(
            "Readout::build_architecture: dim_ must be greater than or equal to 3");

    const size_t n = 1ULL << dim_;
    const int d = static_cast<int>(dim_);

    // Auto-size layers: min(DIM - 2, 2), at least 1.
    int layers = (config_.num_layers > 0)
                     ? config_.num_layers
                     : std::min(d - 2, 2);
    layers = std::max(layers, 1);
    assert(layers <= d - 2);

    auto task_type = (config_.task == ReadoutTask::Classification)
                         ? hcnn::TaskType::Classification
                         : hcnn::TaskType::Regression;
    net_ = std::make_unique<hcnn::HCNN>(
        d, config_.num_outputs, /*input_channels=*/1,
        task_type);

    int ch = config_.conv_channels;
    for (int i = 0; i < layers; ++i) {
        net_->AddConv(ch, hcnn::Activation::TANH, /*use_bias=*/true);
        net_->AddPool(hcnn::PoolType::MAX);
        ch *= 2;
    }

    net_->RandomizeWeights(0.0f, config_.seed);

    scratch_state_.resize(n);
    scratch_embedded_.resize(n);
    scratch_pred_.resize(num_outputs_);
}

// ---------------------------------------------------------------------------
//  Training
// ---------------------------------------------------------------------------

void Readout::Train(const float* states, const float* targets,
                       size_t num_samples, size_t dim,
                       const ReadoutConfig& config)
{
    CNNTrainHooks no_hooks;
    Train(states, targets, num_samples, dim, config, no_hooks);
}

void Readout::Train(const float* states, const float* targets,
                       size_t num_samples, size_t dim,
                       const ReadoutConfig& config,
                       CNNTrainHooks& hooks)
{
    config_ = config;
    dim_ = dim;
    const size_t n = 1ULL << dim;
    num_features_ = n;
    num_outputs_ = static_cast<size_t>(config.num_outputs);
    const bool is_classification = (config.task == ReadoutTask::Classification);

    // --- Input standardization ---
    compute_standardization(states, num_samples, n);

    // --- Standardize inputs ---
    std::vector<float> std_states(num_samples * n);
    for (size_t s = 0; s < num_samples; ++s)
        standardize(states + s * n, std_states.data() + s * n, n);

    // --- Build HCNN ---
    build_architecture();
    net_->SetOptimizer(hcnn::OptimizerType::ADAM);

    // Allow mid-training Predict*/Accuracy/R2 from the hook callback.
    // Weights are fresh but the network is usable; weights_blob_ stays empty
    // until flatten_weights() at the end (only matters for serialization).
    trained_ = true;

    auto fire_hook = [&](int epoch_done_1based, float lr) {
        if (!hooks.epoch_callback || hooks.eval_every_epochs <= 0) return;
        const bool at_interval = (epoch_done_1based % hooks.eval_every_epochs) == 0;
        const bool at_final    = epoch_done_1based == config.epochs;
        if (at_interval || at_final) {
            flatten_weights();
            hooks.epoch_callback(epoch_done_1based, config.epochs, lr);
        }
    };

    // --- Cosine LR schedule (loop-invariant pieces) ---
    const float lr_min = config.lr_max * config.lr_min_frac;
    const auto pi = static_cast<float>(std::numbers::pi);
    const int horizon = (config.lr_decay_epochs > 0)
                            ? config.lr_decay_epochs
                            : config.epochs;

    auto cosine_lr = [&](int epoch) -> float {
        float progress = static_cast<float>(epoch) / static_cast<float>(horizon);
        if (progress > 1.0f) progress = 1.0f;
        return lr_min + 0.5f * (config.lr_max - lr_min) *
               (1.0f + std::cos(pi * progress));
    };

    // --- Prepare task-specific targets ---
    std::vector<int> int_targets;
    std::vector<float> centered_targets;

    if (is_classification) {
        int_targets.resize(num_samples);
        for (size_t s = 0; s < num_samples; ++s)
            int_targets[s] = static_cast<int>(targets[s]);
        target_mean_.clear();
    } else {
        target_mean_.assign(num_outputs_, 0.0);
        for (size_t s = 0; s < num_samples; ++s)
            for (size_t k = 0; k < num_outputs_; ++k)
                target_mean_[k] += targets[s * num_outputs_ + k];
        for (size_t k = 0; k < num_outputs_; ++k)
            target_mean_[k] /= static_cast<double>(num_samples);

        centered_targets.resize(num_samples * num_outputs_);
        for (size_t s = 0; s < num_samples; ++s)
            for (size_t k = 0; k < num_outputs_; ++k)
                centered_targets[s * num_outputs_ + k] =
                    targets[s * num_outputs_ + k] - static_cast<float>(target_mean_[k]);
    }

    // --- Verbose reporting buffers ---
    std::vector<float> verbose_logits;
    std::vector<float> verbose_preds;
    if (config.verbose && config.verbose_train_acc) {
        if (is_classification)
            verbose_logits.resize(num_samples * num_outputs_);
        else
            verbose_preds.resize(num_samples * num_outputs_);
    }

    // --- Epoch loop (shared for both tasks) ---
    for (int e = 0; e < config.epochs; ++e) {
        float lr = cosine_lr(e);

        if (is_classification) {
            net_->TrainEpoch(
                std_states.data(), static_cast<int>(n),
                int_targets.data(),
                static_cast<int>(num_samples), config.batch_size,
                lr, /*momentum=*/0.0f, config.weight_decay,
                /*class_weights=*/nullptr,
                /*shuffle_seed=*/static_cast<unsigned>(e + 1));
        } else {
            net_->TrainEpochRegression(
                std_states.data(), static_cast<int>(n),
                centered_targets.data(),
                static_cast<int>(num_samples), config.batch_size,
                lr, /*momentum=*/0.0f, config.weight_decay,
                /*shuffle_seed=*/static_cast<unsigned>(e + 1));
        }

        if (config.verbose) {
            if (config.verbose_train_acc) {
                if (is_classification) {
                    net_->ForwardBatch(std_states.data(), static_cast<int>(n),
                                       static_cast<int>(num_samples),
                                       verbose_logits.data());
                    size_t correct = 0;
                    for (size_t s = 0; s < num_samples; ++s) {
                        const float* row = verbose_logits.data() + s * num_outputs_;
                        size_t pred = 0;
                        float best = row[0];
                        for (size_t k = 1; k < num_outputs_; ++k)
                            if (row[k] > best) { best = row[k]; pred = k; }
                        if (static_cast<int>(pred) == int_targets[s]) ++correct;
                    }
                    double acc = 100.0 * correct / num_samples;
                    std::printf("  epoch %3d/%d  lr=%.5f  train_acc=%.2f%%\n",
                                e + 1, config.epochs, lr, acc);
                } else {
                    net_->ForwardBatch(std_states.data(), static_cast<int>(n),
                                       static_cast<int>(num_samples),
                                       verbose_preds.data());
                    double mse = 0.0;
                    for (size_t i = 0; i < num_samples * num_outputs_; ++i) {
                        double d = verbose_preds[i] - centered_targets[i];
                        mse += d * d;
                    }
                    mse /= static_cast<double>(num_samples * num_outputs_);
                    std::printf("  epoch %3d/%d  lr=%.5f  train_mse=%.6f\n",
                                e + 1, config.epochs, lr, mse);
                }
            } else {
                std::printf("  epoch %3d/%d  lr=%.5f\n",
                            e + 1, config.epochs, lr);
            }
            std::fflush(stdout);
        }

        fire_hook(e + 1, lr);
        if (hooks.stop_requested) break;
    }

    // --- Flatten weights for serialization ---
    flatten_weights();
}

// ---------------------------------------------------------------------------
//  Online (streaming) training
// ---------------------------------------------------------------------------

void Readout::InitOnline(const float* warmup_states, size_t warmup_count,
                            size_t dim, const ReadoutConfig& config)
{
    config_ = config;
    dim_ = dim;
    const size_t n = 1ULL << dim;
    num_features_ = n;
    num_outputs_ = static_cast<size_t>(config.num_outputs);

    compute_standardization(warmup_states, warmup_count, n);
    build_architecture();
    net_->SetOptimizer(hcnn::OptimizerType::ADAM);
    net_->PrepareBuffers();
    target_mean_.clear();
    trained_ = true;
}

void Readout::TrainOnlineStep(const float* state, int target_class,
                                 float lr, float weight_decay)
{
    if (!trained_ || !net_)
        throw std::logic_error("Readout::TrainOnlineStep: must call Train or InitOnline first");
    const size_t n = num_features_;

    standardize(state, scratch_state_.data(), n);
    net_->TrainStep(scratch_state_.data(), static_cast<int>(n), target_class,
                    lr, /*momentum=*/0.0f, weight_decay);
}

void Readout::TrainOnlineBatch(const float* states, const int* targets,
                                  size_t count, float lr, float weight_decay)
{
    if (!trained_ || !net_)
        throw std::logic_error("Readout::TrainOnlineBatch: must call Train or InitOnline first");
    const size_t n = num_features_;
    const size_t total = count * n;

    if (scratch_batch_.size() < total)
        scratch_batch_.resize(total);

    for (size_t i = 0; i < count; ++i)
        standardize(states + i * n, scratch_batch_.data() + i * n, n);

    net_->TrainBatch(scratch_batch_.data(), static_cast<int>(n),
                     targets, static_cast<int>(count),
                     lr, /*momentum=*/0.0f, weight_decay);
}

void Readout::TrainOnlineStepRegression(const float* state, const float* target,
                                           float lr, float weight_decay)
{
    if (!trained_ || !net_)
        throw std::logic_error("Readout::TrainOnlineStepRegression: must call Train or InitOnline first");
    const size_t n = num_features_;
    const size_t K = num_outputs_;

    standardize(state, scratch_state_.data(), n);

    const float* tgt = target;
    if (!target_mean_.empty()) {
        if (scratch_target_.size() < K)
            scratch_target_.resize(K);
        for (size_t k = 0; k < K; ++k)
            scratch_target_[k] = target[k] - static_cast<float>(target_mean_[k]);
        tgt = scratch_target_.data();
    }

    net_->TrainStepRegression(scratch_state_.data(), static_cast<int>(n), tgt,
                              lr, /*momentum=*/0.0f, weight_decay);
}

void Readout::TrainOnlineBatchRegression(const float* states, const float* targets,
                                            size_t count, float lr, float weight_decay)
{
    if (!trained_ || !net_)
        throw std::logic_error("Readout::TrainOnlineBatchRegression: must call Train or InitOnline first");
    const size_t n = num_features_;
    const size_t K = num_outputs_;
    const size_t total = count * n;

    if (scratch_batch_.size() < total)
        scratch_batch_.resize(total);

    for (size_t i = 0; i < count; ++i)
        standardize(states + i * n, scratch_batch_.data() + i * n, n);

    const float* tgt = targets;
    if (!target_mean_.empty()) {
        const size_t tgt_total = count * K;
        if (scratch_centered_batch_.size() < tgt_total)
            scratch_centered_batch_.resize(tgt_total);
        for (size_t i = 0; i < count; ++i)
            for (size_t k = 0; k < K; ++k)
                scratch_centered_batch_[i * K + k] = targets[i * K + k] - static_cast<float>(target_mean_[k]);
        tgt = scratch_centered_batch_.data();
    }

    net_->TrainBatchRegression(scratch_batch_.data(), static_cast<int>(n),
                               tgt, static_cast<int>(count),
                               lr, /*momentum=*/0.0f, weight_decay);
}

void Readout::ComputeTargetCentering(const float* targets, size_t num_samples)
{
    const size_t K = num_outputs_;
    target_mean_.assign(K, 0.0);
    for (size_t s = 0; s < num_samples; ++s)
        for (size_t k = 0; k < K; ++k)
            target_mean_[k] += targets[s * K + k];
    for (size_t k = 0; k < K; ++k)
        target_mean_[k] /= static_cast<double>(num_samples);
}

// ---------------------------------------------------------------------------
//  Prediction
// ---------------------------------------------------------------------------

void Readout::PredictRaw(const float* state, float* output) const
{
    if (!trained_ || !net_)
        throw std::logic_error("Readout::PredictRaw: must call Train or InitOnline first");
    const size_t n = num_features_;

    standardize(state, scratch_state_.data(), n);
    net_->Embed(scratch_state_.data(), static_cast<int>(n),
                scratch_embedded_.data());
    net_->Forward(scratch_embedded_.data(), scratch_pred_.data());

    const bool is_regression = (config_.task == ReadoutTask::Regression);
    for (size_t k = 0; k < num_outputs_; ++k) {
        output[k] = scratch_pred_[k];
        if (is_regression && !target_mean_.empty())
            output[k] += static_cast<float>(target_mean_[k]);
    }
}

float Readout::PredictRaw(const float* state) const
{
    if (num_outputs_ != 1)
        throw std::logic_error("Readout::PredictRaw: scalar overload requires num_outputs == 1");
    float out;
    PredictRaw(state, &out);
    return out;
}

int Readout::PredictClass(const float* state) const
{
    if (!trained_ || !net_)
        throw std::logic_error("Readout::PredictClass: must call Train or InitOnline first");
    const size_t n = num_features_;

    standardize(state, scratch_state_.data(), n);
    net_->Embed(scratch_state_.data(), static_cast<int>(n),
                scratch_embedded_.data());
    net_->Forward(scratch_embedded_.data(), scratch_pred_.data());

    return static_cast<int>(
        std::max_element(scratch_pred_.begin(),
                         scratch_pred_.begin() + num_outputs_) -
        scratch_pred_.begin());
}

// ---------------------------------------------------------------------------
//  Evaluation
// ---------------------------------------------------------------------------

double Readout::R2(const float* states, const float* targets,
                      size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    const size_t n = num_features_;
    const size_t K = num_outputs_;

    // Predict all samples once, cache results.
    std::vector<float> preds(num_samples * K);
    for (size_t s = 0; s < num_samples; ++s)
        PredictRaw(states + s * n, preds.data() + s * K);

    // Average R2 across outputs.
    double r2_sum = 0.0;
    for (size_t k = 0; k < K; ++k) {
        double tgt_mean = 0.0;
        for (size_t s = 0; s < num_samples; ++s)
            tgt_mean += targets[s * K + k];
        tgt_mean /= static_cast<double>(num_samples);

        double ss_res = 0.0, ss_tot = 0.0;
        for (size_t s = 0; s < num_samples; ++s) {
            double y  = targets[s * K + k];
            double yh = preds[s * K + k];
            ss_res += (y - yh) * (y - yh);
            ss_tot += (y - tgt_mean) * (y - tgt_mean);
        }
        r2_sum += (ss_tot < 1e-12) ? 0.0 : (1.0 - ss_res / ss_tot);
    }
    return r2_sum / static_cast<double>(K);
}

double Readout::Accuracy(const float* states, const float* labels,
                            size_t num_samples) const
{
    if (num_samples == 0) return 0.0;
    const size_t n = num_features_;
    size_t correct = 0;

    if (num_outputs_ > 1) {
        // Multi-class: argmax vs label.
        for (size_t s = 0; s < num_samples; ++s) {
            int pred = PredictClass(states + s * n);
            if (pred == static_cast<int>(labels[s])) ++correct;
        }
    } else {
        // Binary: threshold at 0.
        for (size_t s = 0; s < num_samples; ++s) {
            float pred_val;
            PredictRaw(states + s * n, &pred_val);
            if ((pred_val > 0.0f) == (labels[s] > 0.0f)) ++correct;
        }
    }
    return static_cast<double>(correct) / static_cast<double>(num_samples);
}

// ---------------------------------------------------------------------------
//  Serialization
// ---------------------------------------------------------------------------

const std::vector<double>& Readout::Weights() const
{
    if (weights_blob_.empty() && net_) {
        auto fw = net_->GetWeights();
        weights_blob_.assign(fw.begin(), fw.end());
    }
    return weights_blob_;
}

void Readout::flatten_weights()
{
    if (!net_) { weights_blob_.clear(); return; }
    auto fw = net_->GetWeights();
    weights_blob_.assign(fw.begin(), fw.end());
}

void Readout::rebuild_from_blob()
{
    if (weights_blob_.empty() || dim_ == 0) return;

    // Reconstruct the network from stored config if needed.
    if (!net_) {
        build_architecture();
    }

    std::vector<float> fw(weights_blob_.begin(), weights_blob_.end());
    net_->SetWeights(fw);
}

void Readout::SetState(std::vector<double> weights, double bias,
                          std::vector<float> feature_mean,
                          std::vector<float> feature_scale,
                          std::vector<double> target_mean)
{
    weights_blob_ = std::move(weights);
    input_mean_ = std::move(feature_mean);
    input_scale_ = std::move(feature_scale);
    num_features_ = input_mean_.size();

    // Infer dim from num_features (N = 2^dim).
    if (num_features_ > 0) {
        size_t n = num_features_;
        size_t d = 0;
        while ((1ULL << d) < n) ++d;
        if ((1ULL << d) == n)
            dim_ = d;
    }

    // Restore target centering from the explicit vector if provided;
    // fall back to scalar bias for backward compat with old checkpoints.
    if (!target_mean.empty())
        target_mean_ = std::move(target_mean);
    else
        target_mean_.assign(num_outputs_, bias);

    if (!weights_blob_.empty()) {
        rebuild_from_blob();
        trained_ = true;
    }
}

void Readout::SetConfig(const ReadoutConfig& cfg)
{
    config_ = cfg;
    num_outputs_ = static_cast<size_t>(cfg.num_outputs);
}
