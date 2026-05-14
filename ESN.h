#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <span>
#include <vector>
#include "FractalReservoir.h"
#include "Readout.h"

/// @brief Echo-state network: FractalReservoir -> Readout.
///
/// Wraps a single `FractalReservoir<DEPTH>` (top cube fixed at N=8) and a
/// learned HCNN readout. The reservoir accepts `cfg.num_inputs` channels
/// per timestep (in {1, 2, 4, 8}), striped across the depth-0 cube.
/// `Warmup` and `Run` take a flat `std::span<const float>` of shape
/// `num_steps * num_inputs` (timestep-major, row-major); `num_steps` is
/// derived from `inputs.size() / num_inputs` and the divide must be exact.
///
/// The readout consumes the fractal's deepest-level (leaf) states, packed
/// as a flat `8^DEPTH = 2^(3*DEPTH)` hypercube; outer and intermediate
/// levels are not exposed to the readout.
///
/// Lifecycle: `Warmup` drives the reservoir to wash out the initial
/// transient and discards the states; `Run` drives it and *appends* each
/// step's leaf states to an internal buffer; `Train` fits the readout on
/// that buffer; `Predict*` queries it. `ClearStates` empties the buffer
/// without disturbing the reservoir's live activations.
template <size_t DEPTH>
class ESN
{
    static constexpr size_t N = 8;
    static constexpr size_t OUT = FractalReservoir<DEPTH>::leaf_count;

public:
    explicit ESN(const FractalReservoirConfig<DEPTH>& fractal_cfg);

    ESN(const ESN&)                = delete;
    ESN& operator=(const ESN&)     = delete;
    ESN(ESN&&) noexcept            = default;
    ESN& operator=(ESN&&) noexcept = default;

    // ---------------------------------------------------------------
    //  Reservoir driving
    // ---------------------------------------------------------------

    /// @param inputs  num_steps * num_inputs floats (timestep-major).
    ///                Throws std::invalid_argument if size is not a
    ///                multiple of num_inputs.
    void Warmup(std::span<const float> inputs);

    /// @param inputs  Same layout as Warmup.
    void Run(std::span<const float> inputs);

    /// @brief Empties the collected leaf-state buffer; the reservoir's
    ///        live activations are left running.
    void ClearStates();

    // ---------------------------------------------------------------
    //  Training
    // ---------------------------------------------------------------

    /// @param targets  Regression: train_size * config.num_outputs floats
    ///                 (row-major). Classification: train_size float labels
    ///                 (cast to int internally). train_size is derived from
    ///                 the span size; throws std::invalid_argument on a
    ///                 non-exact divide.
    void Train(std::span<const float> targets,
               const ReadoutConfig& config);

    void Train(std::span<const float> targets,
               const ReadoutConfig& config,
               CNNTrainHooks& hooks);

    /// @param warmup_count  number of timesteps; warmup_inputs holds
    ///                      warmup_count * num_inputs floats.
    void InitOnline(const float* warmup_inputs, size_t warmup_count,
                    const ReadoutConfig& config);

    void TrainLiveStep(float target_class, float lr, float weight_decay = 0.0f);

    void CopyLiveState(float* out) const;

    void TrainLiveBatch(const float* states, const int* targets,
                        size_t count, float lr, float weight_decay = 0.0f);

    void TrainLiveStepRegression(const float* target, float lr,
                                 float weight_decay = 0.0f);

    void TrainLiveBatchRegression(const float* states, const float* targets,
                                  size_t count, float lr, float weight_decay = 0.0f);

    void ComputeTargetCentering(const float* targets, size_t num_samples);

    // ---------------------------------------------------------------
    //  Prediction & evaluation
    // ---------------------------------------------------------------

    /// @brief Readout output for a collected timestep. The scalar overload
    ///        requires num_outputs == 1; the pointer overload writes
    ///        num_outputs floats. Throws std::out_of_range on a bad timestep.
    [[nodiscard]] float PredictRaw(size_t timestep) const;
    void PredictRaw(size_t timestep, float* output) const;

    /// @brief Like PredictRaw, but on the reservoir's current live leaf
    ///        states rather than a collected timestep.
    [[nodiscard]] float PredictLiveRaw() const;
    void PredictLiveRaw(float* output) const;

    /// @brief R-squared on collected timesteps [start, start+count).
    ///        Regression metric only.
    /// @param targets  Must span at least (start+count) * num_outputs floats
    ///                 (row-major). Indexed from targets[start*num_outputs].
    ///                 Throws std::out_of_range if undersized.
    [[nodiscard]] double R2(std::span<const float> targets, size_t start, size_t count) const;

    /// @param targets  Same layout contract as R2.
    [[nodiscard]] double NRMSE(std::span<const float> targets, size_t start, size_t count) const;

    /// @brief Classification accuracy on collected timesteps [start, start+count).
    ///        Multi-class (num_outputs > 1) compares argmax to labels[s];
    ///        binary (num_outputs == 1) compares sign of prediction to sign of labels[s].
    /// @param labels  Must span at least (start+count) floats. Indexed from labels[start].
    ///                Throws std::out_of_range if undersized.
    [[nodiscard]] double Accuracy(std::span<const float> labels, size_t start, size_t count) const;

    [[nodiscard]] size_t NumOutputs() const;

    // ---------------------------------------------------------------
    //  State access
    // ---------------------------------------------------------------

    /// @brief All collected leaf states, flat: num_collected * OUT floats
    ///        (row-major, OUT = 8^DEPTH leaf states per timestep).
    [[nodiscard]] std::vector<float> SelectedStates() const;

    // ---------------------------------------------------------------
    //  Accessors
    // ---------------------------------------------------------------
    [[nodiscard]] size_t NumCollected() const { return num_collected_; }
    [[nodiscard]] size_t NumOutputVerts() const { return OUT; }
    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }

    // --- Config & persistence ---

    [[nodiscard]] FractalReservoirConfig<DEPTH> GetConfig() const;

    struct ReadoutState {
        std::vector<double> weights;
        double bias = 0.0;
        std::vector<float> feature_mean;
        std::vector<float> feature_scale;
        std::vector<double> target_mean;
        bool is_trained = false;
    };

    [[nodiscard]] ReadoutState GetReadoutState() const;
    void SetReadoutState(const ReadoutState& state);
    void SetCNNConfig(const ReadoutConfig& cfg);

private:
    std::unique_ptr<FractalReservoir<DEPTH>> fractal_;
    Readout readout_;

    FractalReservoirConfig<DEPTH> fractal_cfg_;
    size_t num_inputs_ = 1;

    std::vector<float> states_;
    size_t num_collected_ = 0;

    [[nodiscard]] static constexpr size_t EffectiveDIM() { return 3 * DEPTH; }
    const float* ReadoutInput(size_t timestep) const;
    [[nodiscard]] std::vector<float> ReadoutStates(size_t start, size_t count) const;
};
