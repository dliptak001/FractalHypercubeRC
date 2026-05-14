#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <random>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../ESN.h"

/// ========================================================================
///  NARMA-N Generator
/// ========================================================================
///
/// Target-alignment bug fix (see generate_prediction_task below): an
/// earlier version paired inputs[t] = u(t) with targets[t] = y(t+1),
/// which is not learnable and held NARMA NRMSE near 0.65-0.8 regardless
/// of reservoir configuration. NARMA is system identification, not
/// forecasting -- the corrected version pairs u(t) with y(t).

template <typename T = float>
class NARMA_N_Generator
{
public:
    NARMA_N_Generator(size_t N = 10,
                      uint64_t seed = 193,
                      T alpha = T(0.3),
                      T beta = T(0.05),
                      T gamma = T(1.5),
                      T delta = T(0.1),
                      T u_low = T(0.0),
                      T u_high = T(0.5),
                      bool use_tanh = false)
        : N_(N), alpha_(alpha), beta_(beta), gamma_(gamma), delta_(delta),
          u_low_(u_low), u_high_(u_high), use_tanh_(use_tanh),
          rng_(seed), u_dist_(u_low_, u_high_)
    {
        if (N_ < 2)
            throw std::invalid_argument("NARMA_N_Generator: N must be >= 2");
    }

    /// Generate a NARMA-N series for a prediction task.
    /// Returns {inputs_u, targets_y} aligned at the same index:
    /// targets_y[t] = y(t), the NARMA-N output for input u(t).
    ///
    /// BUG FIX: this previously returned targets_y[t] = y(t+1) (with an
    /// extra "+1 for target shift" sample). That made the target depend
    /// on u(t+1) via the gamma*u(t+1)*u(t+1-N) term -- an input the
    /// reservoir, driven only through u(t), has never seen. The product
    /// term was therefore unlearnable and NRMSE collapsed toward 1.0,
    /// the likely cause of this project's NARMA scores plateauing near
    /// 0.65 across all depths and configurations. NARMA is system
    /// identification, not forecasting: y(t) is produced from u(t) and
    /// u(t-N), so u(t) and y(t) are the correct pairing.
    std::pair<std::vector<T>, std::vector<T>>
    generate_prediction_task(size_t num_steps, size_t warmup_steps = 500)
    {
        if (num_steps == 0) return {{}, {}};

        const size_t total = num_steps + warmup_steps;

        std::vector<T> u_series(total);
        std::vector<T> y_series(total, T(0));

        std::deque<T> y_hist(N_, T(0));
        std::deque<T> u_hist(N_, u_dist_(rng_));

        T running_sum_y = T(0);

        for (size_t t = 0; t < total; ++t)
        {
            T u_t = u_dist_(rng_);
            u_series[t] = u_t;

            T y_prev = y_hist.back();
            T sum_y = running_sum_y;
            T u_delayed = u_hist.front();

            T y_t = alpha_ * y_prev
                  + beta_ * y_prev * sum_y
                  + gamma_ * u_delayed * u_t
                  + delta_;

            if (use_tanh_) y_t = std::tanh(y_t);

            y_series[t] = y_t;

            running_sum_y += y_t - y_hist.front();
            y_hist.pop_front();
            y_hist.push_back(y_t);

            u_hist.pop_front();
            u_hist.push_back(u_t);
        }

        // Return the post-warmup portion. inputs[t] and targets[t] are
        // index-aligned: targets[t] = y(t) is the NARMA output for u(t).
        std::vector<T> inputs(num_steps);
        std::vector<T> targets(num_steps);

        for (size_t t = 0; t < num_steps; ++t)
        {
            inputs[t]  = u_series[warmup_steps + t];
            targets[t] = y_series[warmup_steps + t];
        }

        return {inputs, targets};
    }

private:
    size_t N_;
    T alpha_, beta_, gamma_, delta_, u_low_, u_high_;
    bool use_tanh_;
    std::mt19937_64 rng_;
    std::uniform_real_distribution<T> u_dist_;
};

/// ========================================================================
///  NARMA-N Benchmark (depth-aware, production ready)
/// ========================================================================
///
/// Recommended `narma_order` per fractal DEPTH (match N to the reservoir's
/// effective memory horizon — too-low N at deeper DEPTH leaves memory
/// capacity unused, too-high N at shallow DEPTH exceeds what the reservoir
/// can integrate):
///
///   DEPTH | Recommended N
///   ------+--------------
///       1 |         10-12
///       2 |         10-12
///       3 |         12-15
///       4 |         15-20
///       5 |         20-25
///
/// NARMA-10 remains the canonical RC reference task; higher N stresses
/// longer memory horizons.

template <size_t DEPTH>
class NARMA_N_Benchmark
{
public:
    struct Result
    {
        double nrmse;
        double train_time_s;
    };

    NARMA_N_Benchmark(const FractalReservoirConfig<DEPTH>& fractal_config,
                      size_t narma_order,
                      ReadoutConfig readout_config = {},
                      size_t total_steps = 0,   // 0 = auto-select per DEPTH
                      size_t warmup_steps = 0,  // 0 = auto-select
                      bool use_tanh_stability = true)
        : fractal_config_(fractal_config),
          readout_config_(std::move(readout_config)),
          narma_order_(narma_order),
          total_steps_(total_steps),
          warmup_steps_(warmup_steps),
          use_tanh_stability_(use_tanh_stability)
    {
    }

    Result Run()
    {
        // === Depth-aware sizing ===
        if (total_steps_ == 0) total_steps_ = GetRecommendedTotalSteps(DEPTH);
        // Reservoir warmup: a leaky ESN forgets initial conditions in
        // O(1/leak_rate) steps. Default leaks across levels are 0.10–0.60
        // → state is fully settled in well under 100 steps. A fixed 1000-
        // step warmup is two orders of magnitude past the settling time
        // and depth-independent. The previous `2 * total_steps` default
        // (28k at DEPTH=5) was unjustifiable and dominated runtime.
        // Caller can still override via the ctor argument.
        if (warmup_steps_ == 0) warmup_steps_ = 1000;

        // === Generate NARMA-N sequence ===
        NARMA_N_Generator<float> gen(
            narma_order_,
            fractal_config_.seed + 99,
            0.3f, 0.05f, 1.5f,
            (narma_order_ >= 20 ? 0.01f : 0.1f),   // smaller delta for high N
            0.0f, 0.5f,
            use_tanh_stability_);

        // Generate `warmup_steps_ + total_steps_` samples with no
        // generator-internal warmup — the reservoir warmup phase
        // (esn.Warmup below) absorbs the early NARMA-N transient. The
        // generator's `warmup_steps` parameter would discard samples
        // before returning, leaving fewer than needed; we instead pass
        // 0 there and slice ourselves.
        const size_t total_samples = warmup_steps_ + total_steps_;
        auto [u, y] = gen.generate_prediction_task(total_samples,
                                                   /*warmup_steps=*/0);

        // Scale inputs to reservoir range [-1, +1]
        for (auto& val : u)
            val = val * 4.0f - 1.0f;

        // === Run ESN ===
        ESN<DEPTH> esn(fractal_config_);
        esn.Warmup(std::span<const float>(u.data(), warmup_steps_));
        esn.Run(std::span<const float>(u.data() + warmup_steps_, total_steps_));

        // === Train & Evaluate ===
        const size_t train_size = static_cast<size_t>(total_steps_ * 0.80);
        const size_t test_size = total_steps_ - train_size;

        // Targets aligned with the post-warmup Run window of u.
        const float* y_post = y.data() + warmup_steps_;

        auto t0 = std::chrono::steady_clock::now();
        esn.Train(std::span<const float>(y_post, train_size), readout_config_);
        auto t1 = std::chrono::steady_clock::now();

        double nrmse = esn.NRMSE(std::span<const float>(y_post, total_steps_),
                                 train_size, test_size);

        double time_s = std::chrono::duration<double>(t1 - t0).count();

        return {nrmse, time_s};
    }

private:
    FractalReservoirConfig<DEPTH> fractal_config_;
    ReadoutConfig readout_config_;
    size_t narma_order_;
    size_t total_steps_;
    size_t warmup_steps_;
    bool use_tanh_stability_;

    static size_t GetRecommendedTotalSteps(size_t depth)
    {
        switch (depth) {
            case 1: return 2500;
            case 2: return 4000;
            case 3: return 6000;
            case 4: return 9000;
            case 5: return 14000;
            default: return 16000;
        }
    }
};