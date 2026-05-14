/// @file StreamingAnomaly.cpp
/// @brief Anomaly detection: learn normal behavior, flag deviations.
///        Migrated from HypercubeRC/examples/StreamingAnomaly.cpp -- same
///        two-phase pipeline, but driving FractalReservoir<DEPTH> +
///        ESN<DEPTH> instead of the flat hypercube reservoir.
///
/// Phase 1 trains the HCNN readout to predict the next value of a normal
/// multi-harmonic process signal. Phase 2 streams 200-step windows past the
/// frozen readout and flags any window whose RMSE exceeds 10x the held-out
/// baseline. ClearStates() between windows resets the collected-state buffer
/// only; the reservoir's live activations persist, which is what makes the
/// slow washout after sustained anomalies visible.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "../ESN.h"
#include "../FractalReservoir.h"

namespace {

constexpr size_t DEPTH        = 3;          // 512 leaves -- closest to the
                                            //  original DIM=8 / 256 neurons.
constexpr size_t WARMUP       = 500;
constexpr size_t PRIME_STEPS  = 4000;
constexpr size_t WINDOW       = 200;
constexpr float  NORMAL_NOISE = 0.01f;
constexpr float  ANOMALY_THRESHOLD = 10.0f;

void GenerateProcess(float* out, size_t n, size_t t_start,
                     float noise_level, float dc_drift, float freq_mult,
                     std::mt19937_64& rng)
{
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    for (size_t t = 0; t < n; ++t)
    {
        float phase = 0.1f * freq_mult * static_cast<float>(t_start + t);
        float clean = 0.6f * std::sin(phase) + 0.2f * std::sin(3.0f * phase);
        out[t] = clean + dc_drift + noise_level * noise(rng);
    }
}

double ComputeRMSE(const float* pred, const float* targets, size_t n)
{
    double mse = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        double err = static_cast<double>(targets[i]) - pred[i];
        mse += err * err;
    }
    return std::sqrt(mse / n);
}

struct Event
{
    const char* label;
    float noise;
    float drift;
    float freq;
};

// Mirrors `presets::Baseline<DIM>()` in HypercubeRC but inlined -- the fractal
// repo has no Presets.h. Tuned for the streaming-anomaly demo.
ReadoutConfig make_cnn_cfg()
{
    ReadoutConfig c;
    c.num_outputs   = 1;
    c.task          = ReadoutTask::Regression;
    c.num_layers    = 1;
    c.conv_channels = 8;
    c.epochs        = 1000;
    const size_t shift = std::min<size_t>(3 * DEPTH - 1, 9); // cap batch at 512
    c.batch_size    = 1 << shift;
    c.lr_max        = 0.0015f;
    c.lr_min_frac   = 0.1f;
    c.seed          = 420607;
    return c;
}

} // namespace

int main()
{
    constexpr size_t LEAVES = FractalReservoir<DEPTH>::leaf_count;

    // Surveyed per-DEPTH level dynamics, with a hand-picked seed override.
    // The original (flat) example forced a low leak_rate across every level
    // to make sustained anomalies visible: slow leak compounds prediction
    // error across steps, so DC drift and frequency shifts produce large
    // baseline ratios instead of borderline hits. The surveyed DEPTH=3
    // defaults already run leak_rate 0.15-0.20, which sits in that regime;
    // the explicit broadcast override is left below, commented, for
    // experimentation.
    auto fcfg = FractalReservoirConfigDefaults::For<DEPTH>();
    fcfg.seed = 2897;

    //for (auto& lvl : fcfg.levels) lvl.leak_rate = 0.3f;

    std::mt19937_64 signal_rng(fcfg.seed + 777);

    const Event normal    = { "Normal     ",      0.01f, 0.0f,  1.0f };
    const Event spike     = { "Noise spike",      0.12f, 0.0f,  1.0f };
    const Event drift_evt = { "DC drift   ",      0.01f, 0.30f, 1.0f };
    const Event freq_evt  = { "Freq shift ",      0.01f, 0.0f,  1.3f };

    std::vector<Event> schedule;
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);
    for (size_t i = 0; i < 3; ++i) schedule.push_back(spike);
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);
    for (size_t i = 0; i < 3; ++i) schedule.push_back(drift_evt);
    for (size_t i = 0; i < 5; ++i) schedule.push_back(normal);
    for (size_t i = 0; i < 3; ++i) schedule.push_back(freq_evt);
    for (size_t i = 0; i < 6; ++i) schedule.push_back(normal);

    std::cout << "=== FractalHypercubeRC: Streaming Anomaly Detection ===\n\n";
    std::cout << "Scenario: an industrial process produces a multi-harmonic signal.\n";
    std::cout << "The reservoir learns the normal pattern, then monitors for deviations.\n";
    std::cout << "Three types of anomaly are injected, each for 3 windows, separated\n";
    std::cout << "by normal operation to show both detection and recovery.\n\n";

    std::cout << "Anomaly types:\n";
    std::cout << "  1. Noise spike   -- sensor noise jumps 12x (0.01 -> 0.12)\n";
    std::cout << "  2. DC drift      -- systematic +0.30 offset (e.g. sensor fouling)\n";
    std::cout << "  3. Freq shift    -- process speed changes to 1.3x (e.g. motor issue)\n\n";

    ESN<DEPTH> esn(fcfg);

    std::cout << "Config: DEPTH=" << DEPTH
              << "  leaves=" << LEAVES
              << "  leak=surveyed defaults"
              << "  Threshold=" << ANOMALY_THRESHOLD << "x baseline\n\n";

    std::cout << "--- Phase 1: Learn what \"normal\" looks like ---\n\n";

    size_t t_global = 0;
    std::vector<float> prime_signal(WARMUP + PRIME_STEPS + 1);
    GenerateProcess(prime_signal.data(), prime_signal.size(), t_global,
                    NORMAL_NOISE, 0.0f, 1.0f, signal_rng);

    esn.Warmup(std::span<const float>(prime_signal.data(), WARMUP));
    esn.Run   (std::span<const float>(prime_signal.data() + WARMUP, PRIME_STEPS));
    t_global += WARMUP + PRIME_STEPS;

    std::vector<float> prime_targets(PRIME_STEPS);
    for (size_t t = 0; t < PRIME_STEPS; ++t)
        prime_targets[t] = prime_signal[WARMUP + t + 1];

    const size_t train_n = static_cast<size_t>(PRIME_STEPS * 0.7);
    const size_t test_n  = PRIME_STEPS - train_n;

    auto cnn_cfg = make_cnn_cfg();

    std::cout << "Training on " << train_n << " samples ("
              << cnn_cfg.epochs << " epochs, batch=" << cnn_cfg.batch_size
              << ", lr_max=" << std::setprecision(4) << cnn_cfg.lr_max << ")..."
              << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(std::span<const float>(prime_targets.data(), train_n), cnn_cfg);
    auto t1 = std::chrono::steady_clock::now();
    double train_secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(2) << train_secs << "s)\n\n";

    std::vector<float> prime_pred(test_n);
    for (size_t i = 0; i < test_n; ++i)
        prime_pred[i] = esn.PredictRaw(train_n + i);
    double baseline = ComputeRMSE(prime_pred.data(),
                                  prime_targets.data() + train_n, test_n);
    double threshold = baseline * ANOMALY_THRESHOLD;

    std::cout << "Baseline (prime test, RMSE): " << std::setprecision(6) << baseline
              << "   threshold " << threshold << "\n\n";

    std::cout << "--- Phase 2: Monitor the process (" << schedule.size()
              << " windows of " << WINDOW << " steps) ---\n\n";
    std::cout << "Each window is fed to the reservoir, and the readout predicts\n";
    std::cout << "the next value.  An RMSE above " << std::setprecision(0) << ANOMALY_THRESHOLD
              << "x baseline is flagged.\n\n";

    std::cout << "  Window | Condition          |    RMSE     Ratio | Status\n";
    std::cout << "  -------+--------------------+-------------------+---------\n";

    size_t flags = 0;

    for (size_t w = 0; w < schedule.size(); ++w)
    {
        const Event& evt = schedule[w];

        std::vector<float> sig(WINDOW + 1);
        GenerateProcess(sig.data(), sig.size(), t_global,
                        evt.noise, evt.drift, evt.freq, signal_rng);
        t_global += WINDOW;

        std::vector<float> tgt(WINDOW);
        for (size_t t = 0; t < WINDOW; ++t)
            tgt[t] = sig[t + 1];

        // ClearStates() only resets the collected leaf-state buffer
        // (num_collected_ = 0); the reservoir's live `vtx_output_` is
        // preserved, which is what gives sustained anomalies their slow
        // washout signature.
        esn.ClearStates();
        esn.Run(std::span<const float>(sig.data(), WINDOW));

        std::vector<float> pred(WINDOW);
        for (size_t t = 0; t < WINDOW; ++t)
            pred[t] = esn.PredictRaw(t);
        double rmse  = ComputeRMSE(pred.data(), tgt.data(), WINDOW);
        double ratio = rmse / baseline;
        bool anom    = (rmse > threshold);
        if (anom) ++flags;

        char status[16] = "";
        if (anom) std::snprintf(status, sizeof(status), "** ANOMALY **");

        std::cout << "  " << std::setw(5) << (w + 1)
                  << "  | " << evt.label << "        "
                  << " | " << std::fixed << std::setprecision(6) << std::setw(10) << rmse
                  << "  " << std::setprecision(1) << std::setw(5) << ratio
                  << " | " << status << "\n";
    }

    std::cout << "\nFlagged windows: " << flags
              << "  (expected ~11 = 9 anomaly windows + 2 washout windows\n"
              << "                     where the leaky integrator is still ringing)\n\n";

    std::cout << "--- What happened ---\n\n";
    std::cout << "The reservoir learned to predict normal process output during priming.\n";
    std::cout << "During monitoring, prediction error is the anomaly signal:\n\n";
    std::cout << "  Noise spike:  RMSE jumps ~12x -- random disturbance is unpredictable.\n";
    std::cout << "                Recovery is instant (next normal window back to baseline).\n\n";
    std::cout << "  DC drift:     RMSE rises dramatically -- the model didn't learn this offset.\n";
    std::cout << "                The leaky integrator compounds the error across steps.\n";
    std::cout << "                Takes 1-2 windows to wash out after recovery.\n\n";
    std::cout << "  Freq shift:   RMSE spikes -- changed dynamics break the learned pattern.\n";
    std::cout << "                Slowest recovery: reservoir needs 1-2 extra windows to\n";
    std::cout << "                wash out the altered frequency from its internal state.\n";

    return 0;
}
