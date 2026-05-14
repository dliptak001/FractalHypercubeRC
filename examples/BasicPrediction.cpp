/// @file BasicPrediction.cpp
/// @brief Sine-wave forecast on the fractal lattice. Migrated from
///        HypercubeRC/examples/BasicPrediction.cpp -- the canonical
///        end-to-end demo, but driving FractalReservoir<DEPTH> +
///        ESN<DEPTH> instead of the flat hypercube reservoir.
///
/// Predicts sin(0.1*(t+1)) from sin(0.1*t) using only the reservoir's
/// internal state -- the HCNN readout never sees the input directly.
/// Runs DEPTH=2 by default; uncomment the run_depth<1>() / run_depth<3>()
/// calls in main to see the leaf-count effect (8, 64, 512 features)
/// side-by-side.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

#include "../ESN.h"
#include "../FractalReservoir.h"

namespace {

constexpr size_t WARMUP   = 200;
constexpr size_t COLLECT  = 2000;
constexpr size_t HORIZON  = 1;
constexpr double TRAIN_FRACTION = 0.7;

// Mirrors `presets::Baseline<DIM>()` in HypercubeRC but inlined here -- the
// fractal repo doesn't ship Presets.h. Tuned for sine prediction.
ReadoutConfig make_cnn_cfg(size_t depth)
{
    ReadoutConfig c;
    c.num_outputs    = 1;
    c.task           = ReadoutTask::Regression;
    c.num_layers     = 1;
    c.conv_channels  = 8;
    c.epochs         = 1100;
    const size_t shift = std::min<size_t>(3 * depth - 1, 9); // cap batch at 512
    c.batch_size     = 1 << shift;
    c.lr_max         = 0.0015f;
    c.lr_min_frac    = 0.1f;
    c.seed           = 42007;
    return c;
}

template <size_t DEPTH>
void run_depth()
{
    constexpr size_t LEAVES = FractalReservoir<DEPTH>::leaf_count;
    constexpr size_t TOTAL  = WARMUP + COLLECT + HORIZON;

    std::vector<float> signal(TOTAL);
    for (size_t t = 0; t < TOTAL; ++t)
        signal[t] = std::sin(0.1f * static_cast<float>(t));

    std::vector<float> targets(COLLECT);
    for (size_t t = 0; t < COLLECT; ++t)
        targets[t] = signal[WARMUP + t + HORIZON];

    const size_t train_size = static_cast<size_t>(COLLECT * TRAIN_FRACTION);
    const size_t test_size  = COLLECT - train_size;

    // Surveyed per-DEPTH level dynamics, with a hand-picked seed override.
    // The surveyed defaults run a low leak_rate (~0.15) for slow, long-
    // memory integration; seed 2372 was the best sine-forecast pick from
    // a StateRank seed sweep.
    auto fcfg = FractalReservoirConfigDefaults::For<DEPTH>();
    fcfg.seed = 2372;   // 2372 -> NRMSE: 0.000492
    auto cnn_cfg = make_cnn_cfg(DEPTH);

    std::cout << "\n--- DEPTH " << DEPTH
              << "  (leaves=" << LEAVES
              << ", effective DIM=" << (3 * DEPTH) << ") ---\n";
    std::cout << "  Training: " << cnn_cfg.epochs
              << " epochs, batch=" << cnn_cfg.batch_size
              << ", lr=" << cnn_cfg.lr_max
              << " (cosine, floor="
              << (cnn_cfg.lr_max * cnn_cfg.lr_min_frac) << ")\n";

    ESN<DEPTH> esn(fcfg);

    esn.Warmup(std::span<const float>(signal.data(), WARMUP));
    esn.Run   (std::span<const float>(signal.data() + WARMUP, COLLECT));

    std::cout << "  Training..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(std::span<const float>(targets.data(), train_size), cnn_cfg);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << secs << "s)\n";

    auto tgt_span = std::span<const float>(targets.data(), train_size + test_size);
    double r2    = esn.R2   (tgt_span, train_size, test_size);
    double nrmse = esn.NRMSE(tgt_span, train_size, test_size);

    std::cout << "  R2:    " << std::fixed << std::setprecision(6) << r2;
    if      (r2 > 0.9999) std::cout << "  (effectively perfect)";
    else if (r2 > 0.99)   std::cout << "  (excellent)";
    else if (r2 > 0.9)    std::cout << "  (good)";
    std::cout << "\n";
    std::cout << "  NRMSE: " << std::setprecision(6) << nrmse;
    if      (nrmse < 0.001) std::cout << "  (sub-0.1% error)";
    else if (nrmse < 0.01)  std::cout << "  (under 1% error)";
    else if (nrmse < 0.1)   std::cout << "  (under 10% error)";
    std::cout << "\n";

    std::cout << "\n  Sample predictions (test set):\n";
    std::cout << "    Step  |   Actual   |  Predicted  |    Error\n";
    std::cout << "    ------+------------+-------------+-----------\n";
    for (size_t i = 0; i < 10; ++i)
    {
        float actual    = targets[train_size + i];
        float predicted = esn.PredictRaw(train_size + i);
        float error     = actual - predicted;
        std::cout << "    " << std::setw(5) << (train_size + i)
                  << " | " << std::showpos << std::setprecision(5)
                  << std::setw(10) << actual
                  << " | " << std::setw(11) << predicted
                  << " | " << std::setw(10) << error
                  << std::noshowpos << "\n";
    }
}

} // namespace

int main()
{
    std::cout << "=== FractalHypercubeRC: Sine Wave Prediction ===\n\n";
    std::cout << "Task: predict sin(0.1*(t+1)) from sin(0.1*t) via the fractal\n";
    std::cout << "reservoir's leaf states. The HCNN readout never sees the\n";
    std::cout << "input directly -- it discovers features on the 8^DEPTH leaf\n";
    std::cout << "cube of the deepest level.\n";

    //run_depth<1>();
    run_depth<2>();
    //run_depth<3>();

    std::cout << "\nThe HCNN readout learned the sine forecast from the fractal's\n";
    std::cout << "leaf dynamics, with conv kernels operating on the 3*DEPTH-cube.\n";
    return 0;
}
