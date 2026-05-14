/// @file SignalClassification.cpp
/// @brief Multi-class waveform classification on the fractal lattice.
///        Migrated from HypercubeRC/examples/SignalClassification.cpp --
///        same task and analysis, but driving FractalReservoir<DEPTH> +
///        ESN<DEPTH> (HCNN softmax readout on 8^DEPTH leaf features)
///        instead of the flat hypercube reservoir.
///
/// Four waveforms (sine, square, triangle, chirp) cycle in 40-step blocks.
/// The HCNN readout sees only the fractal's leaf states and classifies
/// which waveform is currently driving the reservoir.
///
/// The fractal reservoir replaces the flat one wholesale, but the pipeline
/// (block-cycled signal -> reservoir state -> 4-class softmax) is identical
/// to the HypercubeRC original.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "../ESN.h"
#include "../FractalReservoir.h"

namespace {

constexpr float  PI          = 3.14159265358979323846f;
constexpr size_t NUM_CLASSES = 4;
constexpr size_t DEPTH       = 2;          // 64 leaves -- closest to the
                                           //  original DIM=5 / 32 neurons.
constexpr size_t WARMUP      = 300;
constexpr size_t BLOCK_SIZE  = 40;
constexpr size_t NUM_CYCLES  = 75;
constexpr size_t COLLECT     = BLOCK_SIZE * NUM_CLASSES * NUM_CYCLES;
constexpr double TRAIN_FRACTION = 0.7;
constexpr float  NOISE_LEVEL = 0.15f;
constexpr float  CLASS_FREQ[NUM_CLASSES] = { 0.11f, 0.13f, 0.12f, 0.10f };
constexpr const char* CLASS_NAMES[NUM_CLASSES] =
    {"Sine    ", "Square  ", "Triangle", "Chirp   "};

float GenerateWaveform(size_t waveform, float phase)
{
    switch (waveform)
    {
    case 0: return std::sin(phase);
    case 1: return std::sin(phase) >= 0.0f ? 0.9f : -0.9f;
    case 2:
    {
        float p = std::fmod(phase, 2.0f * PI);
        if (p < 0) p += 2.0f * PI;
        return (p < PI) ? (-1.0f + 2.0f * p / PI) : (3.0f - 2.0f * p / PI);
    }
    case 3: return std::sin(phase + 0.3f * phase * phase);
    default: return 0.0f;
    }
}

// Mirrors `presets::Baseline<DIM>()` in HypercubeRC but inlined here -- the
// fractal repo doesn't ship Presets.h. Tuned for the 4-class classification
// demo on a DEPTH=2 leaf cube (effective DIM=6, 64 features).
ReadoutConfig make_cnn_cfg()
{
    ReadoutConfig c;
    c.num_outputs   = NUM_CLASSES;
    c.task          = ReadoutTask::Classification;
    c.num_layers    = 1;
    c.conv_channels = 8;
    c.epochs        = 50;
    const size_t shift = std::min<size_t>(3 * DEPTH - 1, 9); // cap batch at 512
    c.batch_size    = 1 << shift;
    c.lr_max        = 0.0015f;
    c.lr_min_frac   = 0.1f;
    c.seed          = 42007;
    return c;
}

double analyze_and_print(const std::vector<size_t>& predictions,
                         const size_t* test_labels,
                         size_t test_size,
                         size_t train_size,
                         size_t block_size)
{
    std::cout << "--- Results (test set: " << test_size << " samples) ---\n\n";

    size_t confusion[NUM_CLASSES][NUM_CLASSES] = {};
    size_t correct = 0;
    for (size_t t = 0; t < test_size; ++t)
    {
        confusion[test_labels[t]][predictions[t]]++;
        if (test_labels[t] == predictions[t])
            ++correct;
    }

    double accuracy = 100.0 * static_cast<double>(correct) / static_cast<double>(test_size);

    std::cout << "Overall accuracy: " << std::fixed << std::setprecision(1)
              << accuracy << "%\n\n";

    std::cout << "Per-class breakdown:\n";
    for (size_t c = 0; c < NUM_CLASSES; ++c)
    {
        size_t total_c = 0;
        for (size_t p = 0; p < NUM_CLASSES; ++p)
            total_c += confusion[c][p];
        double acc = (total_c > 0) ? 100.0 * confusion[c][c] / total_c : 0.0;
        std::cout << "  " << CLASS_NAMES[c] << "  " << std::setprecision(1)
                  << std::setw(5) << acc << "%  (" << confusion[c][c]
                  << "/" << total_c << ")";

        if (acc >= 99.9)
            std::cout << "  -- perfectly separable";
        else if (acc >= 98.0)
            std::cout << "  -- near-perfect";
        else if (acc < 90.0)
        {
            size_t max_conf = 0;
            size_t max_conf_class = c;
            for (size_t p = 0; p < NUM_CLASSES; ++p)
            {
                if (p != c && confusion[c][p] > max_conf)
                {
                    max_conf = confusion[c][p];
                    max_conf_class = p;
                }
            }
            if (max_conf > 0)
            {
                double conf_pct = 100.0 * max_conf / total_c;
                std::cout << "  -- confused with " << CLASS_NAMES[max_conf_class]
                          << " " << std::setprecision(0) << conf_pct << "% of the time";
            }
        }
        std::cout << "\n";
    }

    std::cout << "\nConfusion matrix (rows=actual, cols=predicted):\n";
    std::cout << "               ";
    for (size_t c = 0; c < NUM_CLASSES; ++c)
        std::cout << CLASS_NAMES[c] << "  ";
    std::cout << "\n";

    for (size_t a = 0; a < NUM_CLASSES; ++a)
    {
        size_t total_a = 0;
        for (size_t p = 0; p < NUM_CLASSES; ++p)
            total_a += confusion[a][p];

        std::cout << "  " << CLASS_NAMES[a] << " |";
        for (size_t p = 0; p < NUM_CLASSES; ++p)
        {
            double pct = (total_a > 0) ? 100.0 * confusion[a][p] / total_a : 0.0;
            std::cout << std::setw(7) << std::setprecision(1) << pct << "%  ";
        }
        std::cout << "\n";
    }

    std::cout << "\nLock-on dynamics (steps after block transition):\n";
    std::cout << "  Steps after switch  | Accuracy\n";
    std::cout << "  --------------------+---------\n";

    const size_t margins[] = {3, 5, 10, 20, block_size};
    for (size_t margin : margins)
    {
        size_t ok = 0, total = 0;

        for (size_t t = 0; t < test_size; ++t)
        {
            size_t global_t = train_size + t;
            size_t pos_in_block = global_t % block_size;

            if (pos_in_block < margin)
            {
                ++total;
                if (predictions[t] == test_labels[t]) ++ok;
            }
        }

        double acc = (total > 0) ? 100.0 * ok / total : 0.0;
        if (margin == block_size)
            std::cout << "  Entire block        |  " << std::setprecision(1)
                      << std::setw(5) << acc << "%  (overall)\n";
        else
            std::cout << "  0 - " << std::setw(2) << margin
                      << "              |  " << std::setprecision(1)
                      << std::setw(5) << acc << "%\n";
    }
    std::cout << "\n";
    return accuracy;
}

} // namespace

int main()
{
    constexpr size_t LEAVES = FractalReservoir<DEPTH>::leaf_count;

    // Surveyed per-DEPTH level dynamics, with a hand-picked seed override.
    // The original (flat) example detuned leak_rate up to 0.65 to make
    // classification errors at block transitions visible. The surveyed
    // DEPTH=2 defaults run leak_rate=0.15 per level; the demo keeps them
    // as-is. leak_rate is the knob to revisit if lock-on speed after a
    // block transition needs tuning.
    auto fcfg = FractalReservoirConfigDefaults::For<DEPTH>();
    fcfg.seed = 2372;

    std::cout << "=== FractalHypercubeRC: Signal Classification ===\n\n";
    std::cout << "Task: identify which waveform is currently being fed to the reservoir,\n";
    std::cout << "using only the fractal's leaf states -- not the input directly.\n\n";
    std::cout << "Four waveforms cycle in blocks of " << BLOCK_SIZE << " steps:\n";
    std::cout << "  Sine (f=0.11)  |  Square (f=0.13)  |  Triangle (f=0.12)  |  Chirp (f=0.10)\n";
    std::cout << "Frequencies are deliberately close; " << NOISE_LEVEL
              << " noise forces classification by shape, not frequency.\n\n";

    std::vector<float> signal(WARMUP + COLLECT);
    std::vector<size_t> labels(COLLECT);

    std::mt19937_64 noise_rng(fcfg.seed + 555);
    std::uniform_real_distribution<float> noise(-NOISE_LEVEL, NOISE_LEVEL);

    for (size_t t = 0; t < WARMUP; ++t)
        signal[t] = GenerateWaveform(0, CLASS_FREQ[0] * static_cast<float>(t))
                  + noise(noise_rng);

    for (size_t t = 0; t < COLLECT; ++t)
    {
        size_t block_idx = t / BLOCK_SIZE;
        size_t waveform = block_idx % NUM_CLASSES;
        size_t t_in_block = t % BLOCK_SIZE;

        labels[t] = waveform;
        float phase = CLASS_FREQ[waveform] * static_cast<float>(t_in_block);
        signal[WARMUP + t] = GenerateWaveform(waveform, phase) + noise(noise_rng);
    }

    const size_t train_size = static_cast<size_t>(COLLECT * TRAIN_FRACTION);
    const size_t test_size  = COLLECT - train_size;
    const size_t* test_labels = labels.data() + train_size;

    ESN<DEPTH> esn(fcfg);

    std::cout << "Config: DEPTH=" << DEPTH
              << "  leaves=" << LEAVES
              << "  effective DIM=" << (3 * DEPTH)
              << "  Task=Classification  Classes=" << NUM_CLASSES << "\n";

    esn.Warmup(std::span<const float>(signal.data(), WARMUP));
    esn.Run   (std::span<const float>(signal.data() + WARMUP, COLLECT));

    std::vector<float> float_labels(COLLECT);
    for (size_t t = 0; t < COLLECT; ++t)
        float_labels[t] = static_cast<float>(labels[t]);

    auto cnn_cfg = make_cnn_cfg();

    std::cout << "Training: " << cnn_cfg.epochs << " epochs, batch=" << cnn_cfg.batch_size
              << ", lr_max=" << std::setprecision(4) << cnn_cfg.lr_max
              << " (cosine floor " << cnn_cfg.lr_max * cnn_cfg.lr_min_frac << ")\n";
    std::cout << "Training..." << std::flush;
    auto t0 = std::chrono::steady_clock::now();
    esn.Train(std::span<const float>(float_labels.data(), train_size), cnn_cfg);
    auto t1 = std::chrono::steady_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cout << " done (" << std::fixed << std::setprecision(1) << secs << "s)\n\n";

    std::vector<size_t> predictions(test_size);
    std::vector<float> logits(NUM_CLASSES);
    for (size_t t = 0; t < test_size; ++t)
    {
        esn.PredictRaw(train_size + t, logits.data());
        size_t predicted = 0;
        float best = logits[0];
        for (size_t c = 1; c < NUM_CLASSES; ++c)
        {
            if (logits[c] > best)
            {
                best = logits[c];
                predicted = c;
            }
        }
        predictions[t] = predicted;
    }

    analyze_and_print(predictions, test_labels, test_size, train_size, BLOCK_SIZE);

    std::cout << "The HCNN multi-class readout classified waveforms from the fractal's\n";
    std::cout << "leaf dynamics, with conv kernels operating on the " << (3 * DEPTH)
              << "-cube of leaf states.\n";

    return 0;
}
