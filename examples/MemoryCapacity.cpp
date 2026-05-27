/// @file MemoryCapacity.cpp
/// @brief Independent memory-capacity diagnostic for the fractal hypercube
///        reservoir. Mirrors the same-named tool in HypercubeRC so the two
///        reservoir families can be compared on the canonical MC metric.
///
/// Implements the standard Jaeger (2001) MC measurement:
///   1. Drive the reservoir with i.i.d. white noise u(t) ~ Uniform[-1, +1].
///   2. Collect M state vectors (rows) into the F-column matrix X. We read
///      the deepest-level state (`Leaves()`, length 8^DEPTH) — this is the
///      readout-relevant view, analogous to flat HypercubeRC reading
///      `Outputs()` (length 2^DIM). F = min(leaf_count, kFeatureCap).
///   3. Split the rows: first 70% train the ridge readout, last 30% evaluate.
///   4. For each lag k in [1, K_max], fit (XᵀX + λI)w = Xᵀy on the train
///      rows, then compute the squared Pearson correlation r²(k) between
///      target and prediction on the held-out test rows.
///   5. Report r²(k) for each lag and the total MC = Σ_k r²(k).
///
/// The reservoir is not modified — only its raw state is read. No HCNN, no
/// LM_Text coupling. Edit the constexpr parameters and the per-level config
/// loop in `RunMC` to probe a different config (DEPTH, per-level spectral
/// radius / input_scaling / leak_rate / alpha, etc.).
///
/// Held-out evaluation matters: in-sample R² on M ~ a few × F samples
/// overestimates the population r² by a roughly lag-independent margin,
/// inflating the headline MC and every threshold crossing. We pay one
/// Cholesky factorization (over the train Gram) and do the per-lag
/// evaluation on the test split.
///
/// Squared Pearson correlation rather than the regression R² = 1 - SS_res/SS_tot.
/// They agree when the model has an intercept; our linear readout doesn't,
/// so the Pearson form is the canonical MC metric and is always in [0, 1].
///
/// Cost: dominated by building the F×F train Gram matrix and Cholesky
/// factoring it. The per-step reservoir drive is O(state_size) =
/// O(N + N^2 + ... + N^DEPTH) — geometric in DEPTH. At DEPTH 4 with
/// F=4096 and M_train ~ 10k, expect a couple of minutes in Release.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <span>
#include <vector>

#include "FractalReservoir.h"

namespace {

/// In-place lower-triangular Cholesky factorization of an n×n symmetric
/// positive-definite matrix stored row-major. After return, the lower
/// triangle of `G` holds L such that the original G = L · Lᵀ. Returns
/// false if a non-positive pivot is encountered.
bool CholeskyInPlace(double* G, std::size_t n)
{
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j <= i; ++j)
        {
            double s = G[i * n + j];
            for (std::size_t k = 0; k < j; ++k)
                s -= G[i * n + k] * G[j * n + k];
            if (i == j)
            {
                if (s <= 0.0) return false;
                G[i * n + i] = std::sqrt(s);
            }
            else
            {
                G[i * n + j] = s / G[j * n + j];
            }
        }
    }
    return true;
}

/// Solve L · Lᵀ · x = b in place (b becomes x). L is the lower-triangular
/// factor produced by CholeskyInPlace stored in the lower triangle of `L`.
void CholeskySolveInPlace(const double* L, double* b, std::size_t n)
{
    // Forward substitution: L·y = b.
    for (std::size_t i = 0; i < n; ++i)
    {
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j)
            s -= L[i * n + j] * b[j];
        b[i] = s / L[i * n + i];
    }
    // Backward substitution: Lᵀ·x = y.
    for (std::size_t i = n; i > 0; --i)
    {
        const std::size_t ii = i - 1;
        double s = b[ii];
        for (std::size_t j = ii + 1; j < n; ++j)
            s -= L[j * n + ii] * b[j];
        b[ii] = s / L[ii * n + ii];
    }
}

/// Build G = Xᵀ·X for X stored row-major as M rows of F features.
/// G is written full (symmetric).
void BuildGram(const double* X, std::size_t M, std::size_t F, double* G)
{
    std::fill(G, G + F * F, 0.0);
    for (std::size_t t = 0; t < M; ++t)
    {
        const double* xt = X + t * F;
        for (std::size_t i = 0; i < F; ++i)
        {
            const double xi = xt[i];
            double* Grow = G + i * F;
            for (std::size_t j = 0; j <= i; ++j)
                Grow[j] += xi * xt[j];
        }
    }
    for (std::size_t i = 0; i < F; ++i)
        for (std::size_t j = 0; j < i; ++j)
            G[j * F + i] = G[i * F + j];
}

/// xty[f] = sum_t X[t,f] * y[t].
void ComputeXtY(const double* X, const double* y,
                std::size_t M, std::size_t F, double* xty)
{
    std::fill(xty, xty + F, 0.0);
    for (std::size_t t = 0; t < M; ++t)
    {
        const double yt = y[t];
        const double* xt = X + t * F;
        for (std::size_t f = 0; f < F; ++f)
            xty[f] += xt[f] * yt;
    }
}

template <std::size_t DEPTH>
void RunMC()
{
    // ---------------- Configuration (edit these) ----------------
    constexpr std::size_t kFeatureCap   = 8192;   ///< cap on # leaf features used as regressors
    constexpr std::size_t kTWarmup      = 1000;   ///< steps fed to reservoir before any state is recorded
    constexpr std::size_t kTCollect     = 15000;  ///< post-warmup steps whose state is collected
    constexpr std::size_t kKMax         = 100;    ///< largest lag tested
    constexpr double      kTrainFrac    = 0.7;    ///< fraction of usable rows used to fit the readout
    constexpr double      kRidgeLambda  = 1e-4;   ///< Tikhonov regularization on the train Gram diagonal
    constexpr std::uint64_t kInputSeed  = 0xC0FFEEULL;

    // Reservoir config: start from the per-DEPTH validated defaults, then
    // override per-level fields to probe a different operating point. The
    // defaults encode the StateRank-surveyed sr / leak / input_scaling for
    // each level of a standalone DEPTH=K fractal; override below if you
    // want a uniform sweep (e.g. force every level to sr=0.95) for a
    // cleaner comparison against flat-hypercube MC at the same F.
    FractalReservoirConfig<DEPTH> rcfg = FractalReservoirConfigDefaults::For<DEPTH>();
    rcfg.num_inputs = 1;
    // Example uniform override — uncomment & edit to probe:
    // for (auto& lvl : rcfg.levels) {
    //     lvl.spectral_radius = 0.95f;
    //     lvl.input_scaling   = 0.05f;
    //     lvl.leak_rate       = 0.15f;
    //     lvl.alpha           = 1.0f;
    // }
    // ------------------------------------------------------------

    static_assert(kKMax < kTWarmup, "kKMax must be smaller than kTWarmup");
    static_assert(kKMax < kTCollect, "kKMax must be smaller than kTCollect");

    using ResType = FractalReservoir<DEPTH>;
    constexpr std::size_t kLeafCount  = ResType::leaf_count;       // 8^DEPTH
    constexpr std::size_t kStateSize  = ResType::state_size;       // N + N^2 + ... + N^DEPTH
    const std::size_t F = std::min<std::size_t>(kLeafCount, kFeatureCap);
    const std::size_t M = kTCollect - kKMax;  // usable samples per lag
    const std::size_t M_train = static_cast<std::size_t>(static_cast<double>(M) * kTrainFrac);
    const std::size_t M_test  = M - M_train;

    std::cout << "=== FractalHypercubeRC: Memory Capacity ===\n\n";
    std::cout << "Reservoir : DEPTH=" << DEPTH
              << "  leaf_count=" << kLeafCount
              << "  state_size=" << kStateSize
              << "  seed=" << rcfg.seed
              << "  num_inputs=" << rcfg.num_inputs << "\n";
    std::cout << "Per-level : (d  alpha  sr     leak   input_scaling)\n";
    for (std::size_t d = 0; d < DEPTH; ++d)
    {
        std::cout << "             " << d
                  << "   " << std::fixed << std::setprecision(3) << rcfg.levels[d].alpha
                  << "  "  << rcfg.levels[d].spectral_radius
                  << "  "  << rcfg.levels[d].leak_rate
                  << "  "  << rcfg.levels[d].input_scaling << "\n";
    }
    std::cout << std::defaultfloat;
    std::cout << "Features  : " << F << " (first " << F << " leaf states)\n";
    std::cout << "Drive     : T_warmup=" << kTWarmup
              << "  T_collect=" << kTCollect
              << "  K_max=" << kKMax
              << "  M(usable)=" << M << "\n";
    std::cout << "Split     : M_train=" << M_train
              << "  M_test=" << M_test
              << "  train/F=" << std::fixed << std::setprecision(2)
              << (static_cast<double>(M_train) / static_cast<double>(F)) << "\n";
    std::cout << std::defaultfloat;
    std::cout << "Regression: ridge lambda=" << kRidgeLambda
              << "  input_seed=0x" << std::hex << kInputSeed << std::dec << "\n\n";

    // ---- 1. Generate white-noise drive ----
    std::vector<float> u(kTWarmup + kTCollect);
    {
        std::mt19937_64 rng(kInputSeed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (auto& v : u) v = dist(rng);
    }

    // ---- 2. Drive reservoir, collect state matrix X ----
    auto reservoir = ResType::Create(rcfg);

    std::vector<double> X(static_cast<std::size_t>(M) * F);
    for (std::size_t t = 0; t < kTWarmup + kTCollect; ++t)
    {
        const float u_t = u[t];
        reservoir->UpdateState(std::span<const float>(&u_t, 1));
        if (t < kTWarmup) continue;
        const std::size_t row = t - kTWarmup;
        if (row < kKMax) continue;  // discard first K rows; align with lag K targets
        const std::size_t out_row = row - kKMax;
        const float* leaves = reservoir->Leaves();
        double* dst = X.data() + out_row * F;
        for (std::size_t f = 0; f < F; ++f)
            dst[f] = static_cast<double>(leaves[f]);
    }

    // ---- 3. Build & factor the regularized train-Gram matrix ----
    // Gram is built on the first M_train rows only — the readout is fit
    // on those, and the held-out last M_test rows are used to score r².
    std::cout << "Building train Gram matrix (" << F << "x" << F << ") ... " << std::flush;
    std::vector<double> G(static_cast<std::size_t>(F) * F);
    BuildGram(X.data(), M_train, F, G.data());
    for (std::size_t i = 0; i < F; ++i) G[i * F + i] += kRidgeLambda;
    std::cout << "done.\n" << std::flush;

    std::cout << "Cholesky factorization ... " << std::flush;
    if (!CholeskyInPlace(G.data(), F))
    {
        std::cerr << "\nERROR: Gram matrix not positive definite. "
                     "Increase ridge lambda.\n";
        return;
    }
    std::cout << "done.\n\n" << std::flush;

    // ---- 4. Per-lag: fit on train rows, score squared Pearson r² on test ----
    std::vector<double> y(M);
    std::vector<double> w(F);
    std::vector<double> r2(kKMax);
    double total_mc = 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  k    r2(test)\n";
    std::cout << "  ---  --------\n";

    for (std::size_t k = 1; k <= kKMax; ++k)
    {
        // Targets across all M rows: for sample row out_row (state at
        // step kTWarmup + kKMax + out_row), the lag-k target is the
        // input at step kTWarmup + kKMax + out_row - k.
        for (std::size_t out_row = 0; out_row < M; ++out_row)
        {
            const std::size_t u_idx = kTWarmup + kKMax + out_row - k;
            y[out_row] = static_cast<double>(u[u_idx]);
        }

        // Fit on train rows only: w = (X_train^T X_train + λI)^-1 X_train^T y_train.
        ComputeXtY(X.data(), y.data(), M_train, F, w.data());
        CholeskySolveInPlace(G.data(), w.data(), F);

        // Score on test rows: squared Pearson correlation between y and ŷ.
        //   r² = (n·Σyŷ - Σy·Σŷ)² / ((n·Σy² - (Σy)²) · (n·Σŷ² - (Σŷ)²))
        double sum_y = 0.0, sum_h = 0.0;
        double sum_y2 = 0.0, sum_h2 = 0.0, sum_yh = 0.0;
        for (std::size_t t = 0; t < M_test; ++t)
        {
            const double* xt = X.data() + (M_train + t) * F;
            double yhat = 0.0;
            for (std::size_t f = 0; f < F; ++f) yhat += xt[f] * w[f];
            const double yt = y[M_train + t];
            sum_y  += yt;
            sum_h  += yhat;
            sum_y2 += yt * yt;
            sum_h2 += yhat * yhat;
            sum_yh += yt * yhat;
        }
        const double n = static_cast<double>(M_test);
        const double num   = n * sum_yh - sum_y * sum_h;
        const double den_y = n * sum_y2 - sum_y * sum_y;
        const double den_h = n * sum_h2 - sum_h * sum_h;
        const double r2_k = (den_y > 0.0 && den_h > 0.0)
                            ? (num * num) / (den_y * den_h)
                            : 0.0;
        r2[k - 1] = r2_k;
        total_mc += r2_k;

        std::cout << "  " << std::setw(3) << k
                  << "  " << std::setw(8) << r2_k << "\n";
    }

    // ---- 5. Headline metrics ----
    auto last_above = [&](double thresh) -> int {
        int last = 0;
        for (std::size_t i = 0; i < kKMax; ++i)
            if (r2[i] > thresh) last = static_cast<int>(i + 1);
        return last;
    };

    std::cout << "\n=== Summary ===\n";
    std::cout << "Total MC = " << std::setprecision(3) << total_mc
              << "  (theoretical max F=" << F << ")\n";
    std::cout << "Last lag with r^2 > 0.50 : k=" << last_above(0.50) << "\n";
    std::cout << "Last lag with r^2 > 0.10 : k=" << last_above(0.10) << "\n";
    std::cout << "Last lag with r^2 > 0.01 : k=" << last_above(0.01) << "\n";
}

}  // namespace

int main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;

    // Edit DEPTH here (or template-instantiate multiple DEPTHs and call each).
    // DEPTH=4 (leaf_count=4096) gives a head-to-head F-match with HypercubeRC DIM=12.
    // DEPTH=5 (leaf_count=32768) is capped to F=kFeatureCap=8192 and matches DIM=13.
    RunMC<4>();
    return 0;
}
