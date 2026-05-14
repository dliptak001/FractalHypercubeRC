#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cstddef>
#include <cstring>
#include <span>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "../FractalReservoir.h"

/// Effective rank of reservoir state space and input-correlated subspace.
/// Operates on the full FractalReservoir<DEPTH> level-major state buffer
/// (size N + N^2 + ... + N^DEPTH, accessed via FractalReservoir::Outputs()),
/// not just the leaf level. This reveals whether outer/intermediate-level
/// states carry rank that the leaf-only readout currently misses.
/// Eigenvalue spectrum via power iteration + per-vertex R2 from 64 lagged inputs.
/// Single-seed by default; optional multi-seed averaging via the seed-list ctor.
template <size_t DEPTH = 2>
class StateRank
{
    static constexpr size_t N = 8;

public:
    /// Single-seed analysis using `cfg.seed`. The diagnostic respects
    /// whatever seed the caller put on the config; no override.
    explicit StateRank(const FractalReservoirConfig<DEPTH>& cfg)
        : config_(cfg), seeds_{cfg.seed}
    {
    }

    /// Multi-seed averaging across `seeds`. `cfg.seed` is ignored — the
    /// listed seeds take its place per run. Use to smooth out single-seed
    /// outliers when characterizing a config.
    StateRank(const FractalReservoirConfig<DEPTH>& cfg, std::vector<uint64_t> seeds)
        : config_(cfg), seeds_(std::move(seeds))
    {
    }

    /// Aggregated diagnostic output (averaged across `seeds_`).
    struct Result
    {
        std::vector<double> eigenvalues;   ///< sorted descending; padded to max_components
        size_t components_computed = 0;    ///< highest count any seed produced before cutoff
        size_t effective_rank = 0;         ///< eigenvalues > 1% of top, across the full vector
        double top_eigenvalue = 0.0;
        double top1_cum_pct = 0.0;         ///< % of total variance in component #1
        double top10_cum_pct = 0.0;        ///< % of total variance in components 1..10
        double input_correlated_pct = 0.0;
        double mean_r2 = 0.0;
        double r2_p95 = 0.0;               ///< 95th percentile of per-vertex r2
        bool linearity_collapsed = false;  ///< r2_p95 > 0.99 (top-tail is pure FIR)
        double high_r2_pct = 0.0;          ///< % of vertices with R² > 0.5
    };

    /// Compute the diagnostic and return it as data. No I/O.
    [[nodiscard]] Result Run(size_t max_components = 30)
    {
        constexpr size_t warmup = 200;
        // collect = 32*N = 256 gives valid = 256 - 64 = 192 samples after
        // the K=64-lag input-correlation regression — enough that the
        // noise tail of `sum_corr_sq` under H0 (true r²=0) lies well below
        // 1.0 (mean ≈ 0.335, std ≈ 0.084 → P(>1) ≈ 0). At the previous
        // 18*N = 144, valid was only 80 and ~9% of purely-recurrent
        // vertices clipped at 1.0 from sampling noise alone, pinning
        // r2_p95 to 1.00 regardless of reservoir quality. Bumping collect
        // here costs ~1.8x per StateRank call but is what makes r2_p95
        // (and r2_avg) actually discriminating.
        constexpr size_t collect = 32 * N;

        std::vector<double> ev_sum(max_components, 0.0);
        size_t ev_count = 0;
        double s_mean_r2 = 0.0, s_r2_p95 = 0.0;
        double s_input_pct = 0.0;
        double s_high_r2_pct = 0.0;

        for (uint64_t seed : seeds_)
        {
            std::mt19937_64 rng(seed + 99);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);

            size_t total = warmup + collect;
            std::vector<float> inputs(total);
            for (size_t i = 0; i < total; ++i)
                inputs[i] = static_cast<float>(dist(rng));

            FractalReservoirConfig<DEPTH> fcfg = config_;
            fcfg.seed = seed;
            fcfg.num_inputs = 1;  // diagnostic always drives single-channel

            auto res = FractalReservoir<DEPTH>::Create(fcfg);

            // Warmup: drive without collecting.
            for (size_t t = 0; t < warmup; ++t)
                res->UpdateState(std::span<const float>(inputs.data() + t, 1));

            // Collect: drive and snapshot the full level-major state buffer
            // (size = N + N^2 + ... + N^DEPTH) per timestep. Bypasses ESN
            // because ESN::Run only stores Leaves(); we want every level.
            constexpr size_t M = FractalReservoir<DEPTH>::state_size;
            std::vector<float> selected(collect * M);
            for (size_t t = 0; t < collect; ++t)
            {
                res->UpdateState(std::span<const float>(inputs.data() + warmup + t, 1));
                std::memcpy(selected.data() + t * M, res->Outputs(), M * sizeof(float));
            }
            const float* states = selected.data();

            std::vector<double> mean(M, 0.0);
            for (size_t t = 0; t < collect; ++t)
                for (size_t v = 0; v < M; ++v)
                    mean[v] += states[t * M + v];
            for (size_t v = 0; v < M; ++v)
                mean[v] /= collect;

            std::vector<double> centered(collect * M);
            for (size_t t = 0; t < collect; ++t)
                for (size_t v = 0; v < M; ++v)
                    centered[t * M + v] = states[t * M + v] - mean[v];

            auto eigenvalues = ComputeEigenvalues(centered, collect, M, max_components, seed);
            ev_count = std::max(ev_count, eigenvalues.size());
            for (size_t i = 0; i < eigenvalues.size(); ++i)
                ev_sum[i] += eigenvalues[i];

            auto [mean_r2, r2_p95, input_pct, high_r2_pct] =
                ComputeInputCorrelation(states, inputs.data() + warmup, collect, M);

            s_mean_r2 += mean_r2;
            s_r2_p95 += r2_p95;
            s_input_pct += input_pct;
            s_high_r2_pct += high_r2_pct;
        }

        const double n = static_cast<double>(seeds_.size());

        Result out;
        out.eigenvalues.assign(ev_sum.begin(), ev_sum.begin() + ev_count);
        for (auto& e : out.eigenvalues) e /= n;
        out.components_computed = ev_count;

        if (!out.eigenvalues.empty()) {
            out.top_eigenvalue = out.eigenvalues[0];
            const double total = std::accumulate(out.eigenvalues.begin(),
                                                 out.eigenvalues.end(), 0.0);
            const double thresh = out.top_eigenvalue * 0.01;
            double cum = 0.0;
            for (size_t i = 0; i < out.eigenvalues.size(); ++i) {
                if (out.eigenvalues[i] > thresh) ++out.effective_rank;
                cum += out.eigenvalues[i];
                if (i == 0)  out.top1_cum_pct  = (total > 0) ? cum / total * 100.0 : 0.0;
                if (i == 9)  out.top10_cum_pct = (total > 0) ? cum / total * 100.0 : 0.0;
            }
            // If fewer than 10 components, top10 == total.
            if (out.eigenvalues.size() < 10)
                out.top10_cum_pct = 100.0;
        }

        out.input_correlated_pct = s_input_pct / n;
        out.mean_r2 = s_mean_r2 / n;
        out.r2_p95  = s_r2_p95 / n;
        out.linearity_collapsed = out.r2_p95 > 0.99;
        out.high_r2_pct = s_high_r2_pct / n;
        return out;
    }

    /// Convenience: Run() + pretty-print to stdout.
    void RunAndPrint(size_t max_components = 30)
    {
        constexpr size_t warmup = 200;
        constexpr size_t collect = 32 * N;  // kept in sync with Run()
        PrintHeader(warmup, collect, max_components, config_);

        const Result r = Run(max_components);

        std::cout << "State covariance eigenvalues (top 10):\n";
        std::cout << "  #  | Eigenvalue | % of max | Cumulative %\n";
        std::cout << "  ---+------------+----------+-------------\n";

        const double max_ev = r.top_eigenvalue;
        const double total_ev = std::accumulate(r.eigenvalues.begin(),
                                                r.eigenvalues.end(), 0.0);
        double cumulative = 0.0;
        const size_t show = std::min(r.eigenvalues.size(), static_cast<size_t>(10));
        for (size_t i = 0; i < show; ++i) {
            const double ev = r.eigenvalues[i];
            cumulative += ev;
            std::cout << "  " << std::setw(2) << (i + 1)
                      << " | " << std::scientific << std::setprecision(3) << std::setw(10) << ev
                      << " | " << std::fixed << std::setprecision(1) << std::setw(7)
                      << (max_ev > 0 ? ev / max_ev * 100.0 : 0.0) << "%"
                      << " | " << std::setw(7)
                      << (total_ev > 0 ? cumulative / total_ev * 100.0 : 0.0) << "%\n";
        }
        std::cout << "  Effective rank (>1% of max): " << r.effective_rank
                  << " of " << r.components_computed << " computed\n";

        std::cout << "\nInput-correlated variance (64 lags";
        if (seeds_.size() > 1) std::cout << ", " << seeds_.size() << "-seed avg";
        std::cout << "):\n";
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  Input-correlated: " << r.input_correlated_pct << "%\n";
        std::cout << std::setprecision(3);
        std::cout << "  Per-vertex R2 (mean): " << r.mean_r2 << "\n";
        std::cout << "  Per-vertex R2 (p95):  " << r.r2_p95 << "\n";
        std::cout << "  Linearity collapsed: "
                  << (r.linearity_collapsed ? "yes" : "no") << "\n";
        std::cout << "  Vertices with R2 > 0.5: " << std::setprecision(1)
                  << r.high_r2_pct << "%\n";
    }

private:
    FractalReservoirConfig<DEPTH> config_;
    std::vector<uint64_t> seeds_;

    struct InputCorr
    {
        double mean_r2, r2_p95, input_pct, high_r2_pct;
    };

    static InputCorr ComputeInputCorrelation(const float* states, const float* input_ptr,
                                              size_t collect, size_t num_verts)
    {
        constexpr size_t K = 64;
        size_t valid = collect - K;

        std::vector<float> lagged(valid * K);
        for (size_t t = 0; t < valid; ++t)
            for (size_t k = 0; k < K; ++k)
                lagged[t * K + k] = input_ptr[K + t - k];

        // Adjusted-R² bias correction:
        //   R²_adj = 1 - (1 - R²_raw) * (valid - 1) / (valid - K - 1)
        // Standard small-sample correction for K predictors on n samples.
        // Removes the K/(n-1) noise floor at H0 *and* tapers smoothly to
        // zero correction as R²_raw → 1 (a perfect-FIR vertex still reports
        // R²=1, not 0.19 — which is what a flat `R² - K/(n-1)` subtraction
        // gives). At collect=32*N=256 and K=64, valid=192 → valid - K - 1
        // = 127 → adj_mult ≈ 1.50, a mild correction well clear of the
        // noise blow-up you get when K crowds valid.
        if (valid <= K + 1)
            throw std::invalid_argument(
                "StateRank::ComputeInputCorrelation: collect must satisfy "
                "collect - K > K + 1 (K=64 lags); bump collect");
        const double adj_mult = static_cast<double>(valid - 1)
                              / static_cast<double>(valid - K - 1);

        double total_var = 0.0, input_var = 0.0;
        size_t high_r2_count = 0;
        double sum_r2 = 0.0;
        std::vector<double> r2s;
        r2s.reserve(num_verts);

        for (size_t v = 0; v < num_verts; ++v)
        {
            double mean_s = 0.0;
            for (size_t t = 0; t < valid; ++t)
                mean_s += states[(K + t) * num_verts + v];
            mean_s /= valid;

            double var_s = 0.0;
            for (size_t t = 0; t < valid; ++t)
            {
                double s = states[(K + t) * num_verts + v] - mean_s;
                var_s += s * s;
            }
            var_s /= valid;
            total_var += var_s;

            double sum_corr_sq = 0.0;
            for (size_t k = 0; k < K; ++k)
            {
                double mean_i = 0.0;
                for (size_t t = 0; t < valid; ++t)
                    mean_i += lagged[t * K + k];
                mean_i /= valid;

                double cov = 0.0, var_i = 0.0;
                for (size_t t = 0; t < valid; ++t)
                {
                    double s = states[(K + t) * num_verts + v] - mean_s;
                    double i = lagged[t * K + k] - mean_i;
                    cov += s * i;
                    var_i += i * i;
                }
                if (var_i > 1e-12 && var_s > 1e-12)
                {
                    double corr = cov / std::sqrt(var_s * valid * var_i);
                    sum_corr_sq += corr * corr;
                }
            }

            // Clamp raw to [0, 1] first — sum_corr_sq can drift above 1.0
            // from finite-sample noise (theoretical max under white input
            // is 1.0 in the population limit). Then apply adjusted-R².
            const double r2_raw = std::min(sum_corr_sq, 1.0);
            const double r2 = std::clamp(1.0 - (1.0 - r2_raw) * adj_mult, 0.0, 1.0);
            input_var += var_s * r2;
            sum_r2 += r2;
            r2s.push_back(r2);
            if (r2 > 0.5) ++high_r2_count;
        }

        // p95 via nth_element — O(N) and we don't need the full sort.
        double r2_p95 = 0.0;
        if (!r2s.empty()) {
            size_t idx = static_cast<size_t>(0.95 * (r2s.size() - 1));
            std::nth_element(r2s.begin(), r2s.begin() + idx, r2s.end());
            r2_p95 = r2s[idx];
        }

        return {sum_r2 / num_verts, r2_p95,
                total_var > 0 ? input_var / total_var * 100.0 : 0.0,
                high_r2_count * 100.0 / num_verts};
    }

    static std::vector<double> ComputeEigenvalues(const std::vector<double>& centered,
                                                   size_t collect, size_t num_verts,
                                                   size_t max_components, uint64_t seed)
    {
        std::vector<double> eigenvalues;
        std::vector<std::vector<double>> eigenvectors;

        for (size_t comp = 0; comp < max_components && comp < num_verts; ++comp)
        {
            std::vector<double> q(num_verts);
            std::mt19937_64 rng(seed + 99999 + comp);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            double norm = 0.0;
            for (size_t v = 0; v < num_verts; ++v)
            {
                q[v] = dist(rng);
                norm += q[v] * q[v];
            }
            norm = std::sqrt(norm);
            for (size_t v = 0; v < num_verts; ++v) q[v] /= norm;

            double eigenvalue = 0.0;
            for (int iter = 0; iter < 100; ++iter)
            {
                std::vector<double> y(collect, 0.0);
                for (size_t t = 0; t < collect; ++t)
                    for (size_t v = 0; v < num_verts; ++v)
                        y[t] += centered[t * num_verts + v] * q[v];

                std::vector<double> z(num_verts, 0.0);
                for (size_t t = 0; t < collect; ++t)
                    for (size_t v = 0; v < num_verts; ++v)
                        z[v] += centered[t * num_verts + v] * y[t];

                for (size_t v = 0; v < num_verts; ++v)
                    z[v] /= collect;

                for (size_t p = 0; p < eigenvectors.size(); ++p)
                {
                    double dot = 0.0;
                    for (size_t v = 0; v < num_verts; ++v)
                        dot += z[v] * eigenvectors[p][v];
                    for (size_t v = 0; v < num_verts; ++v)
                        z[v] -= dot * eigenvectors[p][v];
                }

                norm = 0.0;
                for (size_t v = 0; v < num_verts; ++v) norm += z[v] * z[v];
                norm = std::sqrt(norm);
                eigenvalue = norm;

                if (norm > 1e-15)
                    for (size_t v = 0; v < num_verts; ++v) q[v] = z[v] / norm;
                else
                    break;
            }

            if (eigenvalue < 1e-12) break;
            eigenvalues.push_back(eigenvalue);
            eigenvectors.push_back(q);
        }

        return eigenvalues;
    }

    void PrintHeader(size_t warmup, size_t collect, size_t max_components,
                     const FractalReservoirConfig<DEPTH>& cfg) const
    {
        std::cout << "=== State Rank Analysis (full FractalReservoir state, ";
        if (seeds_.size() == 1)
            std::cout << "single seed)";
        else
            std::cout << seeds_.size() << "-seed avg)";
        std::cout << " ===\n";
        std::cout << "Seeds: {";
        for (size_t i = 0; i < seeds_.size(); ++i)
            std::cout << (i ? "," : "") << seeds_[i];
        std::cout << "}\n";
        for (size_t d = 0; d < DEPTH; ++d) {
            const auto& lvl = cfg.levels[d];
            std::cout << "  L" << d
                      << " | Alpha: " << lvl.alpha
                      << " | Leak: " << lvl.leak_rate
                      << " | SR: " << lvl.spectral_radius
                      << " | Input scaling: " << lvl.input_scaling << "\n";
        }
        std::cout << "DEPTH=" << DEPTH << "  N=" << N
                  << "  state_size=" << FractalReservoir<DEPTH>::state_size
                  << "  (leaves=" << FractalReservoir<DEPTH>::leaf_count << ")"
                  << "  Warmup: " << warmup << " | Collect: " << collect
                  << " | Max components: " << max_components << "\n\n";
    }
};
