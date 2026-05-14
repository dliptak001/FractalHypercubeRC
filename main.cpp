#include "diagnostics/NARMA_N_Benchmark.h"
#include "diagnostics/StateRankSurvey.h"

#include <cstdio>
#include <string>

// Diagnostic driver — wires the header-only investigation harness
// (diagnostics/) into a runnable program. Two #if-gated blocks:
//
//  1.  StateRank seed sweep — ranks candidate seeds by the survey's
//      "rank desc → inp asc" criterion (see diagnostics/StateRankSurvey.h)
//      and prints the ranked table.
//
//  1b. Single-candidate NARMA run — a hand-specified seed + per-level
//      config, for cross-checking one operating point in isolation.
//
// Each block is guarded by its own `#if 1`; flip to `#if 0` to skip it.
int main()
{
    constexpr size_t DEPTH = 3;

    auto base_cfg = FractalReservoirConfigDefaults::For<DEPTH>();

    // Readout config for the NARMA run below — middle ground between the
    // production ReadoutConfig{} defaults (epochs=200, conv_channels=16;
    // minutes at DEPTH≥3) and a stunted config too small to learn NARMA at
    // all. At these settings NRMSE should land in the 0.2-0.5 range for a
    // healthy reservoir (literature ESN+NARMA-10 baseline). Tens of seconds
    // at DEPTH=3; trim/bump based on observed run time.
    ReadoutConfig validate_readout;
    validate_readout.epochs        = 600;
    validate_readout.batch_size    = 64;
    validate_readout.conv_channels = 8;
    validate_readout.num_layers    = 1;

#if 1
    // ─── Phase 1: StateRank hyperparameter suite ─────────────────────────

    StateRankSurvey<DEPTH>::Options opt;
    // Ceiling = min(state_size, collect) = min(584, 144) = 144 at DEPTH=3.
    // Below 144 the rank field saturates under the default per-level
    // dynamics and the rank-desc primary key collapses into a tie, falling
    // through to inp% and losing primary discrimination. 144 restores it.
    opt.max_components = 144;
    // healthy_top_eig defaults to 1e-5 (see StateRankSurvey::DefaultHealthyTopEig);
    // override here if a sweep needs a different magnitude-collapse floor.

    StateRankSurvey<DEPTH> survey(base_cfg, opt);
    survey.PrintConfigBanner();
    //survey.RunHyperparameterSuite();

    // Sweep seeds 2000-3000, rank and print the top 10.
    survey.PrintRankedTop(survey.SweepSeeds(2000, 3000), 10);

#endif

#if 1
    // ─── Phase 1b: Single-candidate NARMA run ────────────────────────────
    // Hand-specified seed + per-level dynamics. Useful for cross-checking
    // a survey winner under tweaked hyperparameters, or comparing a
    // custom pick against the surveyed default. Edit `single_seed` and
    // (optionally) uncomment + edit the per-level overrides. Readout
    // training uses `validate_readout` above — replace if you want a
    // different HCNN config for this run.
    {
        auto single_cfg = base_cfg;
        single_cfg.seed = 2897;   // edit to cross-check a survey winner

        // Per-level overrides — uncomment any you want to tweak. The
        // defaults from FractalReservoirConfigDefaults::For<DEPTH>() are
        // already in `base_cfg`; touching a level here overrides only that
        // one. Field order: {alpha, spectral_radius, leak_rate, input_scaling}.
        //single_cfg.levels[0] = {1.0f, 0.92f, 0.15f, 0.05f};
        //single_cfg.levels[1] = {1.0f, 0.93f, 0.15f, 0.05f};
        //single_cfg.levels[2] = {1.0f, 0.95f, 0.20f, 0.05f};

        constexpr size_t single_narma_order = 12;

        std::printf("\nSingle-candidate NARMA-%zu (DEPTH=%zu, seed=%llu):\n",
                    single_narma_order, DEPTH,
                    static_cast<unsigned long long>(single_cfg.seed));
        for (size_t d = 0; d < DEPTH; ++d) {
            const auto& lvl = single_cfg.levels[d];
            std::printf("  L%zu: alpha=%.2f SR=%.2f leak=%.2f input=%.2f\n",
                        d, lvl.alpha, lvl.spectral_radius,
                        lvl.leak_rate, lvl.input_scaling);
        }
        std::fflush(stdout);

        // HCNN readout on leaves only — the default ESN<DEPTH> path.
        // 2^(3*DEPTH) leaf features, arranged as a 3*DEPTH-dim hypercube
        // for the conv stack. Outer + intermediate fractal levels are
        // not exposed to this readout.
        NARMA_N_Benchmark<DEPTH> narma(single_cfg, single_narma_order, validate_readout);
        const auto r_hcnn = narma.Run();
        std::printf("  HCNN  (leaves, P=%5zu) NRMSE=%.4f  train=%.1f s\n",
                    FractalReservoir<DEPTH>::leaf_count,
                    r_hcnn.nrmse, r_hcnn.train_time_s);
        std::fflush(stdout);
    }
#endif
    return 0;
}
