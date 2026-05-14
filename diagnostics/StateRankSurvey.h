    #pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../FractalReservoir.h"
#include "StateRank.h"

/// Thin survey wrapper around `StateRank<DEPTH>` for sweeping a single axis
/// (seeds, spectral_radius, leak_rate, input_scaling, ...) across a range of
/// values. Each row is one StateRank evaluation under a perturbed config.
///
/// The class owns the per-row tabular printing and the seed-survey ranking
/// (filter magnitude-collapsed, sort by rank desc / inp%% asc / top_eig desc).
/// Title and any per-axis banner stay at the caller — sweeps over different
/// config knobs want different headlines.
///
/// ## Reading the row table
///
/// Every survey row prints the columns:
///   `top_eig | eff | top1% | top10% | inp% | r2_avg | r2_p95`
///
/// A trailing `*` on `r2_p95` flags a linearity collapse (p95 > 0.99) — the
/// top-5%-most-linear vertices are all pure FIRs.
///
/// What we want to see in each column:
///
///   `top_eig`     — magnitude of the dominant covariance mode. Want it
///                   comfortably above the magnitude-collapse floor
///                   (`healthy_top_eig`, default 1e-5 uniform). Under the
///                   input-queue regime healthy rich-state seeds land at
///                   1e-5..1e-4; SR≈1 dominant-mode seeds land at 1e-1..1.
///                   `< 1e-5` means tanh never engages and state is at the
///                   floating-point noise floor. Numbers >> 1 are fine in
///                   absolute terms but lose discrimination — they are
///                   dominated by a small number of large modes.
///
///   `eff`         — effective rank: count of eigenvalues above 1% of
///                   `top_eig`. The "useful dimensionality" of state.
///                   **HIGHER is better.** This is the primary sort key.
///                   Capped by `max_components` — if eff hits the cap,
///                   every row ties on rank and the sort falls through
///                   to inp%, losing primary discrimination. Bump
///                   `max_components` if you see saturation.
///
///   `top1%`       — variance fraction held by the single dominant mode.
///                   **LOWER is better.** > 90% is degenerate (one direction
///                   eats the spectrum); 25-40% is excellent (broad spread).
///
///   `top10%`      — variance fraction in the top 10 modes. Want it
///                   moderate: a value that climbs well past 50% with a low
///                   `top1%` means a healthy spread across several modes;
///                   pushing toward 100% means the spectrum collapsed into
///                   just the top 10.
///
///   `inp%`        — fraction of state variance explainable by 64 lags of
///                   the input. **Band-targeted: 40-70% is the healthy
///                   range.** ≥ 70% triggers the linearity-collapse filter
///                   (reservoir is essentially a 64-tap FIR of input).
///                   < 40% triggers the dead-recurrent collapse filter
///                   (state geometrically rich but functionally
///                   disconnected from the input — autonomous drift,
///                   NARMA worse than predict-zero). The middle band
///                   carries the seeds where recurrent dynamics
///                   contribute meaningful share alongside input
///                   coupling.
///
///   `r2_avg`      — per-vertex R² to 64 lagged inputs, averaged. Same
///                   intent as `inp%` but measured at the vertex level.
///                   **LOWER is better.** Tracks `inp%` closely.
///
///   `r2_p95`      — 95th-percentile of per-vertex R². **LOWER is better.**
///                   p95 < ~0.6 means even the top-tail of vertices retains
///                   substantial nonlinearity (no large fraction of the cube
///                   is doing pure FIR). `> 0.99` triggers the `linearity_
///                   collapsed` flag (trailing `*` in the row).
///                   Replaces the earlier `r2_min/max` columns — min/max
///                   were extreme-value statistics over thousands of
///                   vertices and saturated at 0.00/1.00 for any non-
///                   trivial state_size; p95 is robust. The estimator
///                   itself is now bias-corrected (subtracts `K/(valid-1)`
///                   from raw `sum_corr_sq`), which is what makes p95
///                   informative — without that correction the noise
///                   floor was ~0.81 and p95 would clamp at 1.0 too.
///
/// **Six-tier strict ordering used by the sort:**
///   `eff desc → inp% asc → r2_avg asc → top1% asc → top_eig desc → label asc`
///
/// **Three failure modes filtered before ranking:**
///   - magnitude collapse: `top_eig < healthy_top_eig` (state never wakes up)
///   - linearity collapse: `inp% ≥ linearity_collapse_input_corr_pct` (FIR-degenerate)
///   - dead-recurrent collapse: `inp% < dead_recurrent_input_corr_pct`
///     (state barely listens to the input; autonomous noise)
template <size_t DEPTH>
class StateRankSurvey
{
public:
    struct Options
    {
        size_t max_components = 30;
        // Magnitude-collapse cutoff. Before the input-queue rework, state
        // magnitude scaled with state size and the per-vertex L2-normalized
        // input weights landed `top_eig` in the 1e-2..1 range — 0.1 cleanly
        // separated engaged-tanh seeds from collapsed-to-zero ones. After
        // that rework (per-vertex input weights deleted, parent->child
        // queue + depth-0 external-input queue added), input gain is
        // delivered through a small per-level `input_scaling` scalar and
        // rich-state seeds now land with `top_eig` in 1e-5..1e-4 — well
        // below the old threshold. Validated at DEPTH=2: 1e-5 surfaces
        // the eff=18 cluster (rank-near-ceiling, moderate inp%) without
        // re-admitting genuinely collapsed seeds; the linearity-collapse
        // filter below catches the FIR-degenerate failure mode separately.
        float healthy_top_eig = DefaultHealthyTopEig();
        // Linearity-collapse cutoff. Reservoirs whose state is
        // essentially a 64-tap FIR of recent input (input_correlated_pct
        // = 100%, r2_p95 ≈ 1.0) report nominally-high effective_rank
        // and would otherwise dominate the top-N. Second of the three
        // failure modes filtered before ranking (see the class header).
        // Depth-independent: the failure mode is structural, not scale-
        // dependent.
        //
        // Under the input-queue regime (depth-0 external queue +
        // parent->child queue) the leaves of a DEPTH=2 cube can read
        // input from up to ~14 lags via the queue cascade. At low SR
        // / low leak, the leaves are dominated by that FIR fan-out and
        // typically report inp% in the 70-100% band even when eff is at
        // the rank ceiling. 99% only rejects pure-FIR seeds; 70% rejects
        // seeds where input variance dominates the state cloud and
        // selects for a cohort where recurrent dynamics contribute a
        // measurable share of state variance.
        //
        // **Empirical caveat (DEPTH=2 seed band [2000, 2500]):** at this
        // threshold the surviving top-5 NRMSE-on-NARMA-20 was slightly
        // *worse* than at 99% (means ~0.85 vs ~0.83). Geometric
        // recurrence-share does not correlate with — and at this
        // operating point appears mildly anti-correlated with — NARMA
        // performance, consistent with this project's headline finding
        // that StateRank geometry does not predict NARMA. The threshold
        // is kept at 70% as a principled structural filter (reject
        // FIR-dominated states), not as a NARMA-selection signal.
        // (Result::linearity_collapsed reports the canonical pure-FIR
        // failure mode directly from r2_p95 > 0.99 — kept as an
        // independent percentile-based signal alongside this bulk-
        // variance threshold.)
        float linearity_collapse_input_corr_pct = 70.0f;
        // Dead-recurrent floor. Reservoirs whose state barely tracks the
        // input (inp% < 40) are geometrically rich but functionally
        // autonomous — the recurrent dynamics oscillate or drift on
        // their own without responding to the driving signal. Empirically
        // (DEPTH=2 top-20 sweep over seed band [2000, 2500], aggressive
        // regime SR=0.95/1.00 leak=0.70): such seeds NARMA-10 at NRMSE
        // > 1.0 (worse than predict-zero baseline). Catches the
        // "criterion rewards dead state" failure mode that the inp% asc
        // tiebreaker otherwise admits. 40% is the empirical knee — above
        // ~40% the worst NARMA NRMSE in the cohort drops back into the
        // 0.7-0.8 band.
        float dead_recurrent_input_corr_pct = 40.0f;
        bool print_per_row = true;     ///< stream rows to stdout as they're computed
    };

    /// @brief Default `Options::healthy_top_eig`. Constexpr so the value
    ///        is baked into Options' inline initializer.
    ///
    /// Uniform across DEPTH. Before the input-queue rework the threshold
    /// was depth-aware (1e-3 at DEPTH=1, 0.1 elsewhere) on the rationale
    /// that state magnitude scales with state size; under the input-queue
    /// regime `top_eig` is dominated by input gain (per-level
    /// `input_scaling`) and the depth-conditional rationale no longer
    /// holds. 1e-5 is the "state above floating-point noise floor"
    /// boundary; the linearity-collapse filter (`input_correlated_pct >=
    /// 70.0%`) catches the FIR-dominated failure mode independently.
    [[nodiscard]] static constexpr float DefaultHealthyTopEig()
    {
        return 1e-5f;
    }

    struct Row
    {
        std::string label;     ///< leftmost column display (e.g. "42" or "0.85")
        double axis_value;     ///< numeric axis value (for ordering / external plotting)
        FractalReservoirConfig<DEPTH> cfg;            ///< config that produced `result` — assigned by SweepSeeds/Sweep before use
        typename StateRank<DEPTH>::Result result;
    };

    StateRankSurvey(const FractalReservoirConfig<DEPTH>& base_cfg, Options opt = {})
        : base_cfg_(base_cfg), opt_(std::move(opt)) {}

    /// Optional config summary; safe to call before or skip. Mirrors the
    /// banner the seed survey used to print inline in main.
    void PrintConfigBanner() const
    {
        std::printf("State Rank\n");
        std::printf("DEPTH %zu Config: seed=%zu num_inputs=%zu  state_size=%zu  leaves=%zu\n",
                    DEPTH,
                    static_cast<size_t>(base_cfg_.seed),
                    base_cfg_.num_inputs,
                    FractalReservoir<DEPTH>::state_size,
                    FractalReservoir<DEPTH>::leaf_count);
        for (size_t d = 0; d < DEPTH; ++d) {
            const auto& lvl = base_cfg_.levels[d];
            std::printf("  L%zu: alpha=%.2f SR=%.2f leak=%.2f input=%.2f\n",
                        d, lvl.alpha, lvl.spectral_radius,
                        lvl.leak_rate, lvl.input_scaling);
        }
        std::printf("\n");
    }

    /// Seed sweep over [start, end] inclusive. The base config's other
    /// fields are held fixed; only `cfg.seed` varies.
    std::vector<Row> SweepSeeds(uint64_t start, uint64_t end)
    {
        if (opt_.print_per_row) PrintTableHeader("seed");

        std::vector<Row> rows;
        rows.reserve(end - start + 1);
        for (uint64_t seed = start; seed <= end; ++seed)
        {
            FractalReservoirConfig<DEPTH> cfg = base_cfg_;
            cfg.seed = seed;

            Row row;
            row.label      = std::to_string(seed);
            row.axis_value = static_cast<double>(seed);
            row.cfg        = cfg;
            row.result     = StateRank<DEPTH>(cfg).Run(opt_.max_components);

            if (opt_.print_per_row) PrintRow(row);
            rows.push_back(std::move(row));
        }
        return rows;
    }

    /// Generic axis sweep — vary any single config field via `setter`.
    /// The base config's other fields (including `seed`) are held fixed.
    ///
    /// `setter`     : `void(FractalReservoirConfig<DEPTH>&, T)` — applies the value to cfg.
    /// `format_value` : `std::string(T)` — produces the leftmost column label.
    template <typename T, typename Setter, typename Formatter>
    std::vector<Row> Sweep(std::string_view axis_name,
                           const std::vector<T>& values,
                           Setter setter,
                           Formatter format_value)
    {
        if (opt_.print_per_row) PrintTableHeader(axis_name);

        std::vector<Row> rows;
        rows.reserve(values.size());
        for (const T& v : values)
        {
            FractalReservoirConfig<DEPTH> cfg = base_cfg_;
            setter(cfg, v);

            Row row;
            row.label      = format_value(v);
            row.axis_value = static_cast<double>(v);
            row.cfg        = cfg;
            row.result     = StateRank<DEPTH>(cfg).Run(opt_.max_components);

            if (opt_.print_per_row) PrintRow(row);
            rows.push_back(std::move(row));
        }
        return rows;
    }

    /// Run the standard 3-axis hyperparameter sweep suite (SR / leak_rate /
    /// input_scaling) at the seed and base config in `base_cfg_`. Each
    /// sweep prints its own title banner and table; rows are returned only
    /// for the caller's reference (parameter sweeps don't usually rank).
    ///
    /// Default value lists bracket the alive operating range for the recursive
    /// cube under Aggressive defaults — denser sampling around the per-axis
    /// default, wider strokes at the extremes. Pass empty vectors for any axis
    /// you want to skip; pass custom values to override the defaults.
    ///
    /// Methodology note: the rank-30 cap matters for the leak sweep at
    /// leak >= 0.45 — set `opt.max_components = 60` (or higher) before calling
    /// this method if you care about the true rank in the high-leak regime.
    struct SuiteResult
    {
        std::vector<Row> sr_rows;
        std::vector<Row> leak_rows;
        std::vector<Row> input_scaling_rows;
    };

    SuiteResult RunHyperparameterSuite(
        std::vector<float> sr_values    = {0.50f, 0.70f, 0.85f, 0.95f, 1.00f, 1.05f, 1.20f, 1.50f},
        std::vector<float> leak_values  = {0.05f, 0.10f, 0.20f, 0.30f, 0.35f, 0.45f, 0.60f, 0.80f, 1.00f},
        std::vector<float> inpsc_values = {0.02f, 0.05f, 0.10f, 0.15f, 0.20f, 0.30f, 0.50f, 0.80f, 1.20f})
    {
        auto fmt2 = [](float v) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "%.2f", v);
            return std::string(buf);
        };

        SuiteResult out;

        // Each axis sweep applies its value uniformly across every level.
        // Per-level sweeps are future work — this preserves the prior
        // behavior of "one knob, one value, every sub-reservoir".
        if (!sr_values.empty()) {
            std::printf("\n=== StateRank SR sweep: DEPTH=%zu ===\n", DEPTH);
            out.sr_rows = Sweep<float>("SR", sr_values,
                [](FractalReservoirConfig<DEPTH>& cfg, float v) {
                    for (auto& lvl : cfg.levels) lvl.spectral_radius = v;
                },
                fmt2);
        }

        if (!leak_values.empty()) {
            std::printf("\n=== StateRank leak sweep: DEPTH=%zu ===\n", DEPTH);
            out.leak_rows = Sweep<float>("leak", leak_values,
                [](FractalReservoirConfig<DEPTH>& cfg, float v) {
                    for (auto& lvl : cfg.levels) lvl.leak_rate = v;
                },
                fmt2);
        }

        if (!inpsc_values.empty()) {
            std::printf("\n=== StateRank input-scaling sweep: DEPTH=%zu ===\n", DEPTH);
            out.input_scaling_rows = Sweep<float>("inpsc", inpsc_values,
                [](FractalReservoirConfig<DEPTH>& cfg, float v) {
                    for (auto& lvl : cfg.levels) lvl.input_scaling = v;
                },
                fmt2);
        }

        return out;
    }

    /// Outcome of RankAndSelect: the chosen top rows plus filter accounting.
    /// The three `*_collapsed` counts mirror the three failure modes
    /// filtered before ranking (see the class header) so the caller can
    /// spot survey artifacts (e.g. an unexpectedly high magnitude-collapse
    /// rate at a new DEPTH).
    struct Selection
    {
        std::vector<Row> top;                 ///< filtered, sorted, truncated to n
        size_t total_input = 0;               ///< rows handed to RankAndSelect
        size_t magnitude_collapsed = 0;       ///< top_eigenvalue below threshold
        size_t linearity_collapsed = 0;       ///< input_corr above threshold (FIR-degenerate)
        size_t dead_recurrent_collapsed = 0;  ///< input_corr below floor (autonomous noise)
        size_t healthy = 0;                   ///< rows that passed all filters
    };

    /// Filter, sort, and truncate `rows` to the top `n`. Filters out the
    /// three known failure modes (magnitude collapse, linearity collapse,
    /// dead-recurrent collapse), sorts the survivors by the six-tier strict
    /// ordering documented at the top of this file, and returns the top `n`
    /// plus per-mode failure counts.
    ///
    /// Sort: `eff desc → inp% asc → r2_avg asc → top1% asc → top_eig desc
    /// → label asc`. Picks rank-near-ceiling seeds with the lowest input-
    /// correlation among ceiling-rank seeds. The final `label asc`
    /// tiebreaker makes the sort deterministic across reruns.
    ///
    /// Note that the criterion is heuristic: StateRank geometry is
    /// necessary but not sufficient for NARMA functional alignment, and
    /// NARMA-N methodology validation has historically not tracked the
    /// top-ranked seeds.
    [[nodiscard]] Selection RankAndSelect(std::vector<Row> rows,
                                          size_t n = 10) const
    {
        Selection sel;
        sel.total_input = rows.size();

        std::vector<Row> healthy;
        healthy.reserve(rows.size());
        for (auto& row : rows)
        {
            const auto& r = row.result;
            // Magnitude collapse: state hovering near zero, tanh hasn't
            // engaged. Reported effective_rank is over noise.
            if (r.top_eigenvalue < opt_.healthy_top_eig)
            {
                ++sel.magnitude_collapsed;
                continue;
            }
            // Linearity collapse: state is a linear FIR of input, so
            // effective_rank looks fine but the reservoir adds nothing
            // a 64-tap FIR couldn't do.
            if (r.input_correlated_pct >= opt_.linearity_collapse_input_corr_pct)
            {
                ++sel.linearity_collapsed;
                continue;
            }
            // Dead-recurrent collapse: state barely tracks the input,
            // recurrent dynamics drift autonomously. Geometrically rich
            // but functionally disconnected from the driving signal —
            // empirically NARMAs worse than the predict-zero baseline.
            if (r.input_correlated_pct < opt_.dead_recurrent_input_corr_pct)
            {
                ++sel.dead_recurrent_collapsed;
                continue;
            }
            healthy.push_back(std::move(row));
        }
        sel.healthy = healthy.size();

        std::sort(healthy.begin(), healthy.end(), LessBalanced);

        if (healthy.size() > n) healthy.resize(n);
        sel.top = std::move(healthy);
        return sel;
    }

    /// Filter, sort, print. Wrapper around RankAndSelect that emits the
    /// filter accounting and the ranked table to stdout.
    void PrintRankedTop(std::vector<Row> rows, size_t n = 10) const
    {
        const Selection sel = RankAndSelect(std::move(rows), n);
        PrintFilterReport(sel);
        PrintRankedTable(sel.top);
    }

private:
    FractalReservoirConfig<DEPTH> base_cfg_;
    Options opt_;

    void PrintTableHeader(std::string_view axis_name) const
    {
        std::printf(" %5.*s | top_eig  | eff | top1%% | top10%% | inp%%  | r2_avg | r2_p95\n",
                    static_cast<int>(axis_name.size()), axis_name.data());
        std::printf(" -----+----------+-----+-------+--------+-------+--------+--------\n");
    }

    void PrintRow(const Row& row) const
    {
        // Trailing `*` flags linearity collapse (r2_p95 > 0.99). Width-stable
        // (1 char) so the column doesn't jitter across rows.
        std::printf("%5s | %.2e | %3zu | %5.1f | %5.1f  | %5.1f | %6.3f | %5.2f%c\n",
                    row.label.c_str(),
                    row.result.top_eigenvalue,
                    row.result.effective_rank,
                    row.result.top1_cum_pct,
                    row.result.top10_cum_pct,
                    row.result.input_correlated_pct,
                    row.result.mean_r2,
                    row.result.r2_p95,
                    row.result.linearity_collapsed ? '*' : ' ');
        std::fflush(stdout);
    }

    void PrintFilterReport(const Selection& sel) const
    {
        std::printf("\nFilter report on %zu rows:\n"
                    "  %3zu magnitude-collapsed (top_eig <  %.2e)\n"
                    "  %3zu linearity-collapsed (input_corr >= %.1f%%)\n"
                    "  %3zu dead-recurrent (input_corr <  %.1f%%)\n"
                    "  %3zu healthy\n",
                    sel.total_input,
                    sel.magnitude_collapsed,
                    static_cast<double>(opt_.healthy_top_eig),
                    sel.linearity_collapsed,
                    static_cast<double>(opt_.linearity_collapse_input_corr_pct),
                    sel.dead_recurrent_collapsed,
                    static_cast<double>(opt_.dead_recurrent_input_corr_pct),
                    sel.healthy);
        std::fflush(stdout);
    }

    void PrintRankedTable(const std::vector<Row>& top) const
    {
        if (top.empty())
        {
            std::printf("\n(no rows passed the healthy filter — nothing to rank)\n");
            std::fflush(stdout);
            return;
        }
        std::printf("\nTop %zu (rank desc, inp%% asc, r2_avg asc, "
                    "top1%% asc, top_eig desc, label asc):\n", top.size());
        std::printf(" %5s | top_eig  | eff | top1%% | top10%% | inp%%  | r2_avg\n", "label");
        std::printf(" -----+----------+-----+-------+--------+-------+--------\n");
        for (const auto& row : top)
        {
            std::printf("%5s | %.2e | %3zu | %5.1f | %5.1f  | %5.1f | %6.3f\n",
                        row.label.c_str(),
                        row.result.top_eigenvalue,
                        row.result.effective_rank,
                        row.result.top1_cum_pct,
                        row.result.top10_cum_pct,
                        row.result.input_correlated_pct,
                        row.result.mean_r2);
        }
        std::fflush(stdout);
    }

    // Strict weak ordering: rank desc → inp asc → r2 asc → top1 asc →
    // top_eig desc → label asc. The final `label asc` tier makes the
    // sort deterministic across reruns.
    static bool LessBalanced(const Row& a, const Row& b)
    {
        if (a.result.effective_rank != b.result.effective_rank)
            return a.result.effective_rank > b.result.effective_rank;
        if (a.result.input_correlated_pct != b.result.input_correlated_pct)
            return a.result.input_correlated_pct < b.result.input_correlated_pct;
        if (a.result.mean_r2 != b.result.mean_r2)
            return a.result.mean_r2 < b.result.mean_r2;
        if (a.result.top1_cum_pct != b.result.top1_cum_pct)
            return a.result.top1_cum_pct < b.result.top1_cum_pct;
        if (a.result.top_eigenvalue != b.result.top_eigenvalue)
            return a.result.top_eigenvalue > b.result.top_eigenvalue;
        return a.label < b.label;
    }
};
