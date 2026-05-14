#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>


/// @brief Per-level dynamics knobs for one depth slice of the fractal.
///
/// Every sub-reservoir at the same depth shares one LevelConfig; different
/// depths can hold different values so outer (slow-integrating, long-memory)
/// and inner (fast-mixing) levels can be tuned independently. The fields are
/// the four standard ESN dynamics knobs — top-level `seed` and `num_inputs`
/// stay on `FractalReservoirConfig<DEPTH>` because they are global to the
/// fractal, not per-level.
struct FractalReservoirLevelConfig
{
    float alpha = 1.0f;
    float spectral_radius = 0.97f;
    float leak_rate = 0.1f; // 1.0 = full replacement, <1.0 = leaky integrator
    float input_scaling = 0.15f;
};


/// @brief Top-level config for FractalReservoir<DEPTH>.
///
/// `seed` seeds the whole-fractal RNG (one stream walks all sub-reservoirs
/// in level-major order during initialization). `num_inputs` controls
/// depth-0 input striping and must be in {1, 2, 4, 8}; deeper sub-reservoirs
/// have no external-input notion and instead receive 8 scalars per timestep
/// (one per vertex) from the parent's 8-deep input queue. Per-depth dynamics
/// live in `levels[d]` for d in [0, DEPTH).
template <size_t DEPTH>
struct FractalReservoirConfig
{
    static_assert(DEPTH >= 1 && DEPTH <= 5,
                  "FractalReservoirConfig DEPTH must be in [1, 5]");
    // Default seed predates the current input-queue design; for tuned
    // per-DEPTH seeds use `FractalReservoirConfigDefaults::SeedFor`.
    uint64_t seed = 193;
    size_t num_inputs = 1; // must be in {1, 2, 4, 8}
    std::array<FractalReservoirLevelConfig, DEPTH> levels{};
};


/// @brief Build a FractalReservoirConfig<DEPTH> that broadcasts one
///        LevelConfig across every depth.
///
/// Convenience for callers that want pre-per-level uniform configs (tests,
/// smoke runs, ad-hoc benchmarks). The materialized config can then be
/// mutated per-level by the caller before constructing the reservoir.
template <size_t DEPTH>
[[nodiscard]] inline FractalReservoirConfig<DEPTH>
MakeUniformConfig(uint64_t seed, size_t num_inputs,
                  FractalReservoirLevelConfig level = {})
{
    FractalReservoirConfig<DEPTH> cfg{};
    cfg.seed = seed;
    cfg.num_inputs = num_inputs;
    for (auto& l : cfg.levels) l = level;
    return cfg;
}


/// @brief Per-DEPTH tuned defaults for FractalReservoirConfig.
///
/// Two lookups are exposed:
///
///   - `SeedFor(K)`  — the validated seed for a standalone DEPTH=K fractal,
///                     pulled from StateRank balanced/autonomy surveys.
///   - `LevelFor(K)` — the dynamics knobs (alpha, SR, leak, input_scaling)
///                     that performed best for a uniform-config DEPTH=K fractal.
///
/// `For<DEPTH>()` composes these into a per-level config: level `d`
/// (0-indexed) of a DEPTH=K fractal uses `LevelFor(d + 1)`. This treats
/// the outermost level as if it were a standalone DEPTH=1 cube (wet/fast),
/// the innermost leaf as if it were a standalone DEPTH=K leaf (tight/slow),
/// and each intermediate level as if it were the leaf of a DEPTH=(d+1)
/// fractal. The whole-fractal seed comes from `SeedFor(DEPTH)`.
///
/// Per-level surveys may eventually displace this composition; for now it
/// extracts the most data we have from the existing per-K tunings instead
/// of broadcasting one value across every level.
struct FractalReservoirConfigDefaults
{
    /// Validated whole-fractal seed for a standalone DEPTH=`depth` fractal.
    [[nodiscard]] static constexpr uint64_t SeedFor(size_t depth)
    {
        // Do not 'trust' these seeds. Find your own.
        switch (depth)
        {
        case 1: return 6328;
        case 2: return 4860;
        case 3: return 2591;
        case 4: return 2261;
        case 5: return 2934;
        default: return 0;
        }
    }

    /// Validated dynamics for level `depth` (1-indexed) when embedded in an
    /// overall DEPTH=`max_depth` fractal. Lower-triangular: callers must
    /// pass `1 <= depth <= max_depth`; out-of-table cells return `{}`.
    ///
    /// Today every column at `depth=k` holds the same tuple across all
    /// `max_depth >= k` rows, so the function is numerically a 1D lookup
    /// on `depth`. The 2D signature reserves the freedom to tune any
    /// `(max_depth, depth)` cell independently in the future without
    /// breaking callers — e.g. if optimal L0 dynamics turn out to differ
    /// when L0 is the outermost level of DEPTH=3 vs DEPTH=5.
    [[nodiscard]] static constexpr FractalReservoirLevelConfig LevelFor(size_t max_depth, size_t depth)
    {
        switch (max_depth)
        {
        case 1:
            return {1.0f, 0.90f, 0.1f, 0.05f};
        case 2:
            switch (depth)
            {
            case 1: return {1.0f, 0.9f, 0.15f, 0.05f};
            case 2: return {1.0f, 0.9f, 0.15f, 0.05f};
            default: return {};
            }
        case 3:
            switch (depth)
            {
            case 1: return {1.0f, 0.92f, 0.15f, 0.05f};
            case 2: return {1.0f, 0.93f, 0.15f, 0.05f};
            case 3: return {1.05f, 0.95f, 0.20f, 0.05f};
            default: return {};
            }
        case 4:
            switch (depth)
            {
            case 1: return {1.0f, 0.92f, 0.15f, 0.05f};
            case 2: return {1.0f, 0.93f, 0.15f, 0.05f};
            case 3: return {1.0f, 0.94f, 0.15f, 0.05f};
            case 4: return {1.0f, 0.95f, 0.15f, 0.05f};
            default: return {};
            }
        case 5:
            switch (depth)
            {
            case 1: return {1.0f, 0.92f, 0.15f, 0.05f};
            case 2: return {1.0f, 0.93f, 0.15f, 0.05f};
            case 3: return {1.0f, 0.94f, 0.15f, 0.05f};
            case 4: return {1.0f, 0.95f, 0.15f, 0.05f};
            case 5: return {1.0f, 0.96f, 0.15f, 0.05f};
            default: return {};
            }
        default: return {};
        }
    }


    template <size_t DEPTH>
    [[nodiscard]] static FractalReservoirConfig<DEPTH> For()
    {
        static_assert(DEPTH >= 1 && DEPTH <= 5,
                      "FractalReservoirConfigDefaults::For: DEPTH must be in [1, 5]");

        FractalReservoirConfig<DEPTH> cfg{};
        cfg.seed = SeedFor(DEPTH);
        cfg.num_inputs = 1;
        // Level d in a DEPTH=K fractal is treated as a standalone DEPTH=(d+1)
        // leaf — outer level = DEPTH=1 (wet/fast), innermost = DEPTH=K (tight/slow).
        for (size_t d = 0; d < DEPTH; ++d)
            cfg.levels[d] = LevelFor(DEPTH, d + 1);
        return cfg;
    }
};


/// @brief A hierarchical echo-state machine where each neuron is itself a full hypercube ESN.
///
/// **Concept.** A standard hypercube reservoir (see HypercubeRC::Reservoir)
/// places a scalar tanh-neuron at every vertex of a Boolean DIM-cube and
/// evolves them via weighted sums of neighbor activations. FractalReservoir
/// recursively replaces each scalar neuron with another reservoir, down
/// to a configurable DEPTH. The activation a parent vertex contributes to
/// its neighbors is no longer tanh(alpha*s) directly, but the scalar readout
/// of a child sub-reservoir. The downward handoff is an 8-deep delay-line
/// input queue per non-leaf vertex (see "Per-step dynamics"): the parent's
/// own pre-activation s is shifted into slot 0 each step, and the 8 slots
/// fan out across the 8 child vertices via identity mapping, so each child
/// vertex sees a different lag (0..7) of its parent's recent activity. At
/// the bottom of the recursion (depth = DEPTH-1) the child reverts to a
/// plain tanh(alpha*s). The motivation is to extend the effective memory
/// horizon of the hypercube reservoir by giving every "neuron" its own
/// internal state and a multi-lag view of the level above, while keeping
/// the public interface vector-in / state-out.
///
/// **Topology (identical at every level).** DIM=3, N=8 vertices per
/// sub-reservoir. Each vertex has NUM_CONNECTIONS = DIM = 3 incoming
/// recurrent edges, all at Hamming distance 1 (masks 0b001, 0b010,
/// 0b100, via NearestMask(0..2)). The cube under nearest-neighbor-only
/// edges is bipartite (every Hamming-1 flip toggles parity), which
/// constrains how aggressively a level's `spectral_radius` can be pushed
/// before the antisymmetric (parity-difference) modes dominate the
/// `tanh(sum(state))` readout; per-level `leak_rate` is the dominant knob
/// for taming this (low leak integrates the parity oscillation away).
/// Neighbor addresses are computed inline as (v XOR mask); no
/// adjacency storage is needed.
///
/// **Recursion structure.** At depth d in [0, DEPTH) there are N^d
/// sub-reservoirs, each owning N contiguous vertex states. Total:
/// (N^DEPTH - 1) / (N - 1) sub-reservoirs and VTX_SIZE = N + N^2 + ... +
/// N^DEPTH total states. Layout is level-major (depth 0 first, then
/// depth 1, ...); LEVEL_OFFSET[d] gives the starting offset. The
/// sub-reservoir at (depth d, path index p) owns vtx_output_ slots
/// [LEVEL_OFFSET[d] + p*N, LEVEL_OFFSET[d] + (p+1)*N). A child's path
/// index is p_parent*N + v_parent -- the parent's vertex index appended
/// to the parent's own path -- which makes every node in the recursion
/// tree uniquely addressable from its call site. Recursion is bounded
/// by DEPTH <= 5.
///
/// **Per-step dynamics.** UpdateState(inputs) drives the depth-0 reservoir
/// once with K = num_inputs scalar channels (K in {1, 2, 4, 8}). The K
/// channels feed an N=8-slot external-input queue partitioned into K
/// sub-queues of length sub_len = N / K (8, 4, 2, 1 for K = 1, 2, 4, 8).
/// Each step, before the v-loop, each channel's sub-queue is shifted right
/// by one and the new input is written at slot 0. The 8 depth-0 vertices
/// then read from this queue using a BLOCK mapping: channel = v / sub_len
/// (high bits of v), lag = v % sub_len (low bits). Contiguous vertex
/// indices share a channel; lag varies fastest. At K=1 a single 8-deep
/// queue fans across all 8 vertices (vertex v reads lag v); at K=8 every
/// sub-queue is 1 slot deep so there is no delay (vertex v reads
/// inputs[v]). Inside a sub-reservoir at depth d, for each vertex v:
///   1. Pre-activation:               s = x_v * input_scaling[d]
///                                        + sum over neighbors of
///                                          state[v XOR mask] * w[mask]
///                                    where x_v at depth 0 = the
///                                    external-input queue cell for vertex v
///                                    under the block mapping above, and
///                                    x_v at depth >= 1 = parent's queued s
///                                    lagged by v timesteps (see below).
///   2. Queue update (non-leaf only): shift this vertex's 8-deep input
///                                    queue right by one slot and write s
///                                    into slot 0 before recursing. The
///                                    child sub-reservoir's 8 vertices read
///                                    slots 0..7 of this queue (identity
///                                    slot mapping; slot 0 is newest).
///   3. Activation:
///        leaf  (depth = DEPTH-1):    a = tanh(alpha[d] * s)
///        inner (depth <  DEPTH-1):   a = child(p*N + v).update(my_queue)
///   4. Leaky integrate, written to a side buffer:
///                                    new_state[v]
///                                      = (1 - leak[d]) * state[v]
///                                      + leak[d]       * a
/// All N vertices read the pre-update state; the side buffer is committed
/// in-place at the end of the call (synchronous / Jacobi-style update).
/// At depth >= 1 the scalar returned to the parent is tanh(sum(state)) --
/// the sub-reservoir's readout, which becomes the activation a that the
/// parent uses for its own vertex. The depth-0 (public) entry returns
/// nothing; consumers read the full state via Outputs() / Leaves(). The
/// queue gives every child vertex of a parent a different lag into the
/// parent's history (0..7), turning the former scalar parent->child
/// handoff into an 8-deep delay-line fan-out.
///
/// **Cost.** A top-level call updates every sub-reservoir exactly once,
/// processing N + N^2 + ... + N^DEPTH ~ N^DEPTH vertex updates per call.
/// Cost grows geometrically in DEPTH; this is the knob that buys longer
/// effective memory. Per-DEPTH scaling (DIM=3, N=8):
///
///     DEPTH | sub-res | state_size | leaf_count | queue_size | readout dim
///     ------+---------+------------+------------+------------+------------
///         1 |       1 |          8 |          8 |          0 |          3
///         2 |       9 |         72 |         64 |         64 |          6
///         3 |      73 |        584 |        512 |        576 |          9
///         4 |     585 |       4680 |       4096 |       4672 |         12
///         5 |    4681 |      37448 |      32768 |      37440 |         15
///
/// where `sub-res = TOTAL_SUB_RES`, `state_size = VTX_SIZE` (also the
/// per-step vertex-update count), `leaf_count = 8^DEPTH` (the width of
/// `Leaves()` and the ESN<DEPTH> readout input), `queue_size =
/// INPUT_QUEUE_FLOATS` (the per-non-leaf-vertex input queue buffer; N
/// slots each), and `readout dim = 3*DEPTH` is the hypercube dimension the
/// HCNN sees. Leaves account for ~88% of the total vertex count. An
/// additional fixed-size N=8-float external-input queue (independent of
/// DEPTH and K) buffers the depth-0 external inputs; see
/// "Per-step dynamics" above.
///
/// **Spectral radius control.** After random initialization, the
/// recurrent weights of every sub-reservoir at depth d are rescaled so
/// that the power-iteration estimate of the recurrent matrix's largest
/// singular value matches `cfg.levels[d].spectral_radius`. There are no
/// per-vertex random input weights; `cfg.levels[d].input_scaling` is
/// applied directly as a per-depth scalar multiplicand to each vertex's
/// input (external at depth 0, parent's queued s at depth >= 1).
///
/// **Quick intuition.** `spectral_radius` is the gain of the recurrent loop
/// -- how strongly the reservoir amplifies its own past. `leak_rate` is how
/// fast it forgets / integrates new input (discrete time constant ~
/// 1/leak_rate steps). Together they set how long and how richly it remembers.
///
/// **Usage.** Construct via Create(cfg), then call
/// UpdateState(span<const float>) per timestep with cfg.num_inputs
/// channels. Read state via Outputs() (full level-major buffer) or
/// Leaves() (deepest level only).
///
/// **Hypothesis (not borne out).** For the same total parameter count, the
/// fractal reservoir was expected to show substantially longer effective
/// memory than a flat ESN. It did not -- see README.md for the retrospective.
template <size_t DEPTH>
class FractalReservoir
{
    static_assert(DEPTH >= 1 && DEPTH <= 5, "DEPTH must be in the range [1,5]");

    static constexpr size_t DIM = 3; // By design, only DIM = 3 is supported.
    static constexpr size_t N = 1ULL << DIM;

    static constexpr size_t ComputeVtxSize()
    {
        size_t m = N, s = 0;
        for (size_t i = 0; i < DEPTH; ++i)
        {
            s += m;
            m *= N;
        }
        return s;
    }

    static constexpr size_t VTX_SIZE = ComputeVtxSize();

    static constexpr size_t ComputeTotalSubRes()
    {
        size_t sum = 0, pow = 1;
        for (size_t i = 0; i < DEPTH; ++i)
        {
            sum += pow;
            pow *= N;
        }
        return sum;
    }

    static constexpr size_t TOTAL_SUB_RES = ComputeTotalSubRes();

    // Level-major layout: states for depth 0 first, then depth 1, ... up to depth DEPTH-1.
    // At depth d there are N^d sub-reservoirs, each with N contiguous vertex states.
    static constexpr std::array<size_t, DEPTH> ComputeLevelOffsets()
    {
        std::array<size_t, DEPTH> a{};
        size_t off = 0, pow = N;
        for (size_t d = 0; d < DEPTH; ++d)
        {
            a[d] = off;
            off += pow;
            pow *= N;
        }
        return a;
    }

public:
    static constexpr std::array<size_t, DEPTH> LEVEL_OFFSET = ComputeLevelOffsets();
    static constexpr size_t dim = DIM;
    static constexpr size_t state_size = VTX_SIZE;
    /// Number of deepest-level (leaf) states per fractal: 8^DEPTH = 2^(3*DEPTH).
    static constexpr size_t leaf_count = 1ULL << (3 * DEPTH);
    static constexpr uint32_t NearestMask(size_t i) { return 1u << i; }

    [[nodiscard]] static std::unique_ptr<FractalReservoir> Create(const FractalReservoirConfig<DEPTH>& cfg)
    {
        return std::unique_ptr<FractalReservoir>(new FractalReservoir(cfg));
    }

    FractalReservoir(const FractalReservoir&) = delete;
    FractalReservoir& operator=(const FractalReservoir&) = delete;
    FractalReservoir(FractalReservoir&&) = delete;
    FractalReservoir& operator=(FractalReservoir&&) = delete;

    /// Drive the reservoir for one timestep with `inputs.size() == num_inputs`
    /// channels (K = num_inputs). Each channel feeds a depth-0 delay-line
    /// sub-queue; channel c is read by the contiguous depth-0 vertex block
    /// [c*(N/K), (c+1)*(N/K)), lag increasing across the block (block mapping
    /// -- see "Per-step dynamics"). The fractal exposes its state through
    /// `Outputs()` / `Leaves()`; consumers (e.g. the HCNN readout) operate on
    /// leaf states rather than any single scalar.
    void UpdateState(std::span<const float> inputs);

    [[nodiscard]] size_t NumInputs() const { return num_inputs_; }

    /// Overwrite the full level-major state buffer with `state_in`
    /// (must point to state_size floats). Pairs with Outputs() to
    /// snapshot/restore the entire fractal hierarchy.
    void RestoreState(const float* state_in) { std::memcpy(vtx_output_, state_in, sizeof(vtx_output_)); }

    /// Reset all vertex states to the randomized initial values saved during
    /// construction (instead of zero). Weights and hyperparameters are preserved.
    /// Also zeros the per-non-leaf-vertex input queue buffer and the depth-0
    /// external-input queue (both repopulate from upstream pre-activations or
    /// external inputs within a few warmup steps).
    void Reset()
    {
        std::memcpy(vtx_output_, vtx_initial_, sizeof(vtx_output_));
        std::memset(input_queue_, 0, sizeof(input_queue_));
        std::memset(external_input_queue_, 0, sizeof(external_input_queue_));
    }

    /// Read-only view of the full level-major state buffer. Length is
    /// state_size = N + N^2 + ... + N^DEPTH; consecutive depth-d slices
    /// of N states each start at LEVEL_OFFSET[d]. Pairs with RestoreState
    /// for snapshot/restore of the entire fractal hierarchy.
    [[nodiscard]] const float* Outputs() const { return vtx_output_; }

    /// Read-only view of this fractal's leaves only — the deepest depth's
    /// 8^DEPTH contiguous states. Pairs with leaf_count.
    [[nodiscard]] const float* Leaves() const { return vtx_output_ + LEVEL_OFFSET[DEPTH - 1]; }

public:
    /// Read-only view of the full per-non-leaf-vertex 8-deep input queue
    /// buffer. Each non-leaf vertex (depths 0..DEPTH-2) owns N=8 contiguous
    /// queue slots; queue for vertex at (depth d, path p, vertex v) starts at
    /// `(LEVEL_OFFSET[d] + p*N + v) * N` for d < DEPTH-1. Slot 0 = newest.
    /// For DEPTH=1 this view is empty (no children, no queues).
    [[nodiscard]] const float* InputQueues() const { return input_queue_; }

    /// Number of floats in `InputQueues()`. Zero for DEPTH=1.
    static constexpr size_t INPUT_QUEUE_FLOATS =
        (DEPTH > 1) ? LEVEL_OFFSET[DEPTH - 1] * N : 0;

    /// Read-only view of the depth-0 external-input delay-line queue.
    /// Length is always N=8 floats, logically partitioned into K=num_inputs
    /// sub-queues of N/K floats each, packed channel-major. Channel c
    /// occupies slots [c*(N/K), (c+1)*(N/K)); slot 0 of each sub-queue is
    /// newest. Vertex v reads cell `(v / (N/K)) * (N/K) + (v % (N/K))`,
    /// i.e. channel = v/(N/K), lag = v%(N/K) (block convention).
    [[nodiscard]] const float* ExternalInputQueue() const { return external_input_queue_; }

private:
    explicit FractalReservoir(const FractalReservoirConfig<DEPTH>& cfg);
    void Initialize();
    float UpdateRecursive(size_t depth, size_t path_idx, const float* parent_queue);
    [[nodiscard]] float EstimateSpectralRadius(const float* weights) const;

    alignas(64) float vtx_output_[VTX_SIZE] = {};
    alignas(64) float vtx_initial_[VTX_SIZE] = {};

    alignas(64) float vtx_weight_[TOTAL_SUB_RES * N * DIM] = {};
    // Per-non-leaf-vertex 8-deep delay-line queue holding the parent's own
    // pre-activation `s` over the last N=8 timesteps. The 8 slots fan out
    // to the 8 vertices of the child sub-reservoir (identity map: slot v ->
    // child vertex v). DEPTH=1 needs no queue; size-1 dummy keeps the array
    // well-formed under the language rules.
    alignas(64) float input_queue_[(INPUT_QUEUE_FLOATS > 0) ? INPUT_QUEUE_FLOATS : 1] = {};

    // Depth-0 external-input delay line. Logically K=num_inputs_ sub-queues
    // of N/K floats each, packed channel-major: channel c occupies slots
    // [c*(N/K), (c+1)*(N/K)). Slot 0 of each sub-queue is newest. Each
    // step, every channel's sub-queue is shifted right by one and the new
    // input is written at slot 0; then vertex v reads cell
    // (v / (N/K)) * (N/K) + (v % (N/K)) (block mapping). At K=8 each
    // sub-queue is 1 slot deep so the buffer is the input vector itself
    // (no delay).
    alignas(64) float external_input_queue_[N] = {};

    uint64_t rng_seed_ = 0;
    size_t num_inputs_ = 1;
    std::array<float, DEPTH> alpha_{};
    std::array<float, DEPTH> spectral_radius_{};
    std::array<float, DEPTH> leak_rate_{};
    std::array<float, DEPTH> input_scaling_{};
};
