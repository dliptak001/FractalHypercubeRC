# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Status: archived — project sunset 2026-05-14.** This is a negative-result repository: the recursive design did not overcome the shallow-memory limitation it targeted. See `README.md` for the full retrospective. This file is kept reconciled with the final code for anyone reading or building it; no further work is planned.

## Project Overview

FractalHypercubeRC is a C++23 reservoir-computing library implementing a recursive (self-similar) echo-state reservoir on a 3-dimensional Boolean hypercube. A standard hypercube reservoir places a scalar `tanh` neuron at every vertex of a Boolean cube; FractalHypercubeRC recursively replaces each scalar neuron with another reservoir of identical topology, down to a configurable DEPTH (1..5). Parent and child exchange information **bidirectionally**: each non-leaf vertex feeds an 8-deep delay-line queue *downward* to its child sub-reservoir's 8 vertices (one lag each), and each parent vertex's recurrent coupling reads its neighbors' child sub-reservoir's *full 8-state vector* upward. There is no scalar handoff up the recursion — every vertex activates locally and information flows up implicitly through the shared level-major state buffer. The motivation was to extend the effective memory horizon of HypercubeRC by giving every "neuron" its own persistent state. The public interface is vector-in / state-out: the depth-0 cube accepts `cfg.num_inputs ∈ {1, 2, 4, 8}` channels striped across its 8 vertices; consumers read state via `Outputs()` / `Leaves()`.

`FractalReservoir<DEPTH>` is the reservoir core. `ESN<DEPTH>` is the public-facing wrapper that pairs it with an HCNN-based `Readout` that consumes only the deepest-level (leaf) states.

## Build

This is a CLion CMake/Ninja/MinGW project. cmake and g++ are **not** on the system PATH — they are bundled with CLion. Never reconfigure `cmake-build-*` directories (no `cmake -B` with `-G` flags); CLion owns those.

**Build (Release):**
```bash
powershell.exe -File - <<'PS1'
$cmake = 'C:\Program Files\JetBrains\CLion 2024.3.2\bin\cmake\win\x64\bin\cmake.exe'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
& $cmake --build C:\CLion\FractalHypercubeRC\cmake-build-release 2>&1
PS1
```

Replace `cmake-build-release` with `cmake-build-debug` for Debug builds. Prefer Release for tests and diagnostics (Debug has different float behavior with `-ffast-math`).

**Run:** (`FractalHypercubeRC.exe` is the diagnostic driver; `BasicPrediction.exe` / `StreamingAnomaly.exe` / `SignalClassification.exe` are the usage examples). The `$env:PATH` line is required because the binaries depend on MinGW runtime DLLs (notably `libgomp-1.dll` for OpenMP).
```bash
powershell.exe -File - <<'PS1'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
& "C:\CLion\FractalHypercubeRC\cmake-build-release\FractalHypercubeRC.exe" 2>&1
PS1
```

**Critical:** Never use bare bash to invoke g++ or cmake — compiler errors are silently swallowed. Always use the PowerShell heredoc pattern above.

**Recovery:** If a `cmake-build-*` directory is missing or in a broken state, delete it and reload CMake from CLion (`File → Reload CMake Project`) — do not regenerate it manually with `cmake -B` / `-G`, since CLion owns the generator choice.

## Architecture

### Hypercube topology (fixed at DIM=3)

Each sub-reservoir is a length-N=8 cube; the template only varies DEPTH. Per vertex, `NUM_CONNECTIONS = DIM = 3` incoming recurrent edges, all **nearest neighbors** at Hamming distance 1 — masks 0b001, 0b010, 0b100 via `NearestMask(0..2)`. The cube under nearest-neighbor-only edges is bipartite (every Hamming-1 flip toggles parity), which constrains how aggressively a level's `spectral_radius` can be pushed before antisymmetric (parity-difference) modes dominate the dynamics; `leak_rate` is the dominant knob for taming this.

Neighbor addresses are computed inline as `v ^ mask`; no adjacency storage. Recurrent weights have **two layouts** (see Per-step update): non-leaf sub-reservoirs use a `(N, DIM, N)` vector-coupling tensor (`NON_LEAF_WEIGHTS_PER_SUB = 192` floats), leaf sub-reservoirs use a `(N, DIM)` scalar-coupling matrix (`LEAF_WEIGHTS_PER_SUB = 24` floats). `vtx_weight_` packs all non-leaf subs first, then all leaves; per-sub offsets come from `WeightOffset(depth, path)`.

### Fractal recursion

`FractalReservoir<DEPTH>` is templated on recursion depth (1..5). At depth `d in [0, DEPTH)` there are `N^d` sub-reservoirs, each owning N contiguous vertex states. Total states `VTX_SIZE = N + N^2 + ... + N^DEPTH` live in `vtx_output_` in level-major layout (depth 0 first, then depth 1, ...); `LEVEL_OFFSET[d]` gives the starting offset. The sub-reservoir at (depth d, path index p) owns `vtx_output_[LEVEL_OFFSET[d] + p*N .. (p+1)*N)`. A child's path index is `p_parent*N + v_parent` — the parent's vertex index appended to the parent's own path. `WeightOffset(depth, path)` maps a sub-reservoir to its slice in `vtx_weight_`.

`vtx_weight_` has an **independent slice** per sub-reservoir (each sub gets its own recurrent weight tensor — non-leaf or leaf layout per above). There are **no** per-vertex input weights; each level applies a single scalar `input_scaling`. Hyperparameters are **per-depth** (`cfg.levels[d]`), not shared across the whole fractal.

### Per-step update

Public entry `void UpdateState(std::span<const float> inputs)` validates `inputs.size() == cfg.num_inputs` (throws `std::invalid_argument` on mismatch) and runs the depth-0 update inline. Recursion into deeper sub-reservoirs is via `UpdateRecursive(depth, path_idx, parent_queue)` for `depth ≥ 1` — which returns **void** (severed-cascade design; see below). Inside a sub-reservoir, for each vertex v (synchronous / Jacobi-style update):

1. Pre-activation:
   - Leaf (`depth == DEPTH-1`): scalar sibling coupling — `s = x_v * input_scaling[d] + sum over 3 neighbors of state[v ^ mask] * w[v,i]`.
   - Non-leaf (`depth < DEPTH-1`): vector child coupling — `s = x_v * input_scaling[d] + sum over 3 neighbors of dot(child_state[v ^ mask], W[v,i,:])`, where `child_state[u]` is the 8-state vector of the depth-(d+1) sub-reservoir rooted at parent vertex u. The child chunk is snapshotted before the v-loop so every parent vertex sees pre-step children.
   - `x_v` at depth 0 is read from the N=8-slot external-input delay-line queue (`external_input_queue_`), partitioned into K=`num_inputs` sub-queues of length `sub_len = N/K`. Each step the K channels are shifted into their sub-queues (newest at slot 0), then vertex `v` reads cell `(v / sub_len) * sub_len + (v % sub_len)` — channel = `v / sub_len`, lag = `v % sub_len` (block convention; contiguous vertices share a channel). K=8 degenerates to no delay (`sub_len=1`).
   - `x_v` at depth ≥ 1 is `parent_queue[v]` — slot v of the parent vertex's 8-deep delay-line queue, i.e. the parent's pre-activation `s` lagged by v timesteps.
2. Queue update (non-leaf only): shift this vertex's 8-deep `input_queue_` slot right by one and write `s` into slot 0, then recurse via `UpdateRecursive(depth+1, path_idx*N + v, my_queue)` (depth-0's call is `UpdateRecursive(1, v, my_queue)`).
3. Activation: `a = tanh(alpha[d] * s)` — local at every depth, leaf and non-leaf alike. There is **no** scalar value inherited from the recursive call.
4. Leaky integrate into a side buffer: `new_state[v] = (1 - leak_rate[d])*state[v] + leak_rate[d]*a`.

`state` is read-only during the v-loop; `new_state` is `memcpy`'d back at the end (Jacobi-style commit). `UpdateRecursive` returns void at every depth — information flows upward implicitly: the grandparent's next-step pre-activation reads this sub-reservoir's `state[]` directly through the level-major buffer. The depth-0 public entry returns nothing; consumers read state via `Outputs()` (full level-major buffer) or `Leaves()` (deepest level only).

`Reset()` restores `vtx_output_` to the small randomized initial state saved during construction (in `vtx_initial_`), not zero, and zeros both the per-non-leaf-vertex input queue buffer (`input_queue_`) and the depth-0 external-input queue; weights and hyperparameters are preserved (no expensive spectral-radius rescaling on episode boundaries).

### Cost

A top-level `UpdateState` call updates every sub-reservoir exactly once: `N + N^2 + ... + N^DEPTH ~ N^DEPTH` vertex updates. Cost grows geometrically in DEPTH; this is the knob that buys longer effective memory.

### Spectral radius control

Two regimes after random weight initialization:
- **Leaf depth:** the recurrent matrix is square `N x N`. `EstimateSpectralRadius()` runs power iteration (up to 100 iterations with relative-tolerance early termination) to estimate its largest singular value, and `Initialize()` rescales the leaf weight slice so the estimate matches `cfg.levels[DEPTH-1].spectral_radius`.
- **Non-leaf depths:** the recurrent operator is rectangular (`R^{N*N} -> R^N`). Power-iterating it is overkill for a tunable knob, so weights are scaled analytically by `spectral_radius / sqrt(DIM * N)` after a uniform `[-1, 1]` draw. `spectral_radius` stays a 0..1ish dial but is not literally the operator norm.

Per-depth dynamics live in `cfg.levels[d]` (`alpha`, `spectral_radius`, `leak_rate`, `input_scaling`) — outer (slow-integrating, long-memory) and inner (fast-mixing) levels can be tuned independently. `FractalReservoirConfigDefaults::For<DEPTH>()` composes per-DEPTH tuned seeds (`SeedFor`) and per-level dynamics (`LevelFor`) into a ready-to-use config; `MakeUniformConfig<DEPTH>()` broadcasts one `LevelConfig` across every depth for tests and ad-hoc runs.

### ESN and Readout

`ESN<DEPTH>` (in `ESN.h/cpp`) wraps a single `FractalReservoir<DEPTH>` and a `Readout`. It accepts the same `cfg.num_inputs` channels per timestep (forwarded to the reservoir) and exposes warmup / run / batch-train / online-train / predict phases. The readout consumes the fractal's leaf states only — `8^DEPTH` features per timestep, packed as a `(3*DEPTH)`-dimensional hypercube. Outer and intermediate fractal levels are not exposed.

`Readout` (in `Readout.h/cpp`) is HCNN-based via PIMPL on `hcnn::HCNN`: standardize → Conv+Pool stack (`max(min(EffectiveDIM-2, 2), 1)` pairs by default, requires DIM ≥ 3) → Flatten → Linear → optional de-center for regression with target centering. Input layout is timestep-major / row-major: `Warmup`, `Run`, and `InitOnline` take a flat `num_steps * num_inputs` array.

**Caller-contract preconditions throw, not assert.** Out-of-range timestep indices in `ESN::PredictRaw` / `R2` / `NRMSE` / `Accuracy` throw `std::out_of_range`. State-machine violations in `Readout` (e.g. predicting before training) and the scalar `PredictRaw` overload's `num_outputs == 1` requirement throw `std::logic_error`. `FractalReservoir`'s constructor validates each `cfg.levels[d]` (positive `alpha` / `spectral_radius` / `input_scaling`, `leak_rate ∈ (0, 1]`) and `cfg.num_inputs ∈ {1, 2, 4, 8}`, and `UpdateState` validates the input span size — all throw `std::invalid_argument`. CMake's default Release passes `-DNDEBUG`, which compiles `assert` out — so anything that needs to survive Release must throw. The one remaining `assert` in `Readout::build_architecture` (`layers <= d - 2`) guards an internal post-clamp invariant, not a caller contract.

### Diagnostics

The `diagnostics/` headers are a header-only investigation harness wired together by `main.cpp`; there is no separate test executable.

- `diagnostics/NARMA_N_Benchmark.h` — header-only NARMA-N benchmark with depth-aware total/warmup sizing. `narma_order` is a required ctor argument (no default) — every caller picks N explicitly. Input/target sequences seeded from `cfg.seed + 99`.
- `diagnostics/StateRank.h` — effective-rank / input-linearity measurement of the reservoir's state cloud.
- `diagnostics/StateRankSurvey.h` — hyperparameter sweep built on `StateRank`, with NARMA-N validation of the top candidates.

### Key design decisions

- Factory `Create()` returns `unique_ptr`; copy/move disabled.
- Explicit template instantiations in `FractalReservoir.cpp` for `DEPTH = 1..5`.
- Executables (`FractalHypercubeRC` driver plus the `examples/` programs) are only added when this repo is the top-level CMake project (`if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)`); the `FractalHypercubeRCCore` static library can be consumed by another project (e.g. via FetchContent) without those tagging along.
- State buffers (`vtx_output_`, `vtx_initial_`) and weight arrays are `alignas(64)` for SIMD/cache friendliness on the hot recursive loop.
- Recurrent weights are drawn uniform `[-1, 1]` with low-magnitude rejection (`|w| < 0.05` resampled) so a near-zero draw doesn't silence a signal path. Leaf slices are then scaled by `1/sqrt(DIM)` and rescaled per-sub by power iteration to `cfg.levels[DEPTH-1].spectral_radius`; non-leaf slices are scaled analytically by `spectral_radius / sqrt(DIM*N)` (see Spectral radius control). There are no random input weights — `input_scaling` is applied directly as a per-depth scalar.
- Weights use `float`; RNG draws `double` then casts.
