# FractalHypercubeRC

> **⚠️ Archived — project sunset 2026-05-14. Negative result.**
>
> This repository is published as a **cautionary reference**, not a usable
> library. The central hypothesis did not hold: a recursive, self-similar
> echo-state reservoir does **not** overcome the shallow-memory limitation of
> the flat hypercube reservoir it was built to extend.
>
> Stacking sub-reservoirs deepens each "neuron's" internal state, but it does
> not lengthen the *effective memory horizon* enough to justify the geometric
> blow-up in compute and parameters with `DEPTH`. Several redesigns of the
> inter-level coupling (scalar handoff → severed upward cascade → bidirectional
> vector exchange) each failed to close the gap. The final state is tagged
> [`sunset-shallow-memory`](https://github.com/dliptak001/FractalHypercubeRC/releases/tag/sunset-shallow-memory).
>
> **If you are considering a similar architecture:** the bottleneck was never
> the per-neuron state capacity — it was the bipartite, nearest-neighbor
> hypercube topology and the `tanh`-of-mean-style readout dynamics. Adding
> recursive depth multiplies state without changing the spectral structure
> that actually governs memory. See [What we learned](#what-we-learned) below.
>
> The code is left intact and buildable for reference. No further work is
> planned.

---

## What it is

FractalHypercubeRC is a C++23 reservoir-computing library implementing a
**recursive (self-similar) echo-state reservoir** on a 3-dimensional Boolean
hypercube. A standard hypercube reservoir places a scalar `tanh` neuron at
every vertex of a Boolean cube; FractalHypercubeRC recursively replaces each
scalar neuron with another hypercube reservoir of identical topology, down to a
configurable `DEPTH` (1..5).

The motivation was to extend the effective memory horizon by giving every
"neuron" its own internal state that persists across timesteps, while keeping
the public interface vector-in / scalar-out.

- `FractalReservoir<DEPTH>` — the recursive reservoir core.
- `ESN<DEPTH>` — public-facing wrapper pairing the reservoir with an
  HCNN-based learned `Readout` that consumes only the deepest-level (leaf)
  states.

## What we learned

The working hypothesis was: *for the same total parameter count, a fractal
reservoir exhibits substantially longer effective memory than a flat ESN.* It
did not pan out.

- **Recursive depth multiplies state, not memory.** A `DEPTH`-level fractal has
  `8 + 8^2 + ... + 8^DEPTH` internal states, but the memory horizon is governed
  by the spectral structure of the recurrent operator — and every sub-reservoir
  reuses the *same* bipartite, nearest-neighbor cube topology. The smoking gun:
  a `StateRank` survey measured the *effective rank* of the state covariance and
  it ceilings hard. From DEPTH 1 to 5 the state buffer grows `8 → 72 → 584 →
  4,680 → 37,448` (~4,600×), but effective rank only crawls from ~2 to ~28. The
  scalar handoff between sub-reservoirs caps how much rank propagates up the
  hierarchy — depth balloons the state vector without growing the useful
  dimensionality, and useful dimensionality is what bounds memory.
- **The bipartite cube fights you.** Under nearest-neighbor-only edges, every
  Hamming-1 flip toggles parity, so the cube is bipartite. Antisymmetric
  (parity-difference) modes dominate as `spectral_radius` is pushed up, and the
  `tanh(mean(state))`-style readout washes them out. `leak_rate` tames the
  oscillation but at the cost of the very memory you were trying to buy.
- **Coupling redesigns didn't rescue it.** The inter-level interface went
  through three generations — a scalar `tanh(mean(state))` handoff up the
  recursion, then a severed upward cascade (each vertex activates locally), then
  a bidirectional vector exchange (downward delay-line queues + upward
  full-vector child coupling). Each was more expressive than the last; none
  meaningfully extended the memory horizon on NARMA-style benchmarks. NARMA-N
  NRMSE sat at ≈ 0.65 across seeds and depths — and the *same* plateau appeared
  on both an HCNN-on-leaves readout and a ridge regression on the full state
  buffer. That two very different readouts hit the identical wall is what
  pointed the finger at the parent↔child interface rather than the readout.
- **Cost scales the wrong way.** Per-step compute grows as `~8^DEPTH`. By the
  time `DEPTH` is large enough to *maybe* matter, the model is far more
  expensive than a flat ESN with the same — or better — memory.

### Diagnostic traps worth knowing about

Two methodology lessons cost real time here and generalize well beyond this
project:

- **State-space geometry does not predict task performance.** We built a
  `StateRank` diagnostic measuring the effective rank and input-linearity of the
  reservoir's state cloud. A reservoir can score "excellent" on every geometric
  axis — rich, high-rank, non-degenerate state — and still have *zero functional
  alignment* with the task. Good geometry tells you the reservoir is alive; it
  does not tell you it computes anything you need. A diagnostic that measures the
  wrong thing will happily green-light a dead end.
- **High-dimensional readouts turn cross-seed comparison into a luck contest.**
  At DEPTH=5 the readout had 32,768 leaf features. Against a short training
  window, cross-seed NRMSE was dominated by overfit-then-test luck, not reservoir
  quality — *within-seed* hyperparameter trends were real, *cross-seed* rankings
  were noise. If your feature count rivals your sample count, you are
  benchmarking initialization variance, not the architecture.

The earlier, pre-refactor system (templated on `<DIM, DEPTH>` with the *top*
cube dimension varying) once reached NRMSE ≈ 0.122 on NARMA-10 at DIM=10. The
N=8 / DEPTH-scaling redesign documented here never reproduced that, and the
project was sunset before a full re-sweep — the negative trend across redesigns
was clear enough.

## Architecture

> The description below reflects the **final** code state (the
> "severed-cascade / bidirectional vector exchange" design). Note that
> the Doxygen comments may still describe intermediate designs in places —
> `FractalReservoir.h` is the authoritative source.

### Hypercube topology (fixed at DIM=3)

Each sub-reservoir is a length-`N=8` cube; the template only varies `DEPTH`.
Per vertex, `NUM_CONNECTIONS = DIM = 3` incoming recurrent edges, all nearest
neighbors at Hamming distance 1 (masks `0b001`, `0b010`, `0b100`). Neighbor
addresses are computed inline as `v ^ mask`; no adjacency storage. The cube
under nearest-neighbor-only edges is **bipartite**, which constrains how
aggressively `spectral_radius` can be pushed before antisymmetric modes
dominate the readout — `leak_rate` is the dominant taming knob.

### Fractal recursion

At depth `d in [0, DEPTH)` there are `8^d` sub-reservoirs, each owning `N`
contiguous vertex states. Total internal state
`state_size = 8 + 8^2 + ... + 8^DEPTH` lives in `vtx_output_` in level-major
layout (depth 0 first, then depth 1, ...); `LEVEL_OFFSET[d]` gives the start of
level `d`. The sub-reservoir at (depth `d`, path index `p`) owns
`vtx_output_[LEVEL_OFFSET[d] + p*N .. (p+1)*N)`. A child's path index is
`p_parent*N + v_parent`.

### Inter-level coupling (severed-cascade, bidirectional vector exchange)

- **Downward.** Each non-leaf vertex owns an 8-deep delay-line input queue. The
  parent's own pre-activation `s` is shifted into slot 0 each step; the 8 slots
  fan out across the 8 child vertices (identity map), so each child vertex sees
  a different lag (0..7) of its parent's recent `s`.
- **Upward.** Each parent vertex's recurrent neighbor coupling reads its
  neighbor's *child sub-reservoir's full 8-state vector*, not a scalar summary.
  The non-leaf recurrent weight is therefore an `(N, DIM, N)` tensor (192 floats
  per sub-reservoir). At leaf depth there are no children, so cross-coupling
  reverts to classical scalar sibling wiring, an `(N, DIM)` matrix (24 floats).
- **No scalar reduction up the recursion.** Every vertex (leaf or non-leaf)
  computes its activation locally as `tanh(alpha * s)` and leaky-integrates it
  into its own scalar state. Information flows upward implicitly through the
  level-major state buffer that the grandparent's coupling reads.

### Per-step update

Public entry `void UpdateState(std::span<const float> inputs)` validates
`inputs.size() == num_inputs` and drives the depth-0 reservoir once with
`K = num_inputs` channels (`K ∈ {1, 2, 4, 8}`). The K channels feed an
8-slot external-input queue partitioned into K sub-queues of length `N/K`
(block mapping: vertex `v` reads channel `v / (N/K)`, lag `v % (N/K)`).
Recursion into deeper sub-reservoirs is via
`UpdateRecursive(depth, path_idx, parent_queue)`.

Inside a sub-reservoir, for each vertex `v` (synchronous / Jacobi-style update):

1. **Pre-activation.**
   - Leaf (`depth == DEPTH-1`): `s = x_v * input_scaling[d] + Σ state[v ^ mask_i] * w[v,i]`
     (scalar sibling coupling).
   - Inner (`depth < DEPTH-1`): `s = x_v * input_scaling[d] + Σ dot(child_state[v ^ mask_i], W[v,i,:])`
     (vector coupling on the child's 8-state vector).
   - `x_v` at depth 0 is the external-input queue cell; at depth ≥ 1 it is the
     parent's queued `s`, lagged by `v` timesteps.
2. **Queue update** (non-leaf only): shift this vertex's 8-deep queue and write
   `s` into slot 0 before recursing.
3. **Activation:** `a = tanh(alpha[d] * s)` — local at every depth.
4. **Recurse** (non-leaf only): advance the child sub-reservoir.
5. **Leaky integrate** into a side buffer: `new_state[v] = (1 - leak[d]) * state[v] + leak[d] * a`.

`state` is read-only during the v-loop; the side buffer is committed at the end
of the call. Consumers read state via `Outputs()` (full level-major buffer) or
`Leaves()` (deepest level only).

### Configuration

`FractalReservoirConfig<DEPTH>` is templated on `DEPTH` and holds:

- `seed` — whole-fractal RNG seed.
- `num_inputs` — depth-0 input striping, must be in `{1, 2, 4, 8}`.
- `levels[d]` — a `FractalReservoirLevelConfig` *per depth*
  (`alpha`, `spectral_radius`, `leak_rate`, `input_scaling`), so outer
  (slow-integrating, long-memory) and inner (fast-mixing) levels can be tuned
  independently.

`FractalReservoirConfigDefaults::For<DEPTH>()` composes per-DEPTH tuned seeds
(`SeedFor`) and per-level dynamics (`LevelFor`) into a ready-to-use config.
`MakeUniformConfig<DEPTH>(seed, num_inputs, level)` broadcasts one
`LevelConfig` across every depth for tests and ad-hoc runs.

### Spectral radius control

Two regimes after random weight initialization:

- **Leaf depth:** the recurrent matrix is square `N x N`; power iteration
  estimates the largest singular value and weights are rescaled so it matches
  `levels[DEPTH-1].spectral_radius`.
- **Non-leaf depths:** the recurrent operator is rectangular (maps `8N`
  child-state floats into 8 parent pre-activations). Rather than power-iterate a
  rectangular operator for a knob that is already just a tunable dial, weights
  are scaled analytically by `spectral_radius / sqrt(DIM * N)` after a uniform
  `[-1, 1]` draw with low-magnitude rejection.

There are **no** per-vertex random input weights — `input_scaling` is applied
directly as a per-depth scalar multiplicand.

### ESN and Readout

`ESN<DEPTH>` wraps a single `FractalReservoir<DEPTH>` and a `Readout`. It
forwards `num_inputs` channels per timestep and exposes warmup / run /
batch-train / online-train / predict phases. The readout consumes the fractal's
**leaf states only** — `8^DEPTH` features per timestep, packed as a
`(3*DEPTH)`-dimensional hypercube. Outer and intermediate levels are not
exposed.

`Readout` is HCNN-based via PIMPL on `hcnn::HCNN`: standardize → Conv+Pool stack
(`max(min(EffectiveDIM-2, 2), 1)` pairs by default) → Flatten → Linear →
optional de-center for regression with target centering. Input layout is
timestep-major / row-major.

**Caller-contract preconditions throw**, not assert (`std::invalid_argument`,
`std::out_of_range`, `std::logic_error`) — CMake's Release passes `-DNDEBUG`,
which compiles `assert` out, so anything that must survive Release throws.

### Per-DEPTH scaling (DIM=3, N=8)

| DEPTH | sub-reservoirs | state_size | leaf_count | queue_size | readout dim |
|------:|---------------:|-----------:|-----------:|-----------:|------------:|
| 1     |              1 |          8 |          8 |          0 |           3 |
| 2     |              9 |         72 |         64 |         64 |           6 |
| 3     |             73 |        584 |        512 |        576 |           9 |
| 4     |            585 |       4680 |       4096 |       4672 |          12 |
| 5     |           4681 |      37448 |      32768 |      37440 |          15 |

`state_size` is also the per-step vertex-update count; leaves account for ~88%
of it. A top-level `UpdateState` call updates every sub-reservoir exactly once,
so cost grows geometrically (`~8^DEPTH`) in `DEPTH`.

## Diagnostics

The `diagnostics/` headers are the investigation harness — `main.cpp` wires them
together. They are header-only and consumed directly by the driver; there is no
separate test executable.

- `diagnostics/NARMA_N_Benchmark.h` — header-only NARMA-N benchmark + generator
  with depth-aware total/warmup sizing. `narma_order` is a required ctor
  argument.
- `diagnostics/StateRank.h` — effective-rank / input-linearity measurement of
  the reservoir's state cloud.
- `diagnostics/StateRankSurvey.h` — hyperparameter sweep built on `StateRank`,
  with NARMA-N validation of the top candidates.

## Build

This is a CLion CMake/Ninja/MinGW project. `cmake` and `g++` are bundled with
CLion and **not** on the system PATH; never reconfigure the `cmake-build-*`
directories (CLion owns the generator choice).

**Dependency — HypercubeCNN.** The CNN readout links against the sibling
project [HypercubeCNN](https://github.com/dliptak001/HypercubeCNN).
`CMakeLists.txt` expects it as a **sibling directory** and links its prebuilt
static library, so clone it alongside this repo and build it Release-first:

```
parent/
├── FractalHypercubeRC/   (this repo)
└── HypercubeCNN/         (git clone https://github.com/dliptak001/HypercubeCNN.git)
```

Build HypercubeCNN in Release (see its own README) so that
`../HypercubeCNN/cmake-build-release/libHypercubeCNNCore.a` exists before
building FractalHypercubeRC.

**Build (Release):**
```bash
powershell.exe -File - <<'PS1'
$cmake = 'C:\Program Files\JetBrains\CLion 2024.3.2\bin\cmake\win\x64\bin\cmake.exe'
$env:PATH = "C:\Program Files\JetBrains\CLion 2024.3.2\bin\mingw\bin;" + $env:PATH
& $cmake --build C:\CLion\FractalHypercubeRC\cmake-build-release 2>&1
PS1
```

Replace `cmake-build-release` with `cmake-build-debug` for Debug. Prefer
Release for tests and diagnostics (Debug has different float behavior with
`-ffast-math`). The binaries depend on MinGW runtime DLLs (notably
`libgomp-1.dll`), so the `$env:PATH` line is required to run them too.

## Usage (reference only)

```cpp
// Construct a tuned reservoir + ESN at DEPTH=2.
auto cfg = FractalReservoirConfigDefaults::For<2>();
ESN<2> esn(cfg);

// Warmup / Run take a flat span of num_steps * num_inputs floats (timestep-major).
esn.Warmup(std::span<const float>(inputs.data(), warmup_steps * num_inputs));
esn.Run   (std::span<const float>(inputs.data() + warmup_steps * num_inputs,
                                  run_steps * num_inputs));
esn.Train (std::span<const float>(targets.data(), train_size * num_outputs),
           ReadoutConfig{});
double r2 = esn.R2(targets, 0, esn.NumCollected());

// NARMA-N benchmark — narma_order is required (no default).
auto narma_cfg = FractalReservoirConfigDefaults::For<3>();
NARMA_N_Benchmark<3> benchmark(narma_cfg, /*narma_order=*/12, ReadoutConfig{});
auto result = benchmark.Run();  // -> { nrmse, train_time_s }
```

## Project structure

```
FractalHypercubeRC/
├── CMakeLists.txt
├── README.md                       This file
├── main.cpp                        Diagnostic driver (StateRank survey + NARMA-N validation)
├── FractalReservoir.h/.cpp         Recursive hypercube reservoir core (authoritative architecture source)
├── ESN.h/.cpp                      Echo-state network wrapper (reservoir + readout)
├── Readout.h/.cpp                  HCNN-based learned readout (PIMPL on hcnn::HCNN)
├── diagnostics/                    Header-only investigation harness (wired into main.cpp)
│   ├── NARMA_N_Benchmark.h         NARMA-N benchmark + generator
│   ├── StateRank.h                 Effective-rank / input-linearity measurement
│   └── StateRankSurvey.h           Hyperparameter sweep + NARMA-N validation
└── examples/                       Standalone usage examples
```
