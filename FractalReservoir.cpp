#include "FractalReservoir.h"
#include <algorithm>
#include <random>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

template <size_t DEPTH>
FractalReservoir<DEPTH>::FractalReservoir(const FractalReservoirConfig<DEPTH>& cfg)
    : rng_seed_(cfg.seed),
      num_inputs_(cfg.num_inputs)
{
    if (num_inputs_ != 1 && num_inputs_ != 2 && num_inputs_ != 4 && num_inputs_ != 8)
        throw std::invalid_argument("fractal reservoir num_inputs must be in {1, 2, 4, 8}");

    for (size_t d = 0; d < DEPTH; ++d)
    {
        const auto& lvl = cfg.levels[d];
        const std::string ld = "fractal reservoir level " + std::to_string(d);
        if (lvl.alpha <= 0.0f)
            throw std::invalid_argument(ld + ": alpha must be positive");
        if (lvl.spectral_radius <= 0.0f)
            throw std::invalid_argument(ld + ": spectral_radius must be positive");
        if (lvl.leak_rate <= 0.0f || lvl.leak_rate > 1.0f)
            throw std::invalid_argument(ld + ": leak_rate must be in (0.0, 1.0]");
        if (lvl.input_scaling <= 0.0f)
            throw std::invalid_argument(ld + ": input_scaling must be positive");

        alpha_[d]           = lvl.alpha;
        spectral_radius_[d] = lvl.spectral_radius;
        leak_rate_[d]       = lvl.leak_rate;
        input_scaling_[d]   = lvl.input_scaling;
    }

    Initialize();
}

template <size_t DEPTH>
void FractalReservoir<DEPTH>::Initialize()
{
    std::mt19937_64 rng(rng_seed_);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // With only NUM_CONNECTIONS = 3 recurrent weights per vertex, a single
    // near-zero draw silences a meaningful fraction of that vertex's input.
    // Reject draws below `min_weight` and resample.
    auto nonzero_draw = [&](const float min_weight)
    {
        float v;
        do { v = static_cast<float>(dist(rng)); }
        while (std::abs(v) < min_weight);
        return v;
    };

    constexpr size_t num_weights = N * DIM;
    const float w_scale = 1.0f / std::sqrt(static_cast<float>(DIM));

    // Recurrent weights. Walk sub-reservoirs in level-major order so each
    // level's per-sub draws are consecutive; this matches the per-level SR
    // rescale that follows each sub-reservoir's draws below.
    for (size_t depth = 0; depth < DEPTH; ++depth)
    {
        const size_t depth_sub_count  = 1ULL << (3 * depth);     // N^depth
        const size_t depth_sub_offset = LEVEL_OFFSET[depth] / N;
        const float  sr_target        = spectral_radius_[depth];
        for (size_t s = 0; s < depth_sub_count; ++s)
        {
            const size_t sub = depth_sub_offset + s;
            const size_t offset = sub * num_weights;
            for (size_t i = 0; i < num_weights; ++i)
                vtx_weight_[offset + i] = nonzero_draw(0.05f) * w_scale;

            // Rescale recurrent weights to this level's target spectral radius.
            float current_sr = EstimateSpectralRadius(vtx_weight_ + offset);
            if (current_sr > 1e-6f)
            {
                float scale = sr_target / current_sr;
                for (size_t i = 0; i < num_weights; ++i)
                    vtx_weight_[offset + i] *= scale;
            }
        }
    }

    // No per-vertex random input weights. Each level applies a single scalar
    // `input_scaling_[depth]` to whatever it receives from above (external at
    // depth 0; parent's queued pre-activation `s` at depth >= 1). Used inline
    // in UpdateState / UpdateRecursive; nothing to initialize here.

    // Input queues start at zero (Reset zeros them too). Warmup repopulates
    // them with parent pre-activations within a few timesteps.

    // Randomized initial state with small amplitude: keeps the initial transient
    // short and makes Reset() restore a non-zero state (preserved in vtx_initial_).
    // Fresh distribution so initial values are independent of weight draws.
    std::uniform_real_distribution<double> init_dist(-0.05, 0.05);
    for (size_t i = 0; i < VTX_SIZE; ++i)
    {
        float val = static_cast<float>(init_dist(rng));
        vtx_output_[i] = val;
        vtx_initial_[i] = val;
    }
}

template <size_t DEPTH>
void FractalReservoir<DEPTH>::UpdateState(std::span<const float> inputs)
{
    if (inputs.size() != num_inputs_)
        throw std::invalid_argument(
            "FractalReservoir::UpdateState: inputs.size() must equal cfg.num_inputs");

    // Synchronous ESN update at depth 0. The K=num_inputs_ external channels
    // feed an N=8-slot delay-line queue partitioned into K sub-queues of
    // length sub_len = N/K (8, 4, 2, 1 for K = 1, 2, 4, 8). Each step, before
    // the v-loop, every channel's sub-queue is shifted right by one and the
    // new input is written to slot 0. Vertex v then reads cell
    // (v / sub_len) * sub_len + (v % sub_len) -- channel = v / sub_len (high
    // bits), lag = v % sub_len (low bits). Block mapping: contiguous vertex
    // indices share a channel; lag varies fastest. At K=8 sub_len=1 so each
    // sub-queue is a single slot and the buffer is just the input vector.
    // External-input handling is a depth-0 concern only; deeper sub-reservoirs
    // receive 8 scalars per step from the parent vertex's 8-deep input queue
    // (one slot per child vertex, identity map) and have no notion of
    // `num_inputs`.
    const float alpha0 = alpha_[0];
    const float leak0  = leak_rate_[0];
    const float in_w0  = input_scaling_[0];

    const size_t sub_len = N / num_inputs_;
    for (size_t c = 0; c < num_inputs_; ++c)
    {
        float* sub = external_input_queue_ + c * sub_len;
        if (sub_len > 1)
            std::memmove(sub + 1, sub, (sub_len - 1) * sizeof(float));
        sub[0] = inputs[c];
    }

    float* state = vtx_output_;
    float new_state[N];
    for (size_t v = 0; v < N; ++v)
    {
        const float* w = vtx_weight_ + v * DIM;
        const size_t ch  = v / sub_len;
        const size_t lag = v % sub_len;
        float s = external_input_queue_[ch * sub_len + lag] * in_w0;

        for (size_t i = 0; i < DIM; ++i)
            s += state[v ^ NearestMask(i)] * w[i];

        float activation;
        if constexpr (DEPTH == 1)
        {
            activation = std::tanh(alpha0 * s);
        }
        else
        {
            // Shift this vertex's queue and enqueue the freshly computed s
            // at slot 0 BEFORE recursing — the child sub-reservoir reads
            // slot 0 as its newest input on this step.
            float* my_queue = input_queue_ + v * N;
            std::memmove(my_queue + 1, my_queue, (N - 1) * sizeof(float));
            my_queue[0] = s;
            activation = UpdateRecursive(1, v, my_queue);
        }

        new_state[v] = (1.0f - leak0) * state[v] + leak0 * activation;
    }

    std::memcpy(state, new_state, sizeof(new_state));
}

// Synchronous ESN update at one fractal level (depth >= 1). The sub-reservoir
// at (depth, path_idx) owns N contiguous states at LEVEL_OFFSET[depth] +
// path_idx*N. `parent_queue` points to the parent vertex's 8-deep input
// queue; child vertex v reads `parent_queue[v]` as its input scalar (identity
// slot mapping; slot 0 is newest). For non-leaf depths each child vertex
// also shifts its OWN pre-activation into its own queue before recursing.
// Returns tanh of the sum of this sub-reservoir's post-update states -- the
// scalar readout handed up to the parent as that parent vertex's activation.
template <size_t DEPTH>
float FractalReservoir<DEPTH>::UpdateRecursive(const size_t depth, const size_t path_idx,
                                               const float* parent_queue)
{
    // Sub-reservoir index in flat per-sub arrays. The geometric-sum base
    // (1 + N + N^2 + ... + N^(depth-1)) equals LEVEL_OFFSET[depth] / N.
    const size_t sub_res_idx = LEVEL_OFFSET[depth] / N + path_idx;

    const size_t offset_vtx = sub_res_idx * N * DIM;

    const float alpha_d = alpha_[depth];
    const float leak_d  = leak_rate_[depth];
    const float in_w_d  = input_scaling_[depth];

    float* state = vtx_output_ + LEVEL_OFFSET[depth] + path_idx * N;

    float new_state[N];
    for (size_t v = 0; v < N; ++v)
    {
        const float* w = vtx_weight_ + offset_vtx + v * DIM;
        float s = parent_queue[v] * in_w_d;

        for (size_t i = 0; i < DIM; ++i)
            s += state[v ^ NearestMask(i)] * w[i];

        float activation;
        if (depth == DEPTH - 1)
        {
            activation = std::tanh(alpha_d * s);
        }
        else
        {
            float* my_queue = input_queue_ + (LEVEL_OFFSET[depth] + path_idx * N + v) * N;
            std::memmove(my_queue + 1, my_queue, (N - 1) * sizeof(float));
            my_queue[0] = s;
            activation = UpdateRecursive(depth + 1, path_idx * N + v, my_queue);
        }

        new_state[v] = (1.0f - leak_d) * state[v] + leak_d * activation;
    }

    std::memcpy(state, new_state, sizeof(new_state));

    float out = 0.0f;
    for (size_t v = 0; v < N; ++v)
        out += state[v];
    return std::tanh(out);
}

// Power iteration on the (non-symmetric) recurrent weight matrix.
// This computes the spectral norm (largest singular value), which is the
// standard proxy for the spectral radius in reservoir computing literature.
template <size_t DEPTH>
float FractalReservoir<DEPTH>::EstimateSpectralRadius(const float* weights) const
{
    std::vector<float> x(N), y(N);

    std::mt19937_64 rng(rng_seed_ + 3423984);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    float norm = 0.0f;
    for (size_t v = 0; v < N; v++)
    {
        x[v] = static_cast<float>(dist(rng));
        norm += x[v] * x[v];
    }
    norm = std::sqrt(norm);
    for (size_t v = 0; v < N; v++) x[v] /= norm;

    float eigenvalue = 0.0f;
    float prev_eigenvalue = 0.0f;
    for (int iter = 0; iter < 100; iter++)
    {
        for (size_t v = 0; v < N; v++)
        {
            float s = 0.0f;
            const float* w = weights + v * DIM;

            for (size_t i = 0; i < DIM; i++)
                s += w[i] * x[v ^ NearestMask(i)];

            y[v] = s;
        }

        norm = 0.0f;
        for (size_t v = 0; v < N; v++) norm += y[v] * y[v];
        norm = std::sqrt(norm);
        eigenvalue = norm;

        if (norm > 1e-12f)
            for (size_t v = 0; v < N; v++) x[v] = y[v] / norm;

        if (iter > 5 && std::abs(eigenvalue - prev_eigenvalue) < eigenvalue * 1e-6f)
            break;
        prev_eigenvalue = eigenvalue;
    }

    return eigenvalue;
}

template class FractalReservoir<1>;
template class FractalReservoir<2>;
template class FractalReservoir<3>;
template class FractalReservoir<4>;
template class FractalReservoir<5>;
