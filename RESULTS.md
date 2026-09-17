## Purpose

Model Runner V2's native Gumbel sampler reduces its block maxima with `torch.argmax`, then launches a separate `torch.gather` to recover the winning vocabulary token ID. Fuse those two final operations into one Triton kernel. This removes one GPU launch and the intermediate block-index tensor per sampler call.

The opportunity was identified while reviewing a submission to a [Pareton AI inference-optimization campaign](https://www.pareton.ai/dashboard/campaigns/7e0462e4-5806-44ac-9f5b-af0542a4bb86). This change extracts the general sampler improvement for upstream review. It does not import the submission's model-specific settings or present its campaign score as evidence for this isolated change.

Credit to @xavierlyu and @Danbog32.

### Change and scope

The new reduction selects the winning block and directly loads its mapped token ID. It preserves first-index ties, the first-NaN behavior of `torch.argmax`, FP64 comparison precision, and the int64 result. The existing first-stage Gumbel computation, seeds, draft/target noise separation, temperature application, and logits-cache writes are unchanged.

This applies to calls through `vllm/v1/worker/gpu/sample/gumbel.py`, including the Model Runner V2 native sampler. It does not accelerate the FlashInfer sampling branch or Model Runner V1's separate sampler. The serving checks below exercise Model Runner V2 with explicit seeds; no speculative decoding is enabled in those model runs.

### Related work

Open-PR searches and diff inspection on September 17, 2026 found no change replacing this final reduction in the existing `gumbel_sample` path:

- [#38399](https://github.com/vllm-project/vllm/pull/38399) introduces fused draft LM-head sampling and a related block-reduction helper for that new path. Its diff retains the existing `gumbel_sample` argmax/gather fallback. This contribution optimizes that existing shared sampler and preserves its FP64/NaN semantics.
- [#47524](https://github.com/vllm-project/vllm/pull/47524) changes the random-number construction; this change leaves it untouched.
- [#50843](https://github.com/vllm-project/vllm/pull/50843) bounds tile-local token indices, rather than fusing the final block reduction.
- [#56494](https://github.com/vllm-project/vllm/pull/56494) migrates sampler warmup, and [#46499](https://github.com/vllm-project/vllm/pull/46499) adds a sampler benchmark.
- [#57235](https://github.com/vllm-project/vllm/pull/57235) removes a distributed full-vocabulary gather for a restricted greedy path. The `gather` removed here is a local tensor lookup after block reduction, with no tensor-parallel communication involved.

Search terms included `gumbel`, `argmax gather`, and `gumbel fuse`. The merged [#57140](https://github.com/vllm-project/vllm/pull/57140) concerns GDN output assembly and is independent of this sampler change.

## Test Plan

Implementation commit: `a73283aca52e0f5b6aa4774b31663bd90fb6f0e4`. The measured production module SHA-256 is `3d3737256cf700bbb2739382d4d7b06ff21ab30734a67accbf12a85c5e4b1820`; the baseline is `78cc3d6ba0142e86ba169443dd3161dbf15c88c2ae4e795b8a9bfc31b96fcba8`.

Implementation is based on `8538017f4b1d01b543345f5dbe1993cb66c80fa9`. GPU tests use source and the published native wheel from its immediate parent, `e6b1a5e5e3fce6b777c5f63398f87c7c911a40b3`, because a wheel for the newer commit was unavailable when testing began. The sampler source is byte-for-byte identical between these commits; their only differences are in the token-in/token-out frontend and its test. This benchmark uses the completions API.

Both GPU arms use the same runtime checkout, wheel, dependencies, pinned model, request manifest, and launch settings. The launcher replaces only `vllm/v1/worker/gpu/sample/gumbel.py`, and records that file's SHA-256 for every server session. Supplementary scripts and raw evidence are in the accompanying validation bundle.

### Environment and model

| Component | Configuration |
| --- | --- |
| GPU | 1 × NVIDIA H100 PCIe, 81,559 MiB reported memory |
| CPU / memory quota | 20 vCPUs, Intel Xeon Platinum 8352Y @ 2.20 GHz; approximately 123 GiB container memory limit |
| OS / driver | Ubuntu 24.04.4 LTS; NVIDIA 580.126.09 |
| Runtime | Python 3.12.3, PyTorch 2.13.0+cu132, CUDA runtime 13.2, Triton 3.7.1, FlashInfer 0.6.18.post1 |
| vLLM | Published wheel for `e6b1a5e5e`, Model Runner V2, compilation and CUDA graphs enabled |
| Model | `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Model execution | FP8 weights, BF16 activations, TP=1, text-only, vocabulary 248,320, no speculative decoding |

The checkpoint uses the shared `Qwen3_5ForConditionalGeneration` architecture implementation; the downloaded model is the specified Qwen3.8 revision. All 66 weight shards were checked against their pinned Hugging Face SHA-256 hashes.

Two environment problems were resolved before measurement: the editable install initially selected a CUDA 13.4 compiler alongside 13.2 runtime headers, and FlashInfer's JIT linker needed unversioned links to the pip-provided CUDA libraries. The reproduction setup pins NVCC/CRT/NVVM to 13.2 and provides local library links. Failed startup logs are retained separately. Neither fix changes vLLM source or differs between benchmark arms.

### Correctness checks

Extend the existing GPU Gumbel suite with eight final-reduction cases spanning 1, 3, 243 and 1,025 blocks in FP32/FP64. They check ties, NaNs, infinities, non-power-of-two padding, mapped token IDs, FP64-distinguishable maxima, and CUDA-graph replay after changing both values and mappings.

A supplementary baseline/patched public-sampler comparison covers 36 combinations of FP32/BF16/FP16 logits, FP32/FP64 sampling, draft/target streams, and vocabulary sizes 1,023, 1,025 and 248,320. It uses strided logits, repeated request mappings, mixed temperatures, distinct cache columns and padded caches. Existing targeted tests cover watermark fallback sampling and unbiased rejection sampling with Gumbel-drawn draft tokens.

```bash
.venv/bin/python -m pytest tests/v1/worker/test_gpu_gumbel_sample.py -q

.venv/bin/python -m pytest \
  tests/watermarking/test_gumbel.py \
  tests/v1/spec_decode/test_rejection_sampler_utils.py \
  -k 'skip_mask_matches_separate_samplers or gumbel_drafted_rejection_sample_is_unbiased' -q

pre-commit run --files \
  vllm/v1/worker/gpu/sample/gumbel.py \
  tests/v1/worker/test_gpu_gumbel_sample.py

pre-commit run mypy-3.12 --hook-stage manual --files \
  vllm/v1/worker/gpu/sample/gumbel.py \
  tests/v1/worker/test_gpu_gumbel_sample.py
```

Static checks run on macOS ARM64/Python 3.12.6. GPU tests and model evaluations run on the H100.

### Kernel benchmark

Compare both the complete stochastic sampler and its isolated final reduction across vocabulary sizes 32,768, 128,256 and 248,320; row counts 1, 4, 8, 16, 32, 128 and 512; FP32/FP64 sampling; and eager/CUDA-graph execution. Full-sampler inputs are FP32 logits with temperature 1 and the draft noise stream. Exact baseline/patched token equality is checked before timing.

Each arm gets 15 warmups and 25 CUDA-event observations, with 50 calls per observation. Graph observations replay a graph containing 50 calls. Each cell runs baseline/patched and then patched/baseline; the table averages the two arm medians. Eager event timings include idle gaps between Python launches. These component measurements do not establish an end-to-end model speedup.

### Serving protocol

A fixed manifest contains 32 requests, eight at each input/output token pair: **256/32, 768/64, 1,536/96, and 3,072/128**. Input token IDs are saved once and reused in the same order; manifest SHA-256: `f68614cd67898b5b166170c52da5ecd65de9f4149dd4405eb12e12f9b437c3de`. Requests use greedy sampling, seed 42 and `ignore_eos=true`.

Run four independent server sessions in **baseline → patched → patched → baseline** order, with two measured sweeps per session at client concurrency **1, 4, 8, 16 and 32**. Submit requests in waves and wait for each wave to complete. This gives four repetitions and 128 measured requests per variant/concurrency. The client runs over loopback on the same VM, with no competing GPU benchmark.

Before measurement, warm all five concurrency settings with 32 requests and 16 output tokens per request, resetting the prefix cache for each setting. Reset the prefix cache again before every measured cell. `VLLM_DEEP_GEMM_WARMUP=skip` disables the broad startup shape sweep; every session executes this workload warmup before timing. Compilation caches are shared, so startup timings are observations rather than a controlled cold-start comparison.

TTFT ends at the first token-bearing stream event. Report the worst per-run p99 across four repetitions, each calculated over 32 requests with NumPy linear interpolation. Throughput is all generated tokens divided by the wall time of the complete sweep, including waits between waves; report the median and min–max of four observations. No failed or slow measured samples are removed.

```bash
VLLM_DEEP_GEMM_WARMUP=skip VLLM_USE_V2_MODEL_RUNNER=1 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  vllm serve /path/to/pinned/Qwen3.8-27B-FP8 \
  --served-model-name qwen-sampler --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 --seed 42 \
  --max-model-len 4096 --max-num-seqs 32 --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.80 --enable-prefix-caching --enable-chunked-prefill \
  --gdn-prefill-backend triton --mamba-cache-mode align \
  --generation-config vllm --language-model-only
```

The full launcher additionally configures the repaired CUDA library paths and development cache-reset/profiling endpoints. Profiling is inactive during measurements. Separate untimed eight-request, eight-output-token profiles in baseline-a and patched-a verify execution of the changed path.

Each server session also evaluates 24 sequential untimed requests returning 64 token IDs and top-5 log probabilities: eight greedy, eight at temperature 0.8, and eight at temperature 0.8/top-p 0.9/top-k 40. Each has an explicit seed. These are finite output comparisons, not a task-accuracy score.

An additional, untimed validation server compares both samplers on the exact
same live Qwen logits before returning the patched result. Its temporary wrapper
asserts equality on every checked call and is enabled only after server readiness.
This separates sampler equivalence from numerical or scheduling variation across
independent server sessions. The instrumented server runs the same 24 seeded
evaluation requests and contributes no performance samples.

## Test Result

### Correctness and static checks

| Check | Result |
| --- | --- |
| Existing GPU Gumbel suite plus new reduction regressions | **30 passed** |
| Targeted watermark fallback and Gumbel-drafted rejection sampling | **4 passed**, 74 unrelated cases deselected |
| Supplementary baseline/patched public-sampler matrix | **36 passed**, exact token and cache equality; padding preserved |
| New CUDA-graph replay regression cases | All eight passed after mutating values and mappings |
| Kernel benchmark equivalence checks | Exact baseline/patched sampled IDs for all 42 full-sampler shapes and all 42 final-reduction shapes |
| Applicable pre-commit hooks, default mypy and explicit Python 3.12 mypy | Passed |
| Patch whitespace and final source hashes | Passed |
| Both samplers on identical live Qwen logits | **1,544 calls / 1,544 rows matched exactly**, during 24 additional seeded requests |

### Isolated kernel timing

CUDA-event timings in microseconds at the Qwen vocabulary size of **248,320**. Values are the average of the two measurement-order medians. The complete matrix, including eager execution, is retained in `kernel-bench.jsonl`.

| Rows | Final reduction, graph FP32 baseline → patched (µs) | Full sampler, graph FP32 baseline → patched (µs) | Full sampler, graph FP64 baseline → patched (µs) |
| --- | --- | --- | --- |
| 1 | 4.43 → 1.55 | 8.05 → 4.89 | 11.42 → 7.73 |
| 4 | 4.64 → 1.61 | 12.55 → 9.04 | 19.48 → 15.43 |
| 8 | 4.80 → 1.62 | 17.60 → 13.89 | 29.99 → 25.93 |
| 16 | 5.25 → 1.59 | 28.02 → 24.04 | 50.90 → 46.63 |
| 32 | 5.25 → 1.63 | 48.11 → 43.96 | 97.73 → 92.04 |
| 128 | 5.45 → 1.74 | 199.61 → 183.94 | 394.47 → 392.09 |
| 512 | 6.84 → 1.96 | 764.23 → 762.65 | 1560.70 → 1556.67 |

| Rows | Final reduction, eager FP32 baseline → patched (µs) | Full sampler, eager FP32 baseline → patched (µs) |
| --- | --- | --- |
| 1 | 23.20 → 17.07 | 74.26 → 67.59 |
| 8 | 23.68 → 17.14 | 75.00 → 67.64 |
| 32 | 23.37 → 16.91 | 74.78 → 67.42 |
| 512 | 23.80 → 17.09 | 749.80 → 763.68 |

Across the entire shape/precision matrix, the graph-executed final reduction is **2.46–3.50× faster**. Complete graph-executed sampling ranges from **1.002–1.79×**; complete eager sampling ranges from **0.981–1.15×**, including a small slower observation at the large-work end. The saving is a few microseconds per sampler call, and becomes a small fraction of total work at large row counts. These are kernel measurements, not model throughput ratios.

### Serving throughput and latency

All **1,280/1,280 measured requests succeeded**, producing **102,400 tokens** across the four server sessions.

| Client concurrency | Baseline output tok/s, median [min–max] | Patched output tok/s, median [min–max] | Median change |
| --- | --- | --- | --- |
| 1 | 48.49 [48.26–48.61] | 48.56 [48.45–48.70] | +0.14% |
| 4 | 94.38 [93.80–94.48] | 94.30 [93.85–94.57] | -0.09% |
| 8 | 145.35 [144.88–145.92] | 144.40 [144.35–145.18] | -0.65% |
| 16 | 200.62 [196.58–206.76] | 199.66 [198.84–201.39] | -0.48% |
| 32 | 250.77 [248.98–254.04] | 249.18 [246.32–253.13] | -0.64% |

| Client concurrency | Worst p99 TTFT, baseline → patched (ms) | Worst p99 stream interval, baseline → patched (ms) |
| --- | --- | --- |
| 1 | 490.9 → 481.5 | 18.7 → 18.7 |
| 4 | 1087.1 → 1062.2 | 133.3 → 134.4 |
| 8 | 1996.6 → 1994.8 | 136.2 → 137.7 |
| 16 | 3924.5 → 3860.4 | 150.1 → 142.3 |
| 32 | 7345.8 → 7456.1 | 142.5 → 140.5 |

The p99 columns report the worst of four finite, 32-request repetitions. They are workload observations, not production tail-latency guarantees. The raw data retains every measured request and all streaming events.

### Generated-output comparisons

Compare complete token-ID sequences against the first baseline sweep at the same concurrency. Other baseline sweeps are a control for model/scheduler variation.

| Client concurrency | Exact matches in other baseline sweeps | Exact matches in patched sweeps |
| --- | --- | --- |
| 1 | 62/96 | 70/128 |
| 4 | 58/96 | 74/128 |
| 8 | 70/96 | 76/128 |
| 16 | 70/96 | 80/128 |
| 32 | 70/96 | 76/128 |

The separate seeded evaluation completed **96/96 requests** across four sessions, producing 6,144 tokens. The first baseline session supplies 24 references; the remaining 72 requests are compared below. Logprob equality covers the complete returned logprob payload.

| Sampling mode | Other baseline: exact tokens / logprobs | Patched: exact tokens / logprobs |
| --- | --- | --- |
| greedy | 3/8 / 0/8 | 14/16 / 0/16 |
| random | 5/8 / 0/8 | 10/16 / 0/16 |
| filtered | 3/8 / 0/8 | 9/16 / 0/16 |

These finite generation checks do not measure task accuracy or establish bitwise full-model equivalence across independently scheduled sessions. In a separate validation server, both samplers returned identical IDs for all **1,544 calls / 1,544 rows** when given the exact same live model logits. That direct comparison isolates the sampler change; it does not make independently generated model logits identical.

### Execution-path and startup evidence

The untimed baseline profile contains **14 `aten::argmax` calls and 14 matching `aten::gather` calls** on `[N, 243]` block scratch tensors. The patched profile contains **14 `_gumbel_sample_reduce_kernel` launches** and no argmax/gather calls on that scratch shape. This verifies that the serving workload reaches the changed path; the profiles are not used as timing samples.

| Session | Reported compilation (s) | Initial profile/warmup (s) | CUDA graph capture (s) |
| --- | --- | --- |
| baseline-a | 41.13 | 0.39 | 4 |
| patched-a | 0.68 | 0.46 | 4 |
| patched-b | 0.66 | 0.51 | 4 |
| baseline-b | 0.70 | 0.57 | 4 |

Startup and request warmup are excluded from measured cells. Compilation/JIT caches were shared and partly populated while repairing the initial setup, so these observations cannot establish a startup improvement or regression. The benchmark servers were stopped after the final run.

### Interpretation and limits

The patch consistently reduces the isolated final-reduction cost and removes the expected launch and intermediate allocation. Serving median throughput changes range from **-0.65% to +0.14%** across the five tested concurrencies. The complete ranges and tail-latency results above must be considered alongside those medians; four repetitions on one H100 are not a general speedup guarantee. These serving runs do not demonstrate a model-level throughput improvement. The demonstrated benefit is the isolated launch/allocation reduction.

This validation covers one Qwen3.8-27B-FP8 revision, one H100 PCIe, TP=1, one serving configuration, and finite synthetic requests. It does not qualify other GPUs, tensor-parallel sizes, quantizations, the FlashInfer sampling branch, or Model Runner V1. Stochastic draft behavior is covered by sampler/rejection tests and component measurements, not speculative full-model serving. No unrelated campaign optimizations are present in either serving arm.

AI assistance was used for implementation, review, and validation support.
