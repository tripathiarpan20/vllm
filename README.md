# H100 sampler fusion validation

This bundle compares the unmodified Model Runner V2 Gumbel sampler with the
final-reduction fusion in `patched_gumbel.py`. It contains no model weights.

Source pins:

- Implementation base: `8538017f4b1d01b543345f5dbe1993cb66c80fa9`.
- GPU runtime source and native wheel: `e6b1a5e5e3fce6b777c5f63398f87c7c911a40b3`,
  the immediate parent. The two commits have identical sampler source; the newer
  commit changes the token-in/token-out frontend, not this sampler or the
  completions endpoint used here. A wheel for the newer commit was unavailable
  when testing began.
- Model: `Qwen/Qwen3.8-27B-FP8`, revision
  `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. Its config uses vLLM's shared
  `Qwen3_5ForConditionalGeneration` implementation. All 66 weight shards were
  SHA-256 checked against the pinned Hugging Face metadata.

## Reproduction

The scripts use `/workspace/pareton-sampler-review` as their working directory.
Use a dedicated H100 with enough disk for the 29 GiB checkpoint and dependencies.
Install `uv`, place this bundle at that path, then:

```bash
cd /workspace/pareton-sampler-review
uv venv --python 3.12 .venv
curl -fLsS https://codeload.github.com/vllm-project/vllm/tar.gz/e6b1a5e5e3fce6b777c5f63398f87c7c911a40b3 -o runtime-base.tar.gz
mkdir runtime-src
tar -xzf runtime-base.tar.gz --strip-components=1 -C runtime-src
git -C runtime-src init
bash setup.sh
uvx --from huggingface_hub hf download Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a --local-dir model
.venv/bin/python collect_environment.py
cp patched_gumbel.py runtime-src/vllm/v1/worker/gpu/sample/gumbel.py
cp test_gpu_gumbel_sample.py runtime-src/tests/v1/worker/test_gpu_gumbel_sample.py
cd runtime-src
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest tests/v1/worker/test_gpu_gumbel_sample.py -q
OMP_NUM_THREADS=4 ../.venv/bin/python -m pytest \
  tests/watermarking/test_gumbel.py tests/v1/spec_decode/test_rejection_sampler_utils.py \
  -k 'skip_mask_matches_separate_samplers or gumbel_drafted_rejection_sample_is_unbiased' -q
cd ..
OMP_NUM_THREADS=4 .venv/bin/python check_sampler_integration.py
OMP_NUM_THREADS=4 .venv/bin/python benchmark_sampler.py
OMP_NUM_THREADS=4 .venv/bin/python run_campaign.py
OMP_NUM_THREADS=4 .venv/bin/python run_live_parity.py
.venv/bin/python analyze_results.py
.venv/bin/python analyze_eval.py
.venv/bin/python analyze_profiles.py
.venv/bin/python make_report.py
```

Preserve the archived `artifacts` directory before reproducing. Start a fresh
output directory named `artifacts`, copying only `requests.json` into it to
reuse the exact input token manifest. The serving client appends to
`sla-results.jsonl`; never combine a repeat with the archived measurements.

`artifacts/pip-freeze.txt` records the tested dependency versions;
`constraints.txt` pins those versions in the reproduction setup. The initial editable
install selected CUDA 13.4 compiler packages with CUDA 13.2 runtime headers;
`setup.sh` aligns NVCC, CRT and NVVM with 13.2 before model JIT compilation.
The failed pre-measurement startup is retained in `artifacts/failed-startup`.
A second startup exposed missing unversioned CUDA library links for FlashInfer;
`cuda-link-libs` and the launcher library paths resolve this without changing
vLLM source. Its log is retained in `artifacts/failed-startup-linker`.

## Protocol and limits

`benchmark_sampler.py` checks exact old/new token equality before measuring
FP32/FP64 sampling across vocabularies 32,768, 128,256 and 248,320, and row
counts 1, 4, 8, 16, 32, 128 and 512. It times both the complete stochastic
sampler (draft noise stream, temperature 1) and the isolated final reduction.
Each cell measures eager dispatch and a CUDA graph of 50 calls. Each arm has
15 warmups and 25 CUDA-event observations, with baseline/patched then
patched/baseline ordering. Eager observations contain 50 calls each. Reported
values average the two arm medians. Eager event timings include GPU idle gaps
between Python launches; graph timings better isolate device work.

`run_campaign.py` runs baseline, patched, patched, baseline in separate server
sessions, with no simultaneous GPU benchmark. Each session performs two
measured sweeps at client concurrency 1, 4, 8, 16 and 32. Each cell uses the
same 32 requests: eight at each input/output length 256/32, 768/64, 1536/96 and
3072/128. Greedy sampling, seed 42 and `ignore_eos` keep output work fixed.
The prefix cache is reset before each measured cell. An untimed 32-request warmup with 16 output tokens at each of the five
concurrencies (160 requests per session) and all startup phases are excluded.
`VLLM_DEEP_GEMM_WARMUP=skip` avoids the broad startup shape sweep; the request
warmups exercise the benchmark workload before timing in every server session.

TTFT ends at the first token-bearing stream event. Throughput includes the
wall time for the entire request sweep, including waiting between waves.
Per-cell p99 uses NumPy linear percentile interpolation over 32 requests;
the report uses the worst such p99 across four repetitions per variant.
Raw stream timing, token IDs and Prometheus snapshots are retained.
The client also records checks against 2-second TTFT and 50-ms stream-gap
thresholds; these are descriptive checks, not Pareton campaign scores.

`eval_client.py` adds 24 sequential untimed requests per session, covering
greedy, temperature 0.8, and temperature 0.8/top-p 0.9/top-k 40. Each uses an
explicit seed and returns 64 token IDs plus top-5 log probabilities. These
finite model-output comparisons are not a task-accuracy benchmark.

Separate untimed eight-request, eight-output-token profiles in baseline-a and patched-a establish
that serving reaches the changed sampler. Profiling is inactive during timing.
Compilation caches are shared; startup observations are not a controlled
cold-start comparison. No speculative decoding is enabled in the serving run.

The patch is adapted from a Pareton AI campaign submission for general upstream
use. Credit: @xavierlyu, then @Danbog32. Campaign-specific settings and unrelated
optimizations are absent. AI assistance was used for implementation and testing.

The final implementation is commit `a73283aca52e0f5b6aa4774b31663bd90fb6f0e4`
on `arpan/fuse-sampler-argmax-gather`. `source-manifest.json` pins the measured
modules, regression tests and standalone patch. The optional report generator
expects the archived module-hash files and completed four-session campaign.

`run_live_parity.py` starts a separate, untimed validation server after the
ABBA campaign. Its temporary wrapper runs both samplers on identical live model
logits and asserts token equality on every checked call. It enables checking
only after server readiness and restores the production module on exit. This
synchronous, instrumented server is never used for throughput measurements.
The validation runs the same 24 seeded evaluation requests.
