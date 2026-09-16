# GDN mixed-output scatter: H100 validation evidence

Validation of vLLM patch `5fe281e336f8f25aab2f89cf85c211bf83594551` against base `0f8fa53acf43a9a6a7a1f7e6b196950c78cd6d7a`, run on September 16, 2026.

Download [gpu-validation.tar.gz](gpu-validation.tar.gz) for the fixed input manifest, benchmark and component-test scripts, raw per-request results, model-token comparisons, counter snapshots, server logs, environment details, profiler trace, and checksums. Extract the archive and read its README for reproduction instructions. [benchmark-summary.json](benchmark-summary.json) contains the aggregate measurements.

The serving comparison uses Qwen/Qwen3.5-4B BF16 with three MTP speculative tokens on one H100 PCIe, client batch sizes 1/4/8/16/32, and identical inputs. Four server sessions run in baseline/patched/patched/baseline order, with two measured sweeps per session. All 2,560 requests succeeded.

The archive preserves the original measurements and analysis. Only the GDN source module differs between the serving variants. Compilation caches were shared between sessions; the logged startup times are not a controlled cold-compilation comparison. This evidence is separate from the upstream code diff.
