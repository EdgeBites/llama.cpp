# EdgeBites/llama.cpp

**Fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) with an on-device LLM stack tuned for Snapdragon 8 Gen 2 / Adreno 740-class GPUs (SM8550, A7X).**

This fork's `master` carries three OpenCL performance fixes for the Adreno backend, each developed and verified on an SM8550 device with the E031 compiler. All three are proposed upstream:

| Fix | Upstream PR | Upstream status |
|---|---|---|
| Enable f32_f16 flash attention on Adreno A7x | [#30639](https://github.com/ggml-org/llama.cpp/pull/30639) | open (draft) |
| Use trans-weight GEMM for narrow q4_0 weights | [#30646](https://github.com/ggml-org/llama.cpp/pull/30646) | open |
| Add q4_0 `o4` GEMV for long-vocab lm_head | [#30658](https://github.com/ggml-org/llama.cpp/pull/30658) | open (draft) |

Full upstream documentation (build instructions, backends, tooling, server, etc.) lives in [README.upstream.md](README.upstream.md).

---

## 1. OpenCL: enable f32_f16 flash attention on Adreno A7x — [#30639](https://github.com/ggml-org/llama.cpp/pull/30639)

The A7x carve-out (#25697) disables OpenCL flash attention for every mixed or quant KV type because the E031 compiler SIGSEGVs inside `clBuildProgram` while building those FA programs. That is broader than needed: bisecting the FA program by kernel guard shows that only the mixed-KV `flash_attn_f32_f16_q1_vec` kernel crashes the compiler — a program containing just that kernel crashes, and excluding it compiles cleanly, independent of DK.

This change:

- excludes `q1_vec` for a7x `f32_f16` programs (plain and `N_SPLIT>1`) via `-D FA_DECODE_MINIMAL`, the same idiom the DK=512 path already uses;
- drops `f32_f16` from the two a7x declines (the condition is duplicated in this branch; both copies are updated). `q8_0`/`q4_0` stay declined — their kernel sources have no such guard.

Net effect on an Adreno 740 (E031.41.03.64): f16-KV FA works again; the prefill, `q1`, `mq` and split decode kernels all compile and run.

Verified on device — Adreno 740, compiler E031.41.03.64, `llama-bench --device GPUOpenCL -ngl -1 -p 256 -n 64 -r 2`:

**LFM2.5-230M Q4_0 (FA on vs off):**

| | pp256 t/s | tg64 t/s |
|---|---|---|
| FA off | 863.7 ± 1.1 | 83.2 ± 0.6 |
| FA on | **2006.5 ± 9.1 (+132%)** | **107.6 ± 1.4 (+29%)** |

Long-context check (same model, ~1.7k-token prompt): FA on pp 1667 / tg 71.2 vs FA off pp 1055 / tg 67.0.

Other models: gemma-4-E2B pp256 31.9 → 45.7 (+45%), tg neutral; gemma-4-E4B pp256 22.0 → 29.0 (+32%), tg neutral.

How the crasher was isolated: a small standalone `clBuildProgram` harness compiling the FA kernel source with the same options as the backend, plus bisecting the source's `FA_*` guards to find which kernel triggers the SIGSEGV.

Related: #25697 (the carve-out), #25870 (closed; same compiler family, different failure mode).

---

## 2. OpenCL: use Adreno trans-weight GEMM for narrow q4_0 weights — [#30646](https://github.com/ggml-org/llama.cpp/pull/30646)

`use_adreno_kernels()` gates the Adreno trans-weight GEMM path behind `ne0 >= 512 && ne1 >= 512` for E031 >= 38.11 compilers (older compilers use 128). As a result, every q4_0 weight with `ne1 < 512` falls back to `kernel_mul_mm_q4_0_f32_l4_lm`, which on Adreno 740 runs at ~5-11 GFLOPS while the trans-weight GEMM reaches ~300-500 GFLOPS on the same shapes — ~50x slower.

The affected weights are exactly the narrow ones: K/V projections (ne1 = kv_dim), per-layer embedding gates/projections (ne1 = n_embd_altup), and low-rank adapters (ne1 = rank). On gemma-4-E2B these run once per layer, and kernel profiling shows the `l4_lm` kernel accounts for **79 % of the prefill time** (`-DGGML_OPENCL_PROFILING=ON`: 4.4 s of 5.6 s per 256-token batch; 92 calls x ~48 ms).

This change lowers the threshold to 128 — the value the same function already uses for older compilers. The shape constraints for the transposed layout (ne0 % 32 == 0, ne1 % 4 == 0, ne2 == ne3 == 1, element count < 2^27) are still enforced by `enable_adreno_trans_weight()`.

Verified on Adreno 740 (E031.41.03.64), `llama-bench --device GPUOpenCL -ngl -1 -p 256 -n 64 -r 2`:

| model | pp256 before | pp256 after | Δpp | tg64 before | tg64 after |
|---|---|---|---|---|---|
| gemma-4-E2B | 45.9 | **205.3** | **+348 %** | 11.5 | 11.6 |
| gemma-4-E4B | 29.2 | **109.5** | **+275 %** | 7.8 | 7.9 |
| gemma-3-1b | 105.9 | **497.0** | **+369 %** | 16.7 | 17.1 |
| granite-350m | 137.8 | **1066.7** | **+674 %** | 48.8 | 47.9 |
| LFM2.5-230M (control) | 2006 | 2115 | +5 % | 107.6 | 107.3 |

Greedy 24-token generation is byte-identical between the two builds.

Evidence: `GGML_SCHED_DEBUG=2` shows only the q4_0 embedding gathers on CPU; the per-kernel profile (`cl_profiling.csv`) attributes 4.4 s of 5.6 s to `kernel_mul_mm_q4_0_f32_l4_lm` at ~10.8 ms per small matmul.

---

## 3. OpenCL: add q4_0 `o4` GEMV for long-vocab lm_head — [#30658](https://github.com/ggml-org/llama.cpp/pull/30658)

Adds a 4-output-per-work-item (`o4`) GEMV for q4_0 decode, mirroring the existing q4_K `o4` variant, for long-vocab lm_head / embed shapes.

**Motivation:** the q4_0 decode GEMV for `ne01 >= 32768` (lm_head / embed, e.g. 262144 rows on gemma-4) is the worst q4_0 GEMV shape: each work item computes only 2 output rows, so the per-block activation read + subgroup broadcasts are amortized over just 2 rows. The q4_K path already solves this with `kernel_gemv_noshuffle_q4_k_f32_o4` ("4-output-per-WI, long-vocab lm_head"); q4_0 had no equivalent.

**Change:**

- `kernel_gemv_noshuffle_q4_0_f32_o4`: 4 consecutive output rows per work item as two 2-row pairs (a, b) that share one activation read per k-block. The existing 2-row `dequantizeBlockAccum_ns_sgbroadcast_*` macros are reused unchanged, so each row's accumulation order — and therefore the result — is **byte-identical** to the base GEMV.
- Dispatch: decode only (`ne1 == 1`), `ne01 % 4 == 0 && ne01 >= 32768`, default on, opt-out `GGML_OPENCL_Q4_0_O4=0`. Launch `{CEIL_DIV(ne01/4,64)*64, 4, 1}`; tail quads past the last row are guarded.

**Benchmarks** (gemma models, Adreno 740 / SM8550), `llama-bench --device GPUOpenCL -ngl -1 -fa on -p 1 -n 64 -r 2`, interleaved A/B (base = same tree without the change):

| model | base tg64 | o4 tg64 | delta |
|---|---|---|---|
| gemma-4-E2B q4_0 | 11.47 | 12.97 | +13.1% |
| gemma-4-E4B qat Q4_0 | 7.90 | 8.71 | +10.3% |
| gemma-3-1b q4_0_s | 17.33 | 17.97 | +3.7% |

- Prefill unchanged: E2B pp256 205.4 → 206.9 (within noise)
- Greedy 24-token output byte-identical between o4 on/off

---

## Runtime defaults

All three changes are active by default in this fork — no extra flags are needed on A7X-class Adreno GPUs:

- f16-KV flash attention compiles and runs (the `q1_vec` crasher stays excluded for `f32_f16` programs).
- The trans-weight GEMM threshold for E031 >= 38.11 is 128, covering narrow q4_0 weights.
- The q4_0 `o4` GEMV is default-on for `ne01 >= 32768` decode; disable with `GGML_OPENCL_Q4_0_O4=0`.

## Building

Use the standard upstream build (see [README.upstream.md](README.upstream.md)); for an OpenCL build on device:

```bash
cmake -B build -DGGML_OPENCL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Attribution

This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). All credit for the base project and its documentation goes to the upstream authors; the fork tracks upstream `master` plus the three changes described above. Same license as upstream (MIT).
