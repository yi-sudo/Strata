# Faster `--batch` decode: one launch per step for all slots

[繁體中文摘要](#繁體中文摘要)

This branch keeps everything Strata does for a single request (MTP drafts, the QSA attention, the VRAM expert
cache) and makes **parallel requests (`--batch N` / `"parallel": N`) faster**. It contains an opt-in kernel patch and
the scripts and raw results of every measurement behind it.

| | upstream 0.1.43 | this branch |
|---|---|---|
| 1 request | 185 tok/s | 181 tok/s |
| 4 requests, `--batch 4`, total | 268 tok/s | **313 tok/s** (+17%) |
| 8 requests, `--batch 8`, total | 308 tok/s | **378 tok/s** (+23%) |

Temperature-0 outputs are **byte-identical** to upstream. With neither env var set, the behaviour is upstream's.

## Setup

- NVIDIA RTX PRO 6000 Blackwell Workstation (96 GB), driver 610.57.04, CUDA 13.0, Linux
- Strata 0.1.43 (`7695859`), Qwen3.8-Flash-Next IQ3_S, all 24,576 experts resident in VRAM
- `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp ... --vision`, `--vram-reserve-mib 20480` unless noted
- Load: [`bench_concurrency.py`](bench_concurrency.py), N streaming clients at the same time, each a different story
  prompt (no shared prefix), 1024 tokens, `reasoning_effort: none`, 2 runs per point

## Why the per-request speed drops with parallel requests

Decoding one token means reading the active experts from VRAM. One request at a time already uses that bandwidth
well, and MTP drafts add more tokens per window. With N slots, Strata verifies all slots' rows in one window
(≤ 8 rows, `kVerifyMaxT`):

| | cost per window | tokens per window |
|---|---|---|
| solo + MTP | ~6.8 ms | ~1.56 (stories), more on code |
| upstream `--batch`, R rows | ~4.8 ms + **~2.4 ms per row** | R |

Only ~0.42 ms of each row is MoE expert reads (already at the bandwidth floor). Most of the rest came from **per-slot
serial launches**: for every row, QSA block scoring, top-k and attention each ran as separate one-row launches,
and the GDN recurrence (both in the window and in the commit) ran once per slot.

## The patch

Two env vars, both off by default:

| env | what it does |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | QSA block scores, top-k and decode attention for **all rows in one launch each**. Each row reads its own slot's indexer keys and K/V pools, passed by value as kernel params, so the CUDA graph captured per row layout stays valid. Falls back to per-row when a slot streams KV (`kv_mode 1`). |
| `STRATA_BATCH_GDN_GROUPS=1` | GDN recurrence in the window, and the 1-row commit, for **all slots in one launch** (`grid.y` = slot group). Conv commits stay per slot. |

Code: commit `Batch decode: one launch for all rows...`. Files: `src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`,
their headers, and `src/core/verify.cpp`.

Enable it with `serve/server.py` through the model config's `env` block:

```json
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

## Results

### Verify window time (ms, from the engine's `strata batch:` stats)

| rows | upstream | attn rows | attn + gdn |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6 (`--batch 8`) | 19.25 | | 15.92 |
| 8 (`--batch 8`) | 24.55 | | 19.91 |

Cost per extra row: ~2.4 ms upstream, ~1.7–1.9 ms with the patch.

### `--batch 4`, total tok/s ([phase4](results/phase4), [phase5](results/phase5))

| clients | upstream | attn rows | attn + gdn |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

### `--batch 8`, reserve 6144 MiB so the experts still all fit ([phase6](results/phase6))

| clients | upstream | attn + gdn |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

### Correctness

[`parity.py`](parity.py) sends 4 concurrent temperature-0 prompts and [`compare_parity.py`](compare_parity.py)
compares the outputs byte by byte against upstream. All `IDENTICAL` for attn rows, attn + gdn, and the MTP
experiment below.

### Upstream baseline: `--batch` and `--batch-mtp` ([production-p1-p4-p8](results/production-p1-p4-p8/summary.txt))

Total tok/s, upstream 0.1.43, reserve 20480 MiB. With `--batch 8` only 17,097 of the 24,576 experts fit in VRAM,
which is why one request is slower there.

| clients | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### Experiment that did not pay: more MTP drafts per slot ([phase3a](results/phase3a), [phase3a2](results/phase3a2))

The second commit (`STRATA_BATCH_MTP_D`, default 1 = upstream) lets each slot chain up to D MTP drafts into the
window. In batch mode the drafts are accepted only **~50–57%** of the time. With a window of ~4.8 ms + ~2.4 ms per
row, an extra row only pays off above ~67%. With 4 clients: D=1 246, D=3 199, D=5 193 tok/s, against 268 for plain
`--batch 4`. Kept for reference only.

### Memory: `--batch` reserves its slots at startup ([context-memory](results/context-memory/summary.txt))

VRAM does not grow while requests run: it was 73,352 MiB before, during and after 4 concurrent requests. Each slot
reserves room for the full `--max-context`, so the context size sets what `--batch 4` costs:

| setting | whole GPU | 1 client | 4 clients (512 tokens) |
|---|---|---|---|
| p1, 262144 | 59.9 GB | 186 | 170 (queued) |
| p4, 65536 | 61.1 GB | 181 | 286 |
| p4, 131072 | 65.7 GB | 181 | 291 |
| p4, 262144 | ~74.9 GB | 184 | ~309 |

All rows use the patched engine except p1. Experts (46.8 GiB) are the same in every row.

## Reproduce

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build --target strata -j 16 && cp build/strata engine/strata
```

The lab scripts start their own server on port 8095 (`lab-start.sh <parallel>`), run the load, save to `results/`,
and stop again (`lab-stop.sh`). Paths are set in [`env.sh`](env.sh). Put machine-specific values in an untracked
`env.local.sh` (Python, `STRATA_DATA`, `CUDA_HOME`, and optionally a dashboard `DASH` whose production instance
is stopped and restarted around the run).

| script | measures |
|---|---|
| `phase1.sh` | lab build without changes, p1 vs p4 baseline (its p1 run, 57 tok/s, is invalid: a second instance was still on the GPU) |
| `phase2.sh` | `STRATA_DECODE_TIMING` / `STRATA_VERIFY_PROFILE` per-stage window time |
| `phase3a.sh`, `phase3a2.sh` | `STRATA_BATCH_MTP_D` multi-draft experiment |
| `phase4.sh` | `STRATA_BATCH_ATTN_ROWS` vs upstream, with parity |
| `phase5.sh` | attn rows + `STRATA_BATCH_GDN_GROUPS`, with parity |
| `phase6.sh` | `--batch 8` upstream vs patch |
| `bench-dashboard.sh`, `context-memory.sh` | restart through a vllm_my-style dashboard at several settings |

`results/` keeps the raw JSON and engine logs of each run (local paths replaced by `~/...`).

## Limits and next steps

- Tested on one GPU and one model (single RTX PRO 6000, IQ3_S, all experts in VRAM). Not tested with multi-GPU,
  experts on PCIe/CPU, or AMD.
- Multi-stream branches were avoided on purpose (upstream disables `STRATA_DF_BRANCH` on Linux open modules, #905).
- Still per slot: the conv commits and the indexer appends. Grouping them is the next obvious step.

---

## 繁體中文摘要

**目標：** 保留 Strata 單一請求的優點（MTP、QSA、專家全放 VRAM），同時讓多個請求並行時的總速度更接近 vLLM。

**原因：** 並行時，上游 Strata 每一輪會替每個請求（slot）分開執行 attention 和 GDN 的 GPU 運算，請求越多，等待的時間越長。

**修改：** 把所有 slot 合併成一次執行，共兩個開關，預設都關閉：
- `STRATA_BATCH_ATTN_ROWS=1`：attention 合併
- `STRATA_BATCH_GDN_GROUPS=1`：GDN 合併

**結果：** temperature 0 時輸出與上游逐字相同。

| | 上游 | 修改後 |
|---|---|---|
| 1 個請求 | 185 | 181 |
| 4 個同時（p4） | 268 | **313** |
| 8 個同時（p8） | 308 | **378** |

**沒有效果的嘗試：** 並行時讓每個請求用 MTP 多猜幾個字，命中率只有 50–57%，不划算，所以預設關閉。

**記憶體：** 每個並行位子會在啟動時就預留最長上下文的空間，跑起來不會再增加。p4 搭配 64K 上下文約 61GB，搭配 262K 約 75GB。
