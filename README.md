# Strata 並行加速（`--batch`）

**繁體中文** · [English](lab/README.md)

這個分支只改一件事：**讓 [Strata](https://github.com/Niko1221/Strata) 同時處理多個請求時，總速度變快。**
單一請求的行為不變，MTP、QSA、專家全放 VRAM 都照舊。原版 Strata 的介紹和安裝方式請看
[原作者的 repo](https://github.com/Niko1221/Strata)。

| | 原版 0.1.43 | 這個分支 |
|---|---|---|
| 1 個請求 | 185 tok/s | 181 tok/s |
| 4 個請求同時跑（`--batch 4`），總速度 | 268 tok/s | **313 tok/s**（+17%） |
| 8 個請求同時跑（`--batch 8`），總速度 | 308 tok/s | **378 tok/s**（+23%） |

temperature 0 時，輸出跟原版**逐字相同**。兩個開關都沒開時，行為就是原版。

## 為什麼原版並行會變慢

Strata 並行時，每一輪會把所有請求的下一個字放進同一個「驗證窗口」（最多 8 行）一起算：

| | 每輪時間 | 每輪產出 |
|---|---|---|
| 單一請求 + MTP | 約 6.8 ms | 約 1.56 個字（寫程式時更多） |
| 原版並行，R 行 | 約 4.8 ms + **每行 2.4 ms** | R 個字 |

每行的 2.4 ms 裡，讀專家權重只佔約 0.42 ms，這部分已經達到頻寬上限。
其餘大多是**每個請求各自排隊執行的小型 GPU 運算**：attention 的打分、選取、計算，以及 GDN 的遞迴，都是一個請求做完才換下一個。

## 改了什麼

兩個環境變數，預設都關閉：

| 開關 | 作用 |
|---|---|
| `STRATA_BATCH_ATTN_ROWS=1` | attention 的打分、top-k、計算，**所有請求合併成一次執行**。每一行仍然讀自己那個請求的資料，算法完全相同。 |
| `STRATA_BATCH_GDN_GROUPS=1` | GDN 遞迴（窗口內與 commit 時）**所有請求合併成一次執行**。 |

改動的檔案：`src/kernels/cuda/{qsa_decode_attn,qsa_select,verify_kernels}.cu`、對應的 header、`src/core/verify.cpp`。

使用方式：在模型設定檔（`strata-<model>.json`）加上：

```json
"env": { "STRATA_BATCH_ATTN_ROWS": "1", "STRATA_BATCH_GDN_GROUPS": "1" }
```

## 測試環境

- NVIDIA RTX PRO 6000 Blackwell Workstation（96 GB）、driver 610.57.04、CUDA 13.0、Linux
- Strata 0.1.43（`7695859`）、Qwen3.8-Flash-Next IQ3_S，24,576 個專家全部放在 VRAM
- `--kv int8 --max-context 262144 --spec 4 --spec-min-p 0.5 --mtp --vision`，沒特別註明時 `--vram-reserve-mib 20480`
- 負載：用 [`bench_concurrency.py`](lab/bench_concurrency.py) 同時送 N 個串流請求。每個請求是不同的故事題目，所以不會共用前綴快取。每個請求輸出 1024 token，每組跑 2 次。

## 測試結果

### 每輪驗證時間（ms）

| 行數 | 原版 | 只合併 attention | attention + GDN |
|---|---|---|---|
| 2 | 9.57 | 9.02 | 8.82 |
| 3 | 11.95 | 10.83 | 10.47 |
| 4 | 14.30 | 12.62 | 12.12 |
| 6（`--batch 8`） | 19.25 | | 15.92 |
| 8（`--batch 8`） | 24.55 | | 19.91 |

每多一行的成本：原版約 2.4 ms，修改後約 1.7–1.9 ms。

### `--batch 4` 總速度（tok/s）

數據：[phase4](lab/results/phase4)、[phase5](lab/results/phase5)

| 同時請求數 | 原版 | 只合併 attention | attention + GDN |
|---|---|---|---|
| 1 | 185.6 | 181.7 | 181.2 |
| 2 | 202.4 | 214.4 | 220.3 |
| 3 | 242.6 | 267.8 | 277.0 |
| 4 | 268.5 | 302.4 | **312.9** |

### `--batch 8` 總速度（tok/s）

數據：[phase6](lab/results/phase6)。這組把預留降到 6144 MiB，讓專家仍然全部放得進 VRAM。

| 同時請求數 | 原版 | attention + GDN |
|---|---|---|
| 1 | 183.2 | 181.2 |
| 4 | 267.6 | 313.6 |
| 6 | 298.6 | 359.2 |
| 8 | 308.5 | **377.8** |

### 正確性

[`parity.py`](lab/parity.py) 同時送 4 個 temperature 0 的請求，[`compare_parity.py`](lab/compare_parity.py) 逐字比對修改版和原版的輸出。所有設定都是 `IDENTICAL`。

### 原版基準：p1 / p4 / p8，以及 `--batch-mtp`

數據：[production-p1-p4-p8](lab/results/production-p1-p4-p8/summary.txt)。原版 0.1.43，預留 20480 MiB。
p8 時只有 17,097 個專家放得進 VRAM，所以單一請求比較慢。

| 同時請求數 | p1 | p4 | p4 + batch-mtp | p8 | p8 + batch-mtp |
|---|---|---|---|---|---|
| 1 | 180.0 | 184.8 | 167.7 | 137.8 | 155.2 |
| 2 | 177.0 | 203.1 | 199.0 | 185.5 | 194.8 |
| 4 | 179.4 | 267.8 | 239.5 | 252.0 | 231.5 |
| 8 | 177.0 | 264.9 | 237.1 | 293.0 | 230.6 |

### 沒有效果的嘗試：並行時每個請求多猜幾個字

數據：[phase3a](lab/results/phase3a)、[phase3a2](lab/results/phase3a2)

第二個 commit 加了 `STRATA_BATCH_MTP_D`（預設 1，等於原版），讓每個請求在窗口裡用 MTP 多猜幾個字。
但並行時猜中率只有 **50–57%**，而每多一行要超過約 67% 才划算。

4 個同時請求的總速度：

| 設定 | tok/s |
|---|---|
| 不開 MTP 的 `--batch 4` | 268 |
| D=1 | 246 |
| D=3 | 199 |
| D=5 | 193 |

只留作參考。

### 記憶體

數據：[context-memory](lab/results/context-memory/summary.txt)

**並行的空間在啟動時就預留好了，跑起來不會再增加。** 4 個請求同時跑之前、之中、之後，都是 73,352 MiB。
每個位子都預留最長上下文的空間，所以上下文長度決定 p4 要多少記憶體：

| 設定 | 整張卡 | 1 個請求 | 4 個請求（輸出 512 token） |
|---|---|---|---|
| p1，262K | 59.9 GB | 186 | 170（排隊） |
| p4，64K | 61.1 GB | 181 | 286 |
| p4，128K | 65.7 GB | 181 | 291 |
| p4，262K | 約 74.9 GB | 184 | 約 309 |

除了 p1，其他都是修改版。專家 46.8 GiB 每一列都一樣。

## 重現

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build --target strata -j 16 && cp build/strata engine/strata
```

`lab/` 裡的腳本會在 port 8095 啟動測試用的 server（`lab-start.sh <並行數>`），跑完存到 `lab/results/`，再自動關掉（`lab-stop.sh`）。

路徑設定在 [`lab/env.sh`](lab/env.sh)。每台機器自己的路徑請寫在 `lab/env.local.sh`（不會上傳），例如 Python、`STRATA_DATA`、`CUDA_HOME`。如果有其他控制台在跑正式版，也可以在這裡設定，測試前後會自動停掉、再恢復。

| 腳本 | 測什麼 |
|---|---|
| `phase1.sh` | 未修改的版本，p1 對 p4 基準（其中 p1 的 57 tok/s 無效，當時 GPU 上還有另一個實例） |
| `phase2.sh` | `STRATA_DECODE_TIMING` / `STRATA_VERIFY_PROFILE` 各階段時間 |
| `phase3a.sh`、`phase3a2.sh` | 多猜幾個字的實驗 |
| `phase4.sh` | 只合併 attention 對原版，含逐字比對 |
| `phase5.sh` | attention + GDN，含逐字比對 |
| `phase6.sh` | `--batch 8` 原版對修改版 |
| `bench-dashboard.sh`、`context-memory.sh` | 透過 vllm_my 風格的控制台，用不同設定重啟後測試 |

## 限制與後續

- 只在一張卡、一個模型上測過（單張 RTX PRO 6000、IQ3_S、專家全在 VRAM）。沒測過多卡、專家放在 PCIe/CPU 的情況，也沒測過 AMD。
- 刻意沒有用多 stream。原版在 Linux open module 上關掉了 `STRATA_DF_BRANCH`（#905）。
- 還有兩部分是每個請求各自執行：conv commit 和 indexer append。下一步可以把它們也合併。
