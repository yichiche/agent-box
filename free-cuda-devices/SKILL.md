---
name: free-cuda-devices
description: >-
  Maps HIP / ROCR device indices to rocm-smi Concise "Device" rows and flags free vs
  occupied GPUs (VRAM% / GPU% thresholds). Fast path: run free_cuda_devices.py
  --table (one rocm-smi, no PyTorch). Optional: CUDA_VISIBLE_DEVICES export via
  torch+PCI. Use when setting HIP_VISIBLE_DEVICES, CUDA_VISIBLE_DEVICES, launching
  sglang/vLLM, or the user mentions 空閒 GPU、對照表、rocm-smi Concise、共用節點、MI300/MI355.
disable-model-invocation: true
---

# 空閒 GPU（ROCm / 共用節點）

**Canonical copy（權威副本）：** `$HOME/agent-box/free-cuda-devices/`。

`$HOME/agent-box/skills/` 在此機器上多為 **root 擁有**（一般使用者無法新增同層目錄），因此本 skill 放在 **agent-box 根目錄**下，與 `skills/` 內其他條目並列維護。若你希望路徑出現在 `skills/` 下，可由管理員建立目錄並改權限，或自行 `ln -s "$HOME/agent-box/free-cuda-devices" "$HOME/agent-box/skills/free-cuda-devices"`。

## 核心對照（先做這步，秒級）

在典型 **MI300 / MI355** 節點上，`rocm-smi` **Concise Info** 第一欄 **`Device`（0…N-1）** 通常就是你在 **`HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES`** 裡寫的**單顆實體 GPU 編號**（與 `GPU[n]` / `cardN` 同一套枚舉）。**不要**假設 `cuda:0` 永遠等於「表格第 0 列」的語意而不看實際負載。

### 指令（推薦）

```bash
python3 "$HOME/agent-box/free-cuda-devices/scripts/free_cuda_devices.py" --table
# 或簡寫
python3 "$HOME/agent-box/free-cuda-devices/scripts/free_cuda_devices.py" -t
```

- **不需 PyTorch**、約 **一次 `rocm-smi`**，適合代理與人類快速決策。
- 閾值（與腳本預設相同）：`FREE_CUDA_MAX_GPU_UTIL`（預設 `5`）、`FREE_CUDA_MAX_VRAM_PCT`（預設 `15`）。任一大於閾值 → 標成 `occupied`，否則 `free`（**僅建議**，最後仍由你判斷）。

### 若使用者已貼 `rocm-smi` 文字

代理應自行產出**同一格式**的 Markdown 表與逐行對照（不必再跑慢指令）：

1. 從 Concise 表讀每列：**Device**、**Node**、**VRAM%**、**GPU%**。
2. 輸出表格欄位：`| HIP (Device) | Node | VRAM% | GPU% | suggested |`。
3. 再輸出逐行：`HIP_VISIBLE_DEVICES=<Device>  →  rocm Device <Device> (Node <Node>)  →  free|occupied`。
4. **逗號清單**：`HIP_VISIBLE_DEVICES=a,b,c` 只露出這幾顆實體卡；程式內**邏輯** `cuda:0` 永遠對到清單裡的**第一顆**（即 `a`），不是實體 0。

### 注意

- 若環境變更（驅動、ROCm、`GPU_DEVICE_ORDINAL`、容器內可見裝置集合），**以當下 `rocm-smi` 為準**；換機後勿沿用舊口訣。
- Concise 解析依賴**英文欄位**；若 `rocm-smi` 本地化導致表頭非英文，改用 `--table` 腳本失敗時請貼原始輸出改用手動表。

---

## 進階：自動組 `CUDA_VISIBLE_DEVICES`（需 PyTorch）

當你需要**邏輯 `cuda:i` 順序**與實體卡的 PCI 嚴格對齊（例如多進程、或懷疑 Device 序與 torch 不一致）時，用**與 sglang 相同**的 Python 跑**不帶** `--table` 的腳本，並必要時設 `FREE_CUDA_PYTHON`：

```bash
FREE_CUDA_PYTHON=/path/to/venv/bin/python3 \
  python3 "$HOME/agent-box/free-cuda-devices/scripts/free_cuda_devices.py"
```

無法 import `torch` 或 PCI 對不上時，腳本**不會**輸出 `CUDA_VISIBLE_DEVICES=` 行；此時請用 **`--table`** 或手動對照。

---

## 代理（Agent）檢查清單

- [ ] 要挑 GPU 時**優先**執行上述 **`--table`**（或使用者已貼 `rocm-smi` 則直接手排表）。
- [ ] 表格與逐行說明使用 **HIP = Concise `Device`**；並提醒 **逗號清單 → 邏輯 `cuda:0` 對第一顆實體編號**。
- [ ] 僅在需要「可 export 的邏輯 CUDA 清單」時才要求工作環境的 PyTorch 跑無 `--table` 模式。
- [ ] 不要無腦建議 `CUDA_VISIBLE_DEVICES=0,1` 而不看當下負載與對照表。

## 腳本位置

- `$HOME/agent-box/free-cuda-devices/scripts/free_cuda_devices.py`（`--table` / `-t` 與預設模式）

依賴：`rocm-smi`；邏輯 CUDA 匯出需與 workload 相同的 `torch`。
