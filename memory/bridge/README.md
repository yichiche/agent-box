# Cross-Container Claude Code Bridge

Host（Cursor / 本機 Claude Code）與 Container（GPU 內 Claude Code）透過 **共享 volume** 溝通。

## 為什麼需要兩層

| 機制 | 用途 | 限制 |
|---|---|---|
| **Native Remote Control** (`/rc`, `claude --remote-control`) | 手機 / claude.ai/code 即時對話 | 需 claude.ai `/login`；**API key / AMD gateway 不支援** |
| **File bus**（本目錄） | Host ↔ Container 架構討論、狀態同步 | 非即時；需雙方讀寫檔案 |

你的 container 用 `ANTHROPIC_BASE_URL=llm-api.amd.com` → **主要靠 file bus**；若 container 另外 `/login` claude.ai，可疊加 native RC。

## 目錄

```
remote/
├── README.md           # 本文件
├── STATUS.md           # 最新快照（雙向讀）
├── INBOX.md            # Host → Container 待辦/問題
├── OUTBOX.md           # Container → Host 回報/問題
├── sessions/           # 結構化 session 狀態 (YAML)
└── scripts/
    ├── remote_snapshot.sh   # 收集環境寫入 STATUS
    └── remote_watch.sh      # 輪詢 INBOX 變化（可選）
```

## 共享前提（run_docker.sh 已有）

```bash
-v "$HOME/.claude:/root/.claude"    # session id、teams、history 共用
-v "$HOME:/home/yichiche/"           # agent-box/memory 共用
```

Container 內路徑：`/home/yichiche/agent-box/memory/bridge/`
Host 路徑：     `$HOME/agent-box/memory/bridge/`

## 協議

### 1. 更新狀態（Container 端，任務開始/結束）

```bash
bash ~/agent-box/memory/bridge/scripts/remote_snapshot.sh \
  --role container \
  --task "PR26858 fused MoE decode profiling" \
  --note "conc4 decode xlsx ready for compare"
```

或 skill：`/remote-status update ...`

### 2. Host 發訊息給 Container

編輯 `INBOX.md` 最上方加一則（或 `/remote-msg to-container ...`）：

```markdown
## [2026-06-27 12:00 host] @container
請比較 0618 vs 0625 的 analysis_decode.xlsx，focus layer17 MoE。
- 路徑: ~/qwen3.5-mxfp4/0605_TP2/IL8k_IL1k/conc4/0625/pr28666/...
- 狀態: pending
```

Container Claude Code 開場讀 `INBOX.md` + `STATUS.md`。

### 3. Container 回報 Host

寫入 `OUTBOX.md`：

```markdown
## [2026-06-27 14:30 container] @host
Decode 瓶頸在 fused_moe dispatch，非 gemm。建議看 pr28658 trace。
- 狀態: done
```

### 4. 對應 Native Remote Control

Container 內若已 `claude /login`（非 AMD key）：

```bash
cd /sgl-workspace/sglang
claude --remote-control "MI355 PR26858 $(hostname)"
# 或 session 內: /remote-control PR26858
```

把 RC session URL 寫進 `STATUS.md` 的 `remote_control_url` 欄，Host 可直接開 claude.ai/code 連同一 session。

## Agent 開場檢查清單

**Container Claude Code 每次啟動：**
1. Read `memory/bridge/INBOX.md` — 有無 `@container` pending
2. Read `memory/bridge/STATUS.md` — 自己是否最新 owner
3. Run `remote_snapshot.sh --role container` 更新狀態

**Host Cursor / Claude Code：**
1. Read `memory/bridge/STATUS.md` + `OUTBOX.md`
2. Read `~/.claude/sessions/*.json` — 找 `status: busy` 的 container session
3. 若要即時對話且 RC 可用 → 開 `remote_control_url`

## 與 memory vault 整合

- 穩定發現 → `/memory-capture` 寫入 `memory/gotchas/`
- Session 摘要 → `memory/journal/YYYY-MM-DD.md`
- 週期整理 → `/memory-consolidate`
