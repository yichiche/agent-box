---
name: prompt-to-goal-command
description: >-
  Rewrites informal prompts into goal-style agent commands (max 4000 chars):
  clear done-state, verifiable checks, scope, and stop/reporting rules. Use
  when the user references this skill, /goal-style workflows, long-horizon
  agents, or asks to turn a vague request into a command aligned with a stated
  goal.
disable-model-invocation: true
category: meta
---

# Prompt → Goal-Aligned Command

靈感來自 [Gary Chen — 我的 AI agent 連續跑了 27 個小時，/goal 功能怎麼用？](https://www.youtube.com/watch?v=PpeCur6fEXc) 的主題：**用「目標 + 如何判定完成」驅動代理**，而不是只堆疊零碎指令。以下流程把使用者的自然語言 prompt **改寫成一條（或一小段）適合與 goal 搭配的 command**。

## 核心觀念（與 /goal 搭配時最重要）

| 元素 | 說明 |
|------|------|
| **Done** | 用可觀察的結果描述「做完」：功能、檔案、測試、建置、清單勾選，避免「盡力」「優化」等無法判定語句。 |
| **Verify** | 每個 goal 至少綁一種**可自動或人工快速確認**的檢查：`npm test`、型別檢查、grep、腳本、或明列「Reviewer 勾選項」。 |
| **Scope** | 路徑/模組/不做的事（non-goals），避免長時程代理擴散到全庫。 |
| **Budget** | 迭代上限、時間、允許的風險操作（例如：可改哪些目錄、是否禁止 force push）。 |
| **Evidence** | 要求代理在階段結束簡短回報：做了什麼、如何驗證、若未完成下一步是什麼。 |
| **Length** | 整份 Goal Command **≤ 4000 字元**（含 markdown、程式碼 fence、換行）。多數 `/goal` 外掛會硬性拒絕超長輸入。 |

## 字元上限（/goal 硬性限制）

Cursor `/goal` 的 **Goal condition 欄位上限為 4000 字元**。超過會被拒（例如 `Goal condition is limited to 4000 characters (got NNNN)`）。

套用本 skill 時：

1. **最終可貼上的 Goal Command 區塊必須 ≤ 4000 字元**（含標題、清單、程式碼 fence 內文字）。
2. **輸出前自行計數**；若超過，先壓縮再給使用者，不要先給超長版再讓使用者手動刪。
3. **詳細表格、長篇診斷、範例對照** 放在 Goal Command **之外**（聊天說明），Goal 內只留可判定完成的精簡條款。
4. 撰寫時以 **≤ 3500 字元** 為目標，預留 buffer 給路徑、數字、環境變數等個案加長。

**壓縮手法（優先順序）**

- 合併 Acceptance 為較少條目（保留可勾選、可測）
- Verify 用短標題 + 一行指令；細節指向既有腳本（`run_*_perf.sh`、`perf_sweep.sh`）
- 刪重複 baseline 表、冗長 RCA、第二份 verify 區塊、裝飾用表格線
- Scope 用 `In:` / `Out:` 各一行；路徑過長用 `...` 省略中間段
- 數值門檻寫在 Acceptance，Verify 只寫怎麼跑

**不可為了縮短而拿掉**：Goal 一句話、Acceptance、Scope In/Out、至少一條 Verify、Budget/stop。

## 改寫流程（對著使用者原始 prompt 做）

1. **抽出真正的 Goal**：一句話寫「成功時世界長怎樣」（產物 + 行為），不是「怎麼做」。
2. **補 Acceptance**：列 3–7 條可勾選的完成條件；能測就寫測試/指令，不能測就寫具體人工檢查步驟。
3. **收斂 Scope**：若使用者沒給，主動補「只動哪些檔案/目錄」或「禁止事項」。
4. **選 Verify**：至少一個會在 goal 達成時應該**通過**的命令或檢查清單。
5. **定 Stop**：何時停止（達成 / 驗證失敗需人類介入 / 達到迭代上限），避免無限繞圈。
6. **輸出一條 Goal Command**：給代理「複製即用」的一塊文字（見下方模板）。
7. **檢查長度**：確認 ≤ 4000 字元；超標則依上一節壓縮後再輸出。

## 輸出模板（產給使用者的 command）

代理在套用本 skill 時，應輸出**單一、≤ 4000 字元**的區塊（可依專案改寫標題與指令）：

`````markdown
## Goal
[一句話可驗證的目標]

## Acceptance（完成即全部為真）
- [ ] …
- [ ] …

## Scope
- In: …
- Out / 禁止: …

## Verify（goal 達成時應通過）
```bash
# 例：建置與測試
…
```

## Budget & stop
- 最多 N 輪迭代 / 或時間上限 …
- 若 verify 失敗：停止並列出失敗原因與最小修復建議，不要擅自擴大範圍。

## Progress / evidence
每個主要階段結束簡述：變更摘要、如何驗證、尚餘項目。
`````

若使用者環境使用明確的 `/goal` 外掛或自訂 slash，可把首行改為其慣用觸發（例如 `/goal …`）但**內文仍須包含** Done、Verify、Scope、Stop，且**總長仍 ≤ 4000**。

## 範例：弱 prompt → Goal command

**弱**：「幫我 refactor 一下 auth 比較乾淨。」

**強（goal-aligned）**：

`````markdown
## Goal
重構 `src/auth/`：行為與公開 API 不變，重複邏輯收斂到單一模組，可讀性提升。

## Acceptance
- [ ] 現有 `auth` 相關單元測試全數通過；無測試則手動列出 3 個關鍵路徑並逐一手動驗證
- [ ] 無新增公開 breaking change（列出若有例外）
- [ ] `src/auth` 內圈複雜度下降（簡述或附 before/after 對照）

## Scope
- In: `src/auth/**`
- Out: 不改 DB schema、不改路由 URL

## Verify
```bash
npm test -- --testPathPattern=auth
npm run build
```

## Budget & stop
最多 2 輪大改；若測試失敗，停止並只修到測試綠為止，不順手改其他模組。
`````

## 注意

- **4000 字元是交付硬性約束**；過長會導致 `/goal` 無法貼上。必要時拆成「可貼上 /goal 的精簡版」+「聊天中的展開說明」兩部分。
- 本 skill **無法替代觀看原影片**的細節；若使用者要完全對齊影片中的工具設定，請其補充截圖或關鍵步驟，再把那些約束併入 **Scope / Verify**（仍須遵守字元上限）。
- 若專案有既有規範（commit、PR、lint），把對應指令寫進 **Verify**，goal command 會更穩。
