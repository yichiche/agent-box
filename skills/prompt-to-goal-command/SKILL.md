---
name: prompt-to-goal-command
description: >-
  Rewrites informal prompts into goal-style agent commands: clear done-state,
  verifiable checks, scope, and stop/reporting rules. Use when the user
  references this skill, /goal-style workflows, long-horizon agents, or asks to
  turn a vague request into a command aligned with a stated goal.
disable-model-invocation: true
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

## 改寫流程（對著使用者原始 prompt 做）

1. **抽出真正的 Goal**：一句話寫「成功時世界長怎樣」（產物 + 行為），不是「怎麼做」。
2. **補 Acceptance**：列 3–7 條可勾選的完成條件；能測就寫測試/指令，不能測就寫具體人工檢查步驟。
3. **收斂 Scope**：若使用者沒給，主動補「只動哪些檔案/目錄」或「禁止事項」。
4. **選 Verify**：至少一個會在 goal 達成時應該**通過**的命令或檢查清單。
5. **定 Stop**：何時停止（達成 / 驗證失敗需人類介入 / 達到迭代上限），避免無限繞圈。
6. **輸出一條 Goal Command**：給代理「複製即用」的一塊文字（見下方模板）。

## 輸出模板（產給使用者的 command）

代理在套用本 skill 時，應輸出類似下列結構（可依專案改寫標題與指令）：

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

若使用者環境使用明確的 `/goal` 外掛或自訂 slash，可把首行改為其慣用觸發（例如 `/goal …`）但**內文仍須包含** Done、Verify、Scope、Stop。

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

- 本 skill **無法替代觀看原影片**的細節；若使用者要完全對齊影片中的工具設定，請其補充截圖或關鍵步驟，再把那些約束併入 **Scope / Verify**。
- 若專案有既有規範（commit、PR、lint），把對應指令寫進 **Verify**，goal command 會更穩。
