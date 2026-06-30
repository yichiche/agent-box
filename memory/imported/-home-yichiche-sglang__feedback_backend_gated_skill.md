---
name: Always apply backend-gated-changes skill
description: When modifying SGLang model/layer code with aiter/HIP/gfx95 optimizations, always follow the rules in /home/yichiche/.cursor/skills/sglang-backend-gated-changes/SKILL.md
type: feedback
originSessionId: 95067461-5030-42f3-8b5b-e74bd8c3ed4c
---
Always read and follow `/home/yichiche/.cursor/skills/sglang-backend-gated-changes/SKILL.md` when modifying code in this project, especially model files (deepseek_v4.py, deepseek_v2.py) and layer files (communicator.py, linear.py, fp8.py).

**Why:** User established strict coding conventions for backend-gated changes to keep diffs minimal and common paths untouched.

**How to apply:**
- Use `_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip` pattern for flag computation
- Use explicit flag checks (`if _use_aiter and _is_gfx95_supported:`) at branch sites — no derived `_use_feature` booleans
- Inline `x_quant if x_quant is not None else x` at the linear call sites (e.g. `self.wq_a(x_quant if x_quant is not None else x)`) — no new variable, `x` never reassigned, compressor/indexer untouched
- Thread data via optional kwargs (`x_quant=None`), never instance state (`self._x_quant`)
- All aiter imports inside `if _use_aiter:` gates
