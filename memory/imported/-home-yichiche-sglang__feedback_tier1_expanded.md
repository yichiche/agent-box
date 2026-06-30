---
name: tier1-expanded-criteria
description: "Tier-1 fusion includes custom Triton kernels, not just existing aiter ops"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7f1f7c9f-9a4f-4054-bf24-ebb976e7c25f
---

Tier-1 fusion opportunities include both existing aiter ops AND custom Triton kernels that can be written to fuse multiple elementwise/small ops. No aiter modifications required for either.

**Why:** Limiting to existing aiter ops misses many practical fusion opportunities that are straightforward Triton work.

**How to apply:** In /kernel-fusion-pipeline and /compare-kernels, classify as Tier-1 any fusion where we can write a Triton kernel or use an existing aiter op — no C++/HIP kernel work needed. Related: [[parallel-subagents]]
