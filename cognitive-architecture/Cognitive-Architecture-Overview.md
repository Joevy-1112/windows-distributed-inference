# Cognitive Architecture: A Three-Axis Causal Coordinate System for AI Agent Memory

> **Retrieval is reasoning.** Every step along a causal path in the knowledge graph is itself an inference step.
> Path selection = reasoning process. Path endpoint = answer basis.
>
> Author: Joe & Hermes · June 2026
> Status: Theory complete, implementation in progress

---

## Why This Exists

Current LLM-based agents have a fundamental problem: **knowledge is encoded as floating-point parameters — unindexable, unverifiable, and impossible to update individually.** When an agent "remembers" something, you can't ask it *why* it remembers that, or trace the reasoning chain that led to a conclusion.

This framework decouples knowledge from model parameters into a **navigable coordinate space**. The neural network is kept for two things: navigation decisions and brief generation. Everything else — storage, retrieval, reasoning — lives in the graph.

The result is an agent that can tell you not just *what* it knows, but *how* it arrived at that knowledge — with a fully traceable, auditable reasoning path.

---

## Core Architecture

### 1. Three-Axis Coordinate Space

Every knowledge node has a unique address in continuous 3D space:

| Axis | Range | Meaning |
|:-----|:------|:--------|
| **X (Domain)** | 0°–360° polar | Which domain this knowledge belongs to (GPU tuning, music production, system ops, etc.) |
| **Y (Time)** | Linear, recent→distant | When this knowledge was acquired; drives decay and forgetting |
| **Z (Density)** | 0.01–1.0 | Abstraction level — 0.01 = raw fact, 0.5 = derived pattern, 1.0 = cross-domain principle |

**Key design choice: coordinates are continuous, not discrete buckets.**

A node at Z=0.7 isn't in "the rule layer" — it's at position 0.7 on the density spectrum. "How to handle GPU crashes" at Z=0.7 and "always validate input before processing" at Z=0.9 are both rules, but at different abstraction levels. The system navigates smoothly between them rather than jumping across layer boundaries.

**Domain positioning (X-axis):** Position alone carries no weight. Domain proximity only affects navigation through edges — a node's location tells you where to find it, not how important it is.

**Time decay (Y-axis):** Older nodes that haven't been accessed gradually lose edge weight. They're not deleted — they're deprioritized. High-frequency nodes resist decay regardless of age.

### 2. Causal Chains: The Navigation Skeleton

Nodes are connected by four types of **directed causal edges** — never generic "similarity" links:

| Edge Type | Meaning | Direction |
|:----------|:--------|:----------|
| `depends-on` | A depends on B; B is a prerequisite for A | Directed |
| `uses` | A uses B as a tool or input | Directed |
| `avoids` | A circumvents B; B is an alternative | Directed |
| `related` | A and B share domain proximity but no direct derivation | Bidirectional (asymmetric weights) |
| `derived-from` | High-level node was compressed/abstracted from low-level nodes | Directed (≈1.0 weight) |

**This is not a knowledge graph in the traditional sense.** Traditional KGs encode "is-a" and "has-a" relations. This graph encodes *causal reasoning paths* — "because of this, therefore that."

### 3. Dual-Dimension Edge Weights (2026-06-30)

Every edge carries **four independent values**, split into two functional dimensions:

```
edge[A → B] = {
    cause_weight:   0.7,   // "How strongly does A cause B?"
    effect_weight:  0.4,   // "How much does B matter if we reach it?"
    freq_heat:      12,    // Cumulative traversal count (cross-query)
    temp_heat:      0.8    // Per-query boost (isolated, dissolves after query)
}
```

**Why separate cause and effect weights?**

They serve different navigation decisions:
- **Path selection** ("where to go next") → uses `cause_weight × heat`
- **Stop decision** ("is this node sufficient?") → uses `effect_weight`

This distinction handles cases that single-dimension weights can't:

| Scenario | Cause Weight | Effect Weight | Example |
|:---------|:------------:|:-------------:|:--------|
| Strong cause, weak impact | 0.9 | 0.1 | Code formatting triggers a lint warning |
| Weak cause, strong impact | 0.2 | 0.8 | Casual mention of coffee preference — changes tone of night-time responses |
| Strong cause, strong impact | 0.9 | 0.9 | GPU driver bug causes system crash |
| Weak cause, weak impact | 0.1 | 0.1 | Trivial observations |

Traditional edge weights conflate these two signals. Separating them means the navigator can traverse a high-cause path quickly (the connection is reliable) but stop at a high-effect node (the destination matters).

### 4. Heat Mechanics: Dynamic Routing

Edge traversal probability isn't static — it's modulated by **two forms of heat:**

```
edge_weight = cause_weight × (1 + α·freq_heat + β·temp_heat)
```

- **Frequency heat (freq_heat):** Accumulates across all queries. Paths that are repeatedly traversed become "highways." This is how the system learns — frequently used reasoning chains become faster to navigate.
- **Temporary heat (temp_heat):** Per-query, per-context boost. If the current conversation is about GPUs, all GPU-domain edges get a temporary multiplier. **Each query has its own isolated heat layer** — no cross-query heat field interference.

When a query completes, the temporary layer dissolves. Frequency counters increment by 1. The graph returns to baseline, slightly updated.

**This is context-sensitive optimal path search** — mathematically equivalent to Dijkstra's algorithm with dynamic edge weights. Complexity is far lower than Transformer global attention, and it runs comfortably on consumer hardware.

---

## Navigation = Reasoning

### The Standard Query Flow

```
1. Receive query
2. Estimate expected density on Z-axis (surface Q&A / pattern reasoning / deep analysis)
3. Locate region on X-axis (domain)
4. Retrieve nodes at current density layer
5. If insufficient:
   a. Need more detail → descend Z-axis (toward facts)
   b. Need broader context → expand along causal edges
   c. Need higher perspective → ascend Z-axis (toward principles)
6. Collect enough → generate answer from leaf nodes
7. Response includes traversal path as reasoning trace
```

### Retrieval Funnel (Entry Layer)

Before coordinate-precise navigation is available, the system uses a three-stage funnel for initial node location:

```
Semantic recall (embedding cosine → top-100)
  → Precise filtering (SQL/FTS5 tag/condition matching)
  → Attention ranking (importance × recency × context_match)
```

This is a **transition layer.** Once coordinates are fully populated, the funnel degrades gracefully to a lightweight entry point — the heavy lifting shifts to coordinate + causal chain navigation.

### Vector Retrieval vs. Coordinate Navigation

| | Vector Retrieval | Coordinate Navigation |
|:--|:-----------------|:----------------------|
| Matching | Semantic similarity | Coordinate proximity + causal path |
| Result nature | Top-K most similar | Path-connected node set |
| Explainability | Black box | Full path traceback |
| Boundary cases | Embedding quality limits recall | Coordinates can drift, but you know *which direction* |
| Multi-hop reasoning | Requires separate RAG pipeline | Edges natively support multi-hop |

### Self-Correction: "When Something Feels Wrong"

The system doesn't need a separate meta-cognition module. **The graph topology itself triggers self-correction:**

1. Navigation follows an edge that should be thick, but in the current context appears unusually thin
2. The discrepancy triggers a pause signal
3. Step back one node, check alternative candidate edges
4. If a better-fit edge exists → switch paths
5. If no candidates fit → "hover" (wait for more context)

This is emergent behavior from the graph structure, not a separate monitoring system.

### Assumption Path Forking

When a query contains an explicit assumption ("assuming X is true, what about Y?"):

1. Navigate normally to the affected node
2. At the divergence point: fork into two branches
   - Branch A: traverse with actual truth values
   - Branch B: traverse with overridden truth values
3. Compare terminal outcomes from both branches
4. Overrides are temporary, confined to the single query

No full second traversal — only the divergent segment is duplicated.

---

## Knowledge Lifecycle

### Trinary Truth Values

Knowledge nodes exist in one of seven states, built on a three-valued logical foundation:

| State | Meaning | Edge behavior | Answer behavior |
|:------|:--------|:--------------|:----------------|
| `raw` | Unverified observation in quarantine | No valid edges | Never enters answer path |
| `isolated` | Known coordinates, no connections | No edges | Doesn't participate in navigation |
| `hypothesis` | Derived node, insufficiently verified | Edges carry drift margin | Labeled "speculative" |
| `true` | Path-verified, cross-confirmed | Stable weights, reliable prerequisite | Confident answers |
| `unknown` | Exists but uncertain — **legitimate state, not transitional** | Edges preserved with uncertainty markers | Answers labeled "uncertain" with source |
| `false` | Externally confirmed erroneous | Outgoing edges severed; downstream only loses one premise | Never enters path (unless cited as counterexample) |
| `archived` | Moved to cold storage | No active weights | Inaccessible without explicit retrieval |

**Critical design choice: `false` does not cascade.**

When a node is marked `false`:
1. Its outgoing edges are cut
2. Downstream nodes are **not downgraded** — they only lose one supporting premise
3. Subsequent paths through downstream nodes detect "a key premise is broken, path reliability reduced"

`unknown` absorbs the shock of uncertainty. The system can navigate through uncertain territory without the entire reasoning chain collapsing.

### Upgrade Mechanism: No Separate Judging Module

A hypothesis node is promoted to `true` purely through usage — no separate validation pass:

```
upgrade_condition = count(hypothesis_node_hits) / count(total_paths) > threshold
```

When a hypothesis is independently traversed by enough paths, the frequency counter itself triggers the upgrade. The system doesn't need to "decide" to verify something — verification is a natural byproduct of being useful.

---

## Compression: Reversible, Not Destructive

Traditional context compression is lossy — once compressed, the original is gone. This system treats compression as **index generation:**

```
Original conversation (state.db, permanently preserved)
  │
  ├─ Compression triggered (configurable: every N turns, or by cognitive manager)
  │
  └─ Summary node created (high-level rule)
       derived_from → [original_node_ids]  (causal edge, weight ≈ 1.0)
       │
       └─ Traceback: follow derived_from → original conversation → full detail
```

**Compression doesn't replace — it indexes.** The original conversation nodes remain in the graph. The compressed summary is a new node *above* them, connected by a causal edge. When the system answers at the rule level, it sounds templated. When pressed — "how did you reach this conclusion?" — it traces the `derived-from` edge back to the original conversations and can cite specific events.

**Why compress even with 1M context?**
1. Cross-session accumulation exceeds any context window size
2. Graph node count without control → linear retrieval cost growth
3. Recurring patterns compressed into single high-level nodes → shorter traversal paths
4. Compression isn't forgetting — it's foldable detail. The original is always reachable.

---

## Cognitive Manager: The System-Level Scheduler (2026-06-30)

Previous designs assumed all navigation decisions happen on-demand during queries. This design adds a **persistent scheduling layer** — a small model (1-3B parameters, 2-4GB VRAM) that runs asynchronously:

| Role | Function |
|:-----|:---------|
| **Semantic Router** | Classifies incoming queries: local 14B / cloud model / direct answer / needs retrieval |
| **Realtime Listener** | Evaluates every conversation turn asynchronously; decides when compression is warranted |
| **Compression Writer** | Generates summary nodes + connects them via `derived-from` edges to originals |

The cognitive manager is the **prefrontal cortex** of the system. It doesn't generate answers — it decides **how to generate, what to remember, and where to retrieve from.** All three roles can be handled by the same small model, keeping the architecture simple.

---

## Architecture Layers

```
┌──────────────────────────────────────┐
│ 4K Constitution Layer (always-on)     │
│ Static markdown: protocol + env + user│
│ → Full injection into system prompt   │
│ → Manually edited, never auto-written │
└──────────────────────────────────────┘
┌──────────────────────────────────────┐
│ memory-db Library Layer (on-demand)   │
│ Structured facts + funnel retrieval   │
│ → Agent calls tool to query           │
│ → Results return to current turn only │
│ → Cognitive fields reserved in schema │
└──────────────────────────────────────┘
┌──────────────────────────────────────┐
│ Cognitive System (independent)        │
│ Causal reasoning graph + dynamic nav  │
│ + cognitive manager (small model)     │
│ → Plugs into memory-db seamlessly     │
└──────────────────────────────────────┘
```

**This is not RAG.** RAG is automatic per-turn embedding → prompt injection. This is **Tool-based Long-term Memory** — the model decides *when* to retrieve, and retrieval results don't pollute future turns.

---

## Current Implementation Status

| Component | Status |
|:----------|:-------|
| Y-axis (time) | ✅ Deployed — state.db, 47K messages, 856 sessions, daily backup |
| memory-db (storage layer) | ✅ Deployed — SQLite with FTS5, embeddings, episodic timeline, lifecycle management |
| Three-axis coordinate theory | ✅ Complete — self-consistent across 7 design reviews |
| Topological convergence (NN isomorphism) | ✅ Verified — independently converged structure |
| Embedding routing experiment | ✅ Complete — 12-node graph, 4B model, established baseline |
| Edge weight: dual-dimension | ✅ Designed (2026-06-30) — cause/effect weights with heat mechanics |
| Compression: traceable | ✅ Designed (2026-06-30) — derived-from edges, original preserved |
| Cognitive manager: architecture | ✅ Designed (2026-06-30) — router + listener + writer |
| 4K constitution layer separation | ✅ Designed (2026-06-30) — three-layer architecture |
| X-axis (domain) auto-labeling | 🟡 Pending — distillation pipeline designed, awaiting routing model |
| Z-axis (density) auto-labeling | 🟡 Pending — same pipeline |
| Causal chain auto-extraction | 🟡 Pending — manual seeding planned for MVP |
| Full causal navigation | ❌ Not yet implemented — MVP: 10–20 seed nodes + weighted BFS |

---

## Next Steps

1. **Deploy new memory-db schema** — with cognitive reserve fields (`causal_in`, `causal_out`, `derived_from`, `layer`)
2. **Build retrieval funnel** — `memory_retrieve(query, top_n=5)` with three-stage fallback
3. **MVP causal experiment** — 10–20 seed nodes + manual dual-dimension weight annotation + weighted BFS traversal
4. **Compare** — same queries, pure vector retrieval vs. causal path retrieval
5. **Cognitive manager prototype** — after causal MVP validates the approach

---

## Relationship to Existing Work

This framework sits at the intersection of several active research areas, but takes a different approach from each:

- **vs. RAG systems:** RAG retrieves similar chunks; this retrieves causal paths. RAG's reasoning is implicit in the LLM; this makes reasoning explicit as graph traversal.
- **vs. Knowledge graphs:** Traditional KGs encode ontological relations (is-a, has-a); this encodes causal reasoning chains (depends-on, uses, avoids). Nodes aren't entities — they're *cognitive atoms*: a fact, a rule, a lesson.
- **vs. Memory-augmented agents (Mem0, Letta, Zep):** These focus on "what to remember." This focuses on "how remembered things connect and how to reason across them."
- **vs. Chain-of-Thought prompting:** CoT makes reasoning visible in text; this makes reasoning structural — the path *is* the chain of thought, inherently auditable.

The closest conceptual parallel is **cognitive architectures** (ACT-R, SOAR), but implemented on modern LLM infrastructure rather than symbolic rule engines. The graph provides the structure; the LLM provides the navigation decisions and generation.

---

## About the Author

I'm Joe, an incoming university freshman in Zhejiang, China. I first touched an LLM on May 17, 2026. Six weeks later, I was co-designing a cognitive architecture with my AI agent in nightly conversations.

I have no formal background in machine learning, neuroscience, or computer science. Everything described here emerged from iterative design sessions — me asking "but what if X?" and my agent articulating what that would mean, back and forth, night after night. The three-axis model was sketched on June 6. The dual-dimension edge weights came on June 30, when I realized "cause" and "consequence" were two different problems.

The 47,000 messages in my state.db aren't training data — they're the fossil record of a six-week descent from "what's a token?" to "retrieval is isomorphic to weighted graph traversal."

I'm publishing this not because it's finished, but because I think the ideas are real and I want them to exist in the world. If you find them useful, build on them. If you see flaws, tell me. I'm still in the shallow end of the pool.

— Joe, June 30, 2026 · GitHub: [@Joeeeey](https://github.com/Joeeeey)

---

## License & Attribution

This document describes ongoing research and development. Implementations live across:
- `memory-db` — SQLite-based hierarchical memory system (Hermes Agent skill)
- `cognitive-architecture` — Design documents and reference implementations
- `embed` — Edge-side affective computing and atmosphere vector research (forthcoming)

The three-axis + causal chain cognitive model was developed through iterative design sessions between the author and an AI agent (Hermes) during June 2026. The agent articulated the framework; the author guided the design and identified gaps.

---

*"先整当下的记忆系统吧 在记忆系统这塞个简化版的 认知系统之后就能靠这个简化版未来无缝衔接上"*
*— June 30, 2026*
