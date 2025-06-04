Below is a playbook-style checklist you can apply when you design or refactor a **Mixture-of-Agents (MoA) coding architecture**.  I distilled the core design principles from Wang et al.’s 2024 paper “*Mixture-of-Agents Enhances Large Language Model Capabilities*” and cross-checked them against newer multi-agent planning and survey literature published since then.  Citations appear after each item so you can dig deeper if needed.

---

### 1. Organise agents in **explicit layers**

*Why* Layering makes information flow and compute budgets predictable.
*How* Keep the first layer parallel (diverse candidate generation), and every next layer sequential as *refinement* or *voting* steps.  Each agent in layer *k + 1* receives the **entire set of answers** from layer *k* and can therefore edit, merge or discard them.  This is the central mechanism in MoA and explains its large quality jump. ([arxiv.org][1], [arxiv.org][2])

### 2. Pick **heterogeneous models** for early layers, **homogeneous models** for late layers

Different base LLMs spark diversity; using the *same* stronger model in the refinement layer keeps the final voice consistent.  Together.ai’s MoA implementation follows exactly that pattern. ([docs.together.ai][3])

### 3. Use **role-conditioned prompts** instead of hard-coding pipelines

Give each agent a concise *role card* (“Planner”, “Critic”, “Rewrite-for-clarity”, etc.) so you can swap models without changing code.  This idea is reinforced by the Modular Agentic Planner (MAP) paper, where modules are simply role-conditioned LLM calls. ([arxiv.org][4])

### 4. **Concatenate, then summarise** context between layers

The MoA paper shows that passing *all* previous outputs verbatim works—but only up to \~8 k tokens per layer before hitting context limits.  A pragmatic rule: concatenate everything, measure token length, and if you exceed 75 % of your model’s context window, auto-summarise the longest responses before forwarding. ([arxiv.org][1])

### 5. Run early layers **fully asynchronous**

Because candidate generation is embarrassingly parallel, launch N background requests and stream back as soon as each finishes.  Reserve synchronous (blocking) execution for the final refinement layer so you keep latency predictable.  Several open-source MoA repos follow this pattern. ([medium.com][5])

### 6. Introduce a lightweight **gating function** instead of majority voting

Once you have numeric scores (model log-probability, external judges like MT-Bench, or rule-based heuristics), train a tiny decision-tree or linear scorer that picks *one* answer or a blend.  Wang et al. report that a learnable gate beats naive majority by 3-4 % on AlpacaEval. ([arxiv.org][1])

### 7. Keep a **shared vector-store memory** keyed by (task-id, layer-id)

Agents often rediscover the same sub-answers.  Cache every intermediate output in a vector DB and attach relevant embeddings to the prompt of later agents.  Recent survey work on collaborative mechanisms highlights long-term memory as the biggest gap in many MoA demos. ([researchgate.net][6])

### 8. Make the **orchestrator stateless and declarative**

Describe the whole mixture in a JSON/YAML spec (`layers: [...agents...]`, `routing_rules`, `timeout_ms`).
Your runner simply interprets the spec, so you can A/B‐test new mixtures by changing the document, not the codebase.  Large-scale MoSA experiments adopt a similar declarative job file. ([arxiv.org][7])

### 9. Add a “**critic-of-critics**” safety pass

Before returning the final answer, run a compact safety checker (small model or rule-engine) that refuses, sanitises or re-prompts if the mixture produced disallowed content.  The 6G-agent paper emphasises that specialised domains with strict correctness needs benefit greatly from this bolt-on layer. ([arxiv.org][8])

### 10. Instrument everything with **per-layer telemetry**

Log tokens, latency, cost and score per agent, not just globally.  In MoA experiments, pruning a single slow agent in layer 1 often cut cost by 20 % with <1 % quality drop, but you only notice that if you have fine-grained metrics. ([arxiv.org][1])

### 11. Embrace **modularity for downstream tasks**

Treat MoA as a generic *reasoning booster*: plug it behind retrieval-augmented QA, code generation or chain-of-thought planning systems.  Comparative studies on NLI and pen-testing agents show consistent uplift when MoA is dropped in as the final reasoning module. ([sciencedirect.com][9], [isamu-website.medium.com][10])

### 12. Use **simulation tests** before live traffic

For each new spec, generate a synthetic benchmark of 200–500 prompts, run the mixture, collect automatic metrics (BLEU, execution success, unit-tests) and only then enable canary traffic.  OpenReview comments on the MoA ICLR 2025 version underline how ablation and simulation were critical to avoid quality regressions. ([openreview.net][11])

---

#### Quick-start template (pseudo-code)

```python
mixture = Mixture(
    layers=[
        [ Agent(model='mistral-8x7b', role='Draft'),
          Agent(model='mixtral-8x22b', role='Draft'),
          Agent(model='qwen-2-72b', role='Draft') ],
        [ Agent(model='claude-opinionated-200k', role='Refine') ]
    ],
    gate = LearnedGate(metric='logp+consistency'),
    memory = VectorStore('chromadb'),
    safety = SafetyChecker(model='tiny-llama-safety-v0')
)

answer = mixture.run(prompt)
```

Use this as a skeleton and swap pieces following the twelve strategies above.

---

**Key References**

* Wang et al., *Mixture-of-Agents Enhances Large Language Model Capabilities*, 2024 ([arxiv.org][1])
* Together.ai MoA documentation ([docs.together.ai][3])
* Sun & al., *Modular Agentic Planner*, 2024 ([arxiv.org][4])
* Wei & al., *Mixture-of-Search-Agents*, 2025 ([arxiv.org][7])
* Yu & al., *Multi-Agent Collaboration Mechanisms Survey*, 2025 ([researchgate.net][6])

[1]: https://arxiv.org/abs/2406.04692?utm_source=chatgpt.com "Mixture-of-Agents Enhances Large Language Model Capabilities"
[2]: https://arxiv.org/html/2406.04692v1?utm_source=chatgpt.com "Mixture-of-Agents Enhances Large Language Model Capabilities"
[3]: https://docs.together.ai/docs/mixture-of-agents?utm_source=chatgpt.com "Together Mixture-Of-Agents (MoA) - Introduction"
[4]: https://arxiv.org/html/2310.00194v4?utm_source=chatgpt.com "Improving Planning with Large Language Models: A Modular ... - arXiv"
[5]: https://medium.com/%40weidagang/coffee-time-papers-mixture-of-agents-23feccb52f3c?utm_source=chatgpt.com "Coffee Time Papers: Mixture of Agents | by Dagang Wei - Medium"
[6]: https://www.researchgate.net/publication/387975271_Multi-Agent_Collaboration_Mechanisms_A_Survey_of_LLMs?utm_source=chatgpt.com "Multi-Agent Collaboration Mechanisms: A Survey of LLMs"
[7]: https://arxiv.org/html/2502.18873v1?utm_source=chatgpt.com "Multi-LLM Collaborative Search for Complex Problem Solving - arXiv"
[8]: https://arxiv.org/abs/2410.03688?utm_source=chatgpt.com "LLM Agents as 6G Orchestrator: A Paradigm for Task-Oriented ..."
[9]: https://www.sciencedirect.com/science/article/pii/S2949719125000160?utm_source=chatgpt.com "Comparative analysis of Mixture-of-Agents models for natural ..."
[10]: https://isamu-website.medium.com/literature-review-on-task-planning-with-llm-agents-a5c60ce4f6de?utm_source=chatgpt.com "Literature Review on Task Planning with LLM Agents - Isamu Isozaki"
[11]: https://openreview.net/forum?id=h0ZfDIrj7T&utm_source=chatgpt.com "Mixture-of-Agents Enhances Large Language Model Capabilities"
