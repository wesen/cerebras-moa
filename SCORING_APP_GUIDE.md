# JSON Scoring Application – Developer Guide

> **Audience:** New contributors who need to build a batch-style scorer that evaluates multiple JSON agent-configuration files against the Competition **goal.md** specification.
>
> **Prerequisites:** Basic Python, Streamlit, familiarity with MOA concepts. Make sure you can run `streamlit run app.py` successfully first.

---

## 1  Purpose & Scope

The existing `app.py` Streamlit GUI already:
1. Lets users craft **Mixture-of-Agents (MOA)** configurations interactively.
2. Executes those agents through `MOAgent.chat()` and streams the answer.
3. (Optionally) embeds the **AI Configuration Challenge** UI that grades a single submission.

Your new **JSON Scoring Application** must instead:
* Load *many* JSON configuration files from disk (e.g. `configs/*.json`).
* For each config, instantiate the MOA stack, run it against the **mission described in** `config/competition/goal.md`, and collect the resulting answer.
* Pass that answer through the existing grading logic (same rules as the competition) to obtain a numeric **score**.
* Output a sortable report (CSV / table / Streamlit dashboard).

No GUI agent editor is required; we only reuse the execution-and-grading core.

---

## 2  High-Level Architecture

```text
+-------------+        +------------------+        +---------------+
| JSON Loader | -----> |  MOA Executor    | -----> |  Grader       |
+-------------+        +------------------+        +---------------+
        |                       |                          |
        |                       v                          v
        |                (MOAgent.chat)             Score & Feedback
        |                                              Summary
        `--> Batch Controller  ----------------------------^
```

1. **JSON Loader** – Reads each configuration file and normalises it into the dict format expected by `MOAgent.from_config()`.
2. **MOA Executor** – Thin wrapper around the existing `MOAgent` class (import from `moa.agent`).
3. **Grader** – Reuse the competition scoring functions currently hidden behind `competition_ui` / `competitive_programming.py` (see *Code reuse* below).
4. **Batch Controller** – Orchestrates load → execute → grade for N configs, aggregates results.

---

## 3  Code Reuse Map

| Existing File | Reusable Pieces | Why & How |
|---------------|----------------|-----------|
| `moa/agent/moa.py` | `MOAgent`, streaming iterator | Core execution of agent layers – **keep unchanged**. |
| `app.py` | `stream_response` helper | Optional if you still want live Streaming in CLI / Streamlit. |
| `competition_ui.py` & friends | Scoring / grading logic | Extract the scorer function (e.g. `calculate_user_metrics`) and its test harness. |
| `QUICK_START.md`, `QUICK_REFERENCE.md` | Environment setup examples | Copy pip install commands, env vars, etc. |

> **Tip:** Start by grepping for `def grade_` or `def score_` inside the competition modules to locate the evaluation routine.

---

## 4  What Needs Modification / Creation

* **Input Schema Handler** – small utility that maps arbitrary keys in incoming JSON to the canonical keys required by `MOAgent.from_config(...)` (`main_model`, `layer_agent_config`, `cycles`, etc.).
* **Batch Execution Script** – new Python module `batch_score.py` that loops over a folder:
  ```python
  for path in Path("configs").glob("*.json"):
      cfg = json.loads(path.read_text())
      moa = MOAgent.from_config(**cfg)
      answer = collect_final_string(moa.chat(USER_PROMPT))
      score = grade(answer)
      results.append((path.name, score))
  ```
* **Grade Wrapper** – expose the private grading routine as importable `from competition.grader import grade`.
* **Reporting** – write a simple Pandas DataFrame to `scores.csv` **and** optionally show a Streamlit table:
  ```python
  st.dataframe(df.sort_values("score", ascending=False))
  ```
* **CLI Entrypoint** – add a `if __name__ == "__main__":` block so you can run `python batch_score.py --config_dir configs/ --output scores.csv`.

---

## 5  Step-by-Step Implementation Checklist

1. **Environment**
   - [ ] Copy `requirements.txt` from root; add `pandas` if missing.
   - [ ] `pip install -r requirements.txt` and export `CEREBRAS_API_KEY`.
2. **Project Layout** (`/scoring_app` suggestion)
   - [ ] `__init__.py`
   - [ ] `batch_score.py` – main controller.
   - [ ] `json_loader.py` – schema normaliser.
3. **Expose Grader**
   - [ ] Move / import the grading logic into a separate module, e.g. `competition/grader.py`.
   - [ ] Ensure it can run *headless* (no Streamlit session state).
4. **Implement Batch Controller**
   - [ ] Iterate over JSONs, call `run_one_config` helper.
   - [ ] Collect timing stats (`time.perf_counter`) for performance scoring.
5. **Parallelism (Optional)**
   - [ ] Use `concurrent.futures.ThreadPoolExecutor` with a small pool to overlap network I/O.
   - [ ] Respect rate limits – honour retries or `time.sleep` when 429 occurs.
6. **Reporting Layer**
   - [ ] Build DataFrame `[['file', 'score', 'duration_s', 'details']]`.
   - [ ] Save as CSV & pretty-print to console.
   - [ ] If running within Streamlit, call `st.bar_chart` or `st.dataframe`.
7. **Documentation & Examples**
   - [ ] Update `/docs/examples` with a sample JSON.
   - [ ] Add Makefile target `make score-all` that runs the script.

---

## 6  Developer Tips & Pitfalls

* **Context Length** – Large layer outputs may blow past model context. Follow the concatenation/summarise rule (see *guide.md* §4) if you plan to chain more than ~3 cycles.
* **Error Handling** – Wrap `MOAgent.chat()` in `try/except` exactly as `app.py` does; map API exceptions to partial scores (don't crash the batch).
* **Cerebras Model Quotas** – Batch jobs can hit rate limits quickly; add exponential backoff.
* **Determinism for Fair Scoring** – Fix `temperature` to `0.0-0.2` when benchmarking; otherwise scores can fluctuate.

---

## 7  Resources & Further Reading

* `QUICK_START.md` – Launch instructions & competition breakdown.
* `QUICK_REFERENCE.md` – Parameter reference, best practices.
* Wang et al., *Mixture-of-Agents Enhances Large Language Model Capabilities*, 2024.
* MOA paper layering guidelines (see `config/competition/guide.md`).

---

## 8  Next Steps

Once the MVP works:
1. Integrate **vector-store caching** so identical prompts reuse previous answers.
2. Swap the console reporter for a **dashboard** (Streamlit or Jupyter) that lets instructors drill into per-criterion scores.
3. Add a **CI job** (GitHub Actions) that runs `batch_score.py` on every PR to prevent regressions.
4. Consider adding a **gating model** (small decision tree) that combines different criteria into a single final score, mirroring §6 of *guide.md*.

Happy scoring! 🎉 