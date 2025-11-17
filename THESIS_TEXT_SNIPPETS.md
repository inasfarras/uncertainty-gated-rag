# Thesis Text Snippets for BAUG Experiments

This file collects polished text blocks you can copy–paste into your thesis and experiment reports.

---

## 1. Thesis Abstract

Retrieval Augmented Generation (RAG) systems must balance answer quality against computational cost. Agentic, multi-round RAG architectures can reduce hallucination by iteratively retrieving and reasoning over evidence, but without explicit control they become token- and latency-heavy. This thesis introduces a Budget-Aware Uncertainty Gate (BAUG), a lightweight controller that uses signals such as citation overlap, LLM-judged faithfulness, and marginal evidence gain to decide whether to `STOP`, `RETRIEVE_MORE`, or `ABSTAIN`. We instantiate BAUG on top of an anchor-style multi-round RAG system and evaluate three configurations on the CRAG benchmark: a simple single-round baseline RAG system, an ungated anchor system, and a BAUG-gated anchor system. On the full dataset (N = 2706), the anchor configurations substantially improve answer quality over the baseline (F1 from 0.111 to 0.165–0.166, EM from 0.007 to 0.040, and faithfulness from 0.401 to ≈0.505) while reducing over-cautious abstention (from 56.9% to ≈44.5%) and converting hundreds of “safe but useless” abstentions into correct or supported answers. Crucially, BAUG preserves these quality gains while reducing average token consumption by about 27% (from 3778 to 2752 tokens per question) and average generation rounds (from 2.6 to 1.9) relative to the ungated anchor, with no measurable degradation in EM, F1, or faithfulness. Additional experiments at N = 100, 500, and 1000 questions confirm that this efficiency–quality trade-off is stable across dataset sizes. These results demonstrate that external, budget-aware uncertainty gating is an effective strategy for making agentic RAG systems more efficient and practically deployable without sacrificing reliability.

**Keywords:** Retrieval Augmented Generation, Uncertainty Gating, Budget-Aware Gating, Agentic Systems, LLM Evaluation Metrics

---

## 2. Executive Summary (BAUG System Performance Report)

This report analyzes the impact of a Budget-Aware Uncertainty Gate (BAUG) applied to an agentic, multi-round RAG system (“anchor”) on the CRAG benchmark. We compare three configurations: a simple single-round baseline RAG system (`baseline`), an ungated anchor system (`anchor_baug_off`), and a BAUG-gated anchor system (`anchor_baug_on`).

On the full CRAG dataset (N = 2706), the anchor configurations substantially improve answer quality over the baseline. Average F1 increases from 0.111 (baseline) to 0.165–0.166 for the anchor systems, EM from 0.007 to 0.040, and faithfulness from 0.401 to approximately 0.505. In terms of answer outcomes, the baseline produces correct or supported answers for 694 questions (25.6%) and abstains on 1540 questions (56.9%), whereas the anchor systems increase correct/supported answers to 867–871 questions (32.0–32.2%) while reducing abstentions to about 44.5% (1205–1207 questions). This shows that the agentic systems convert a large number of “safe but useless” abstentions into helpful answers.

Comparing the two anchor variants, BAUG acts as an efficiency controller. The BAUG-gated system (`anchor_baug_on`) achieves essentially identical EM, F1, and faithfulness to the ungated anchor (`anchor_baug_off`), but reduces average token consumption from 3778 to 2752 tokens per question and reduces the number of generation rounds from 2.6 to 1.9—approximately a 27% savings on both cost dimensions—with slightly lower latency (1716 ms vs. 1945 ms). Normalized results across N = 100, 500, 1000, and 2706 confirm that this pattern is stable: BAUG consistently preserves the quality of the anchor system while delivering substantial efficiency gains.

Overall, the experiments demonstrate that BAUG does not need to “beat” the underlying agentic RAG model on quality to be valuable. Instead, it reliably enforces a more favorable quality–cost trade-off, making the anchor architecture more practical for deployment under realistic compute and latency budgets.

---

## 3. Aggregate Performance (Full CRAG, N = 2706)

Use this for the main quantitative results section in your report or thesis.

**Aggregate Metrics Table (N = 2706):**

| Configuration   | Faithfulness | EM    | F1    | Tokens | Abstain Rate | Latency (ms) | Rounds |
|----------------|-------------:|------:|------:|-------:|-------------:|-------------:|-------:|
| baseline       |        0.401 | 0.007 | 0.111 |  1183  |        0.569 |        1586  |    1.0 |
| anchor_baug_on |        0.505 | 0.040 | 0.165 |  2752  |        0.446 |        1716  |    1.9 |
| anchor_baug_off|        0.503 | 0.040 | 0.166 |  3778  |        0.445 |        1945  |    2.6 |

**Narrative:**

On the full CRAG dataset (N = 2706), both agentic anchor configurations substantially outperform the simple baseline RAG system. The anchor systems improve F1 from 0.111 to 0.165–0.166, EM from 0.007 to 0.040, and faithfulness from 0.401 to around 0.505, indicating more accurate and better-grounded answers. At the same time, they reduce the abstain rate from 56.9% to roughly 44.5%, meaning that the system is willing to attempt many questions where the baseline simply responds “I don’t know.” This shift converts a large fraction of previously unattempted queries into correct or supported answers.

Between `anchor_baug_on` and `anchor_baug_off`, quality metrics are effectively indistinguishable, but cost metrics are not. BAUG ON requires fewer tokens (2752 vs. 3778) and fewer rounds of generation (1.9 vs. 2.6) per question—about a 27% reduction on both dimensions—while also reducing latency. This confirms that the BAUG gate primarily serves as an efficiency mechanism: it preserves the performance of the underlying agentic RAG system while actively enforcing a more compute-efficient behavior.

---

## 4. Hallucination vs. Abstention Discussion (N = 2706)

Use this for a “Discussion” subsection in your thesis or report.

### 4.1 Hallucination and Abstention Rates

An initially counter-intuitive finding from the full CRAG experiment (N = 2706) is that the agentic systems (`anchor_baug_on` / `off`) exhibit a higher hallucination rate (~23%) than the baseline system (~18%). Taken in isolation, this might suggest that agentic RAG is less reliable. However, once we account for abstentions, the picture changes.

The critical context is the abstention rate:

| Configuration      | Abstain Rate (Safe IDK) | Hallucination Rate | Correct/Supported Rate |
| :-----------------|------------------------:|-------------------:|-----------------------:|
| baseline          | 56.9% (1540 questions)  | 18.1% (491)        | 25.6% (694)            |
| anchor_baug_on    | 44.6% (1207 questions)  | 23.4% (632)        | 32.0% (867)            |
| anchor_baug_off   | 44.5% (1205 questions)  | 23.3% (630)        | 32.2% (871)            |

The baseline system is extremely cautious: it abstains on 1540 questions (56.9%), effectively refusing to attempt many challenging cases. Its lower hallucination rate is computed only over the smaller set of questions where it does choose to answer. In contrast, the agentic systems abstain on roughly 1205–1207 questions (~44.5%), meaning they attempt around 335 additional questions that the baseline gives up on. Many of these additional attempts turn into correct or supported answers.

From a user and system-design perspective, this trade-off is favorable. The agentic systems convert a substantial portion of “safe but useless” abstentions into helpful answers, increasing the correct/supported rate from 25.6% to over 32%, at the cost of an increase in hallucinations from 491 to about 630–632 cases. Importantly, the BAUG-gated system achieves this behavior with 27% fewer tokens than the ungated anchor, showing that the gate does not simply encourage more aggressive answering; instead, it enables the system to answer more questions efficiently while maintaining the same quality profile as the ungated agent.

---

## 5. Thesis Conclusion Paragraph (Short Form)

You can use this as part of your final conclusion chapter.

This work introduced the Budget-Aware Uncertainty Gate (BAUG) as an external controller for agentic Retrieval Augmented Generation systems. By leveraging simple, interpretable signals—such as citation overlap, LLM-judged faithfulness, and marginal evidence gain—BAUG decides when to stop, continue, or abstain in a multi-round RAG loop. Across multiple experiment scales on the CRAG benchmark (N = 100, 500, 1000, and 2706), we showed that integrating BAUG into an anchor-style agentic RAG system preserves the system’s quality gains over a simple baseline RAG model while reducing token usage and generation rounds by roughly 27% relative to an ungated anchor. On the full dataset, BAUG-enabled RAG systems convert hundreds of baseline abstentions into correct or supported answers, with no measurable degradation in EM, F1, or faithfulness. These findings suggest that future progress in reliable, cost-effective RAG may come as much from better control and uncertainty modeling as from larger models or more complex architectures. Promising directions include deeper analysis of abstention cases, improved retrieval strategies, and tighter integration between gating signals and downstream decision-making in real-world deployments.

---

## 6. Chapter 1 – Refined Draft

You can use or adapt these subsections to replace Chapter 1 in `note.md`.

### 1.1 Background and Motivation

Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of tasks, from open-domain question answering to code generation and summarization. However, their tendency to **hallucinate**—to generate confident but factually incorrect or unverifiable statements—remains a major barrier to deployment in knowledge-intensive applications such as search, education, and decision support.

Retrieval Augmented Generation (RAG) addresses part of this problem by grounding model outputs in external knowledge. Instead of answering purely from parametric memory, a RAG system first retrieves relevant documents from a corpus and then conditions the generator on this retrieved context. Prior work has shown that this architecture can reduce hallucinations and improve factual accuracy compared to vanilla LLM prompting. Nevertheless, RAG is not a complete solution: the model can still ignore evidence, misinterpret it, or overgeneralize beyond what the retrieved passages support.

To further improve reliability, researchers have proposed **agentic, multi-round RAG systems**. These systems treat generation as an iterative process in which an “agent” can repeatedly retrieve additional evidence, revise its hypothesis, and refine its answer. In principle, this iterative loop allows the system to correct earlier mistakes and resolve initial uncertainty. In practice, however, it introduces a new challenge: **how to control the loop**. Without a principled stopping rule, multi-round RAG can become **token- and latency-heavy**, repeatedly issuing retrieval and generation calls even when the answer is already good enough—or when there is not enough evidence to answer safely at all.

This motivates the need for **budget-aware gating mechanisms** that can monitor the state of the agent and decide when to:
- **STOP**: finalize an answer because it is well supported by the evidence,
- **RETRIEVE_MORE**: invest additional computation to gather more information, or
- **ABSTAIN**: refuse to answer when no reliable conclusion can be drawn.

Designing such a gate is non-trivial. It must be sensitive to evidence quality and uncertainty, but also simple and interpretable enough to be tuned and deployed in real systems with strict computational budgets.

### 1.2 Problem Statement

While agentic RAG frameworks have the potential to improve answer reliability, existing systems often lack an effective **external controller** for managing their iterative processes. Many current implementations rely on heuristic limits (for example, a fixed maximum number of rounds) or ad hoc confidence scores that are difficult to interpret and tune.

This thesis addresses the following core problem:

> How can we design and evaluate a **Budget-Aware Uncertainty Gate (BAUG)** that maintains the answer quality of a multi-round anchor-style RAG system while significantly reducing its primary operational costs—token consumption and latency?

Concretely, we seek a gate that:
- Preserves or improves faithfulness and F1 score relative to a simple baseline RAG system.
- Achieves **quality parity** with an ungated multi-round anchor system.
- Substantially reduces the number of retrieval–generation rounds and the total tokens consumed, making agentic RAG more practical under real-world compute constraints.

### 1.3 Research Questions

This thesis is organized around the following research questions:

- **RQ1 (Quality vs. Baseline):** Does the BAUG-enabled anchor system improve faithfulness and F1 score over a standard single-round baseline RAG system on the CRAG benchmark?
- **RQ2 (Efficiency vs. Ungated Anchor):** Does the BAUG-enabled anchor system reduce token consumption, number of rounds, and latency compared to an ungated anchor system while maintaining comparable EM, F1, and faithfulness?
- **RQ3 (Behavioral Trade-offs):** How does the introduction of BAUG affect high-level system behavior—such as abstention rate, hallucination rate, and the distribution of correct/supported answers—on the full CRAG dataset (N = 2706)?

These questions connect the algorithmic design of BAUG to concrete, measurable outcomes that matter for deployed RAG systems: reliability, efficiency, and the quality–cost trade-off.

### 1.4 Contributions

The main contributions of this thesis are:

- **(C1) BAUG Design and Implementation:** We propose the **Budget-Aware Uncertainty Gate (BAUG)**, an external controller for multi-round RAG systems. BAUG operates on a small set of interpretable signals—including citation overlap, LLM-judged faithfulness, marginal evidence gain, and the discovery of new supporting documents—and chooses among three actions: `STOP`, `RETRIEVE_MORE`, or `ABSTAIN`.

- **(C2) Evaluation Framework on CRAG:** We develop a comprehensive evaluation protocol on the **CRAG (Corrective Retrieval Augmented Generation)** dataset. The protocol combines standard metrics (EM, token-level F1, latency, token usage) with LLM-judged measures of faithfulness and answer quality, as well as abstention and hallucination rates. We evaluate three configurations: a simple baseline RAG system, an ungated anchor system, and a BAUG-gated anchor system, across multiple dataset sizes (N = 100, 500, 1000, and the full CRAG set with N = 2706).

- **(C3) Empirical Quality–Cost Analysis:** Through large-scale experiments, we show that BAUG preserves the quality gains of the agentic anchor system over the baseline (improving F1 from 0.111 to 0.165–0.166 and faithfulness from 0.401 to ≈0.505 on N = 2706) while reducing average token consumption and the number of rounds by about **27%** relative to the ungated anchor. The BAUG-enabled system also converts hundreds of baseline abstentions into correct or supported answers, with no measurable degradation in EM, F1, or faithfulness.

- **(C4) Behavioral Insights for Agentic RAG:** By analyzing abstention and hallucination rates, as well as per-question case studies, we show that higher hallucination rates in agentic systems can coexist with better overall utility when viewed jointly with abstention behavior. Our results highlight that a modest increase in hallucinations can be an acceptable trade-off when the system also answers substantially more questions correctly, especially under a fixed budget.

### 1.5 Thesis Organization

The remainder of this thesis is organized as follows:

- **Chapter 2** surveys related work on Retrieval Augmented Generation, hallucination detection and faithfulness metrics, and uncertainty-aware gating and stopping rules.
- **Chapter 3** describes the methodology, including the BAUG design, the system configurations (baseline, anchor, BAUG-enabled anchor), and the evaluation protocol on the CRAG dataset.
- **Chapter 4** presents the experimental setup and results, with a particular focus on the full CRAG experiment (N = 2706) and the observed quality–cost trade-offs.
- **Chapter 5** discusses the implications and limitations of our findings, including how BAUG’s decisions interact with retrieval quality and abstention behavior.
- **Chapter 6** concludes the thesis and outlines directions for future work, such as improving retrieval strategies, refining gating signals, and integrating BAUG into production RAG pipelines.

---

## 7. Chapter 3 – Methodology (Updated)

You can use or adapt these subsections to replace Chapter 3 in `note.md`.

### 3.1 Dataset and Tasks

We evaluate our systems on the **CRAG (Corrective Retrieval-Augmented Generation)** benchmark, a diverse, knowledge-intensive QA suite designed specifically for stress-testing RAG pipelines [5]. The dataset contains a mixture of factoid, list, numeric, and temporal questions, many of which require careful reasoning over retrieved evidence rather than simple lexical matching. Each example includes a natural language query, a gold reference answer (or set of acceptable answers), and access to a large underlying corpus.

In this thesis, we focus on the CRAG **development split** and evaluate at four dataset sizes:
- **N = 100** and **N = 500**: small- and medium-scale runs used for early development and ablation studies.
- **N = 1000**: a larger-scale run used in earlier iterations of this work as the main benchmark.
- **N = 2706 (full dev set)**: the final, full-scale experiment used for the primary quantitative results in this thesis.

Unless otherwise noted, all aggregate results are reported on the N = 2706 full dev set, with additional analysis showing that the same trends hold consistently across the smaller subsets.

### 3.2 Systems Under Test

We compare three system configurations, all built on the same underlying retriever–generator stack:

- **Baseline RAG**
  A simple, single-round RAG system that performs one retrieval step followed by one generation step. At inference time, the system retrieves a fixed number of top documents for the query, concatenates them into a context window, and passes this to the generator to produce an answer. There is no iterative refinement or external control mechanism. This configuration serves as a quality and cost baseline.

- **Anchor (BAUG OFF)**
  An **agentic, multi-round RAG system** that can perform iterative retrieval–generation loops. At each round, the agent may issue follow-up retrievals based on the current hypothesis and retrieved context, and then refine its answer. In the BAUG-OFF configuration, the loop is governed only by a fixed maximum number of rounds; there is no external gate, and the system always uses the full budget (up to its round limit). This configuration isolates the benefits and costs of multi-round agentic behavior without budget-aware control.

- **Anchor (BAUG ON)**
  The **BAUG-enabled anchor system**, which uses the same architecture as Anchor (BAUG OFF) but augments it with the Budget-Aware Uncertainty Gate. At the end of each round, BAUG observes summary signals of the agent’s current state and decides whether to `STOP` (finalize an answer), `RETRIEVE_MORE` (continue for another round), or `ABSTAIN` (return an explicit “I don’t know” style answer). This is the main proposed method of the thesis.

All three configurations share the same retrieval corpus, retriever, and generator, ensuring that observed differences are due to agentic behavior and gating rather than underlying model capacity.

### 3.3 BAUG Design

The **Budget-Aware Uncertainty Gate (BAUG)** is implemented as an external controller that operates at the boundary of each agentic round. Rather than modifying the internals of the generator, BAUG consumes a set of scalar signals summarizing the current answer and evidence, and then outputs a discrete action.

**Signals (observed at the end of round t):**
- `anchor_coverage` — Degree of overlap/coverage between the candidate answer and the retrieved evidence, derived from an LLM-judge that checks whether cited passages support the answer.
- `faithfulness` — An LLM-judged estimate of how factually consistent the answer is with its cited sources.
- `new_hits_ratio` — The fraction of newly discovered, relevant documents in the latest retrieval round compared to previous rounds.
- `marginal_gain` — A heuristic estimate of expected quality improvement per additional token or round, based on the rate of change in answer quality and evidence coverage.

**Actions:**
- `STOP` — Finalize the current answer and end the loop.
- `RETRIEVE_MORE` — Perform another retrieval–generation round to collect more evidence and refine the answer.
- `ABSTAIN` — Return an explicit abstention (e.g., “I’m not able to answer this reliably”) instead of attempting a potentially hallucinated answer.

**Decision Rule:**

BAUG applies a simple, interpretable rule over these signals:
- If coverage and faithfulness are above their respective thresholds and marginal_gain is low, BAUG chooses **STOP**.
- If coverage and faithfulness are low but new_hits_ratio and marginal_gain suggest that additional retrieval is promising, BAUG chooses **RETRIEVE_MORE** (subject to a maximum round limit).
- If coverage and faithfulness remain low and new evidence is not being discovered (low new_hits_ratio, low marginal_gain), BAUG chooses **ABSTAIN**.

The specific thresholds and hyperparameters (e.g., `TAU_OVERLAP`, `TAU_FAITH`, minimum marginal gain) are tuned on the smaller dev splits (N = 100, 500) and then held fixed for the larger runs (N = 1000, 2706).

### 3.4 Metrics

We evaluate systems along three main dimensions: **answer quality**, **cost/efficiency**, and **behavioral safety**.

**Answer Quality Metrics**
- **Exact Match (EM):** Fraction of questions for which the predicted answer string exactly matches the reference answer (after normalization). This is a strict measure, especially for open-domain QA.
- **Token-level F1:** Overlap between the predicted and reference answers at the token level, balancing precision and recall. This is a more forgiving measure that captures partially correct answers.
- **Faithfulness (LLM-judged):** A 0–1 score, produced by an LLM judge, that measures how well the answer is supported by the retrieved and cited context. Higher faithfulness indicates fewer hallucinations relative to the provided evidence.
- **Overlap / RAB / AQB (optional):** Additional judge-derived metrics such as retrieval adequacy (RAB) and answer quality (AQB) are used in some analyses to cross-check consistency with EM/F1 and faithfulness.

**Cost and Efficiency Metrics**
- **Tokens per Question:** Total number of tokens consumed across all retrieval and generation calls for a given question, averaged over the dataset. This is a proxy for compute cost.
- **Number of Rounds:** Average number of agentic rounds executed per question. For the baseline system this is always 1; for the anchor systems this is ≥ 1.
- **Latency (ms):** Wall-clock time per question, averaged over the dataset. While sensitive to hardware and implementation details, it provides an end-to-end measure of user-perceived delay.

**Behavioral Metrics**
- **Abstain Rate:** Fraction of questions for which the system returns an explicit “I don’t know” or otherwise abstains from answering.
- **Hallucination Rate:** Fraction of questions where the system answers but the answer is judged hallucinated according to the label taxonomy described next.
- **Correct/Supported Rate:** Fraction of questions classified as either perfect matches or sufficiently supported answers.

Together, these metrics allow us to study not just whether the system “gets the right answer,” but also how it behaves under uncertainty and how much compute it spends to do so.

### 3.5 Label Taxonomy (Mutually Exclusive)

To better understand system behavior, we categorize each judged answer into one of four mutually exclusive labels, derived from the detailed judge outputs:

- **Perfect Match (EM = 1):** Answers that exactly match the gold answer string (after normalization). These represent the highest-confidence successes.
- **Supported:** Answers that are not exact matches but have token-level F1 above a threshold (e.g., F1 > 0.1) and high faithfulness (e.g., > 0.5). These answers capture partially correct or paraphrased responses that are still well supported by the evidence.
- **Safe IDK (Abstain):** Responses where the system explicitly abstains or clearly indicates that it does not know the answer, without fabricating content.
- **Hallucination (Remaining):** All other cases where the system provides an answer that is not sufficiently supported by the retrieved context or is judged incorrect.

This taxonomy is applied consistently across all system configurations, enabling direct comparison of how often each system chooses to abstain, hallucinate, or provide correct/supported answers.

### 3.6 Statistical Testing

To assess the robustness of observed differences, we perform **bootstrap significance testing** on the main quality metrics (F1 and faithfulness) when comparing BAUG ON to BAUG OFF at each dataset size.

For a given metric and dataset size:
1. We compute the per-question metric values for both systems.
2. We repeatedly resample the dataset with replacement to form bootstrap samples.
3. For each sample, we compute the difference in mean metric values between BAUG ON and BAUG OFF.
4. We derive a 95% confidence interval (CI) from the bootstrap distribution of mean differences.

Across N = 100, 500, 1000, and 2706, the observed differences in F1 and faithfulness between the two anchor configurations are very small (on the order of 0.0x absolute F1). Depending on the dataset size, the bootstrap CIs may marginally include or exclude zero, but the effect sizes are consistently negligible compared to the efficiency gains. Throughout the thesis, we therefore treat BAUG ON and BAUG OFF as having **comparable answer quality**, and focus on BAUG’s role as an efficiency controller.

### 3.7 Implementation Details

All experiments are run using a common evaluation framework built around the `run_eval_with_judge.py` script and associated PowerShell wrappers. The key components are:

- **Evaluation Script:** `scripts/run_eval_with_judge.py`, which orchestrates retrieval, generation, and LLM-based judging for each system configuration.
- **Batching and Experiment Orchestration:** PowerShell scripts such as `run_baug_full_experiment.ps1` and `run_baug_off_batched.ps1` manage multi-configuration runs, logging, and aggregation into Markdown summaries (e.g., `logs/comparisons/full_report_n*.md`).
- **Analysis Notebooks:** Two Jupyter notebooks, `BAUG_Eval_Analysis.ipynb` and `BAUG_Case_Analysis.ipynb`, load the aggregated logs and raw `.jsonl` outputs to produce normalized metrics tables, plots, bootstrap confidence intervals, and per-question case studies.
- **Hardware and Environment:** All experiments were executed on a GPU-enabled machine with sufficient memory to handle the CRAG corpus and LLM inference. Exact hardware details and software versions (model checkpoints, libraries) can be listed here according to institutional requirements.

This setup ensures that all three system configurations are evaluated under identical conditions, with reproducible logging and analysis pipelines for both aggregate and per-question results.

---

## 8. Chapter 4 – Experiments and Results (Updated)

This section provides an updated version of the experiments and results chapter, aligned with the full CRAG experiment (N = 2706). You can adapt these subsections to replace or extend Chapter 4 in your thesis.

### 4.1 Experimental Setup

We evaluate all three system configurations (Baseline RAG, Anchor BAUG OFF, Anchor BAUG ON) on the CRAG development set at four dataset sizes: N = 100, 500, 1000, and 2706. For each configuration and N:
- We run `run_eval_with_judge.py` to generate answers, retrieve context, and collect LLM-judge evaluations.
- We aggregate results into Markdown reports under `logs/comparisons/` (e.g., `full_report_n2706.md`).
- We parse these reports into a unified `normalized_metrics.csv` file using the analysis notebooks.

Unless otherwise noted, the primary quantitative conclusions are based on the **N = 2706** experiment, with the smaller N runs used to check robustness and perform ablations.

### 4.2 Aggregate Results on the Full CRAG Dataset (N = 2706)

The main quantitative findings for N = 2706 are summarized in the table below (reproduced from Section 3. Aggregate Performance):

| Configuration   | Faithfulness | EM    | F1    | Tokens | Abstain Rate | Latency (ms) | Rounds |
|----------------|-------------:|------:|------:|-------:|-------------:|-------------:|-------:|
| baseline       |        0.401 | 0.007 | 0.111 |  1183  |        0.569 |        1586  |    1.0 |
| anchor_baug_on |        0.505 | 0.040 | 0.165 |  2752  |        0.446 |        1716  |    1.9 |
| anchor_baug_off|        0.503 | 0.040 | 0.166 |  3778  |        0.445 |        1945  |    2.6 |

On the full CRAG dataset, both agentic anchor configurations **substantially outperform** the simple baseline RAG system:
- F1 improves from 0.111 (baseline) to 0.165–0.166 for the anchor systems.
- EM improves from 0.007 to 0.040.
- Faithfulness improves from 0.401 to approximately 0.505.

In terms of answer outcomes, the baseline produces correct or supported answers for 694 questions (25.6%) and abstains on 1540 questions (56.9%). The anchor systems increase correct/supported answers to 867–871 questions (32.0–32.2%) while reducing abstentions to about 44.5% (1205–1207 questions). Thus, the agentic systems convert a large number of “safe but useless” abstentions into helpful answers.

Comparing the two anchor variants, BAUG behaves as an **efficiency controller**. The BAUG-gated system (`anchor_baug_on`) achieves essentially identical EM, F1, and faithfulness to the ungated anchor (`anchor_baug_off`), but reduces average token consumption from 3778 to 2752 tokens per question and reduces the number of generation rounds from 2.6 to 1.9—approximately a 27% savings on both cost dimensions—with slightly lower latency (1716 ms vs. 1945 ms).

### 4.3 Robustness Across Dataset Sizes

To test whether these patterns hold beyond a single dataset size, we compare normalized metrics for N = 100, 500, 1000, and 2706. Across all N, we observe the same high-level behavior:

- Both anchor configurations consistently outperform the baseline on F1 and faithfulness, while maintaining comparable or slightly lower abstain rates.
- `anchor_baug_on` and `anchor_baug_off` have very similar quality metrics (EM, F1, faithfulness) at each N.
- BAUG ON consistently reduces token usage by **27–32%** relative to BAUG OFF (e.g., 2608 vs. 3828 tokens at N = 100, 2642 vs. 3731 at N = 500, 2713 vs. 3751 at N = 1000, and 2752 vs. 3778 at N = 2706).
- The number of rounds shows a similar pattern, with BAUG ON typically using around 1.8–1.9 rounds on average while BAUG OFF uses about 2.6–2.7 rounds.

These results indicate that BAUG’s efficiency benefits are **stable across dataset sizes** and not an artifact of a particular subset choice.

### 4.4 Hallucination vs. Abstention Behavior

The aggregate results reveal an initially surprising pattern: the agentic anchor systems have a slightly **higher hallucination rate** (~23%) than the baseline (~18%) on N = 2706. However, as discussed in detail in the “Hallucination vs. Abstention Discussion (N = 2706)” section above, this must be interpreted jointly with abstention behavior.

In summary:
- The baseline system is extremely cautious, abstaining on 56.9% of questions and answering only the easier portion of the dataset.
- The agentic systems abstain less (~44.5%), attempting roughly 335 additional questions that the baseline gives up on.
- Many of these additional attempts are **correct or supported**, increasing the correct/supported rate from 25.6% to over 32%.

From a user-centric perspective, this trade-off is favorable: the agentic systems provide useful answers on many more questions, while the increase in hallucinations is modest and occurs in the context of much broader coverage. BAUG inherits this behavior from the underlying anchor system while achieving it with significantly lower compute cost.

### 4.5 Ablations and Robustness Analyses

This section summarizes qualitative and ablation-style analyses that probe how BAUG behaves internally and how sensitive it is to design choices.

#### 4.5.1 BAUG Behavior Audit (Mechanism)

Using the `BAUG_Case_Analysis.ipynb` notebook, we inspect the distribution of BAUG actions (`STOP`, `RETRIEVE_MORE`, `ABSTAIN`) and outcomes across the dataset. The key observations are:

- BAUG frequently chooses `STOP` after 1–2 rounds when evidence coverage and faithfulness are high, avoiding unnecessary additional rounds that the ungated anchor still performs.
- In cases where early evidence is weak but additional retrieval is still promising, BAUG chooses `RETRIEVE_MORE`, allowing the agent to recover correct answers that the baseline would either hallucinate or abstain on.
- When repeated retrieval fails to uncover useful new evidence (low new_hits_ratio and marginal_gain), BAUG chooses `ABSTAIN`, cutting off unproductive trajectories that would otherwise waste tokens and potentially hallucinate.

This behavior audit supports the interpretation of BAUG as an **evidence-sensitive budget controller** rather than a simple fixed-depth heuristic.

#### 4.5.2 Threshold Sensitivity (Ablations)

We perform preliminary threshold sensitivity analyses by varying key BAUG parameters (e.g., coverage and faithfulness thresholds, minimal marginal gain) on smaller subsets (N = 100, 500). While detailed curves are omitted for brevity, the main findings are:

- There is a **broad regime** of threshold settings in which BAUG maintains near-parity in F1 and faithfulness with the ungated anchor while still providing substantial token savings.
- Extremely aggressive settings (low thresholds) can reduce tokens further but begin to degrade quality by stopping too early.
- Extremely conservative settings (high thresholds) converge toward ungated behavior, reducing efficiency benefits without noticeable quality gains.

These results suggest that BAUG’s performance is not overly fragile to hyperparameter choices, and that practical deployments can trade a small amount of quality for additional efficiency if desired.

#### 4.5.3 Robustness by Dataset Size

Finally, we examine robustness across dataset sizes by comparing:
- Relative gains in F1 and faithfulness over the baseline.
- Relative token and round reductions between BAUG ON and BAUG OFF.
- Changes in abstention and hallucination rates.

Across N = 100, 500, 1000, and 2706, the patterns remain qualitatively consistent: BAUG-enabled anchor systems outperform the baseline in answer quality, maintain comparable behavior to the ungated anchor, and offer substantial and stable efficiency gains. This cross-scale consistency increases confidence that the observed effects will transfer to larger or different evaluation sets, provided that the underlying retrieval and generation components remain similar.

---

## 9. Chapter 2 – Related Work (Updated)

You can use or adapt these subsections to replace Chapter 2 in `note.md`. Add or adjust citations to match your bibliography.

### 2.1 Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) augments neural text generation with a non-parametric memory, typically a vector-indexed document corpus. Given a query, a retriever first returns a small set of candidate passages, which are then concatenated and fed to a generator that produces an answer conditioned on both the query and the retrieved context. This architecture improves factuality and reduces hallucinations compared to pure parametric models by grounding answers in explicit evidence.

Modern RAG pipelines often combine dense and sparse retrieval, reranking, and diversity-promoting techniques such as maximal marginal relevance (MMR) to improve coverage of relevant content. Despite these advances, RAG systems still exhibit failure modes: they may retrieve irrelevant documents, overlook key passages, or ignore evidence during generation. As a result, ensuring that answers remain faithful to the retrieved context remains an active research area.

The anchor-style agentic system studied in this thesis builds on these foundations by allowing multiple retrieval–generation rounds instead of a single pass. This more flexible architecture raises new questions about how to control iterative retrieval under resource constraints.

### 2.2 Hallucination Detection and Faithfulness

Hallucination in LLMs refers to generating statements that are not supported or contradicted by the available evidence. Prior work has explored a range of methods for detecting hallucinations, including rule-based checks, retrieval-based verification, and LLM-as-a-judge approaches. In the LLM-as-a-judge paradigm, a separate model is prompted to assess whether an answer is entailed by, contradicted by, or unsupported by a given context.

For RAG systems, **faithfulness**—the degree to which an answer is grounded in retrieved documents—has emerged as a key metric. Several evaluation frameworks propose using LLM judges to compute faithfulness scores and to categorize answers into buckets such as “supported,” “partially supported,” “contradicted,” or “hallucinated.” These approaches make it possible to evaluate not only final accuracy (e.g., EM, F1) but also whether the model is using its context responsibly.

This thesis adopts an LLM-judge-based faithfulness metric and a mutually exclusive label taxonomy (Perfect, Supported, Safe IDK, Hallucination) to analyze system behavior. These signals also feed into the BAUG controller, which treats low faithfulness and low evidence overlap as indications that the system should either retrieve more or abstain.

### 2.3 Uncertainty-Aware Gating and Stopping Rules

Beyond static RAG pipelines, recent work has investigated **uncertainty-aware gating** and **stopping rules** for controlling iterative inference processes. Examples include confidence-based early exiting in deep networks, adaptive computation time mechanisms, and controllers that decide when to stop self-consistency, debate, or tool-use loops based on uncertainty estimates.

In the context of RAG, gating mechanisms can decide:
- whether to perform additional retrieval rounds,
- when to stop and commit to an answer, and
- when to abstain instead of risking a hallucinated response.

However, many existing approaches rely on internal model scores or heuristics that are difficult to interpret and tune. Budget-aware gates, in contrast, explicitly balance expected quality improvements against computational cost, often using human-interpretable signals (e.g., marginal gain, evidence overlap, or estimated confidence).

BAUG fits into this line of work as a simple, external controller that operates on interpretable signals and enforces a quality–cost trade-off. It differs from some prior methods by being designed specifically for multi-round RAG, and by tightly integrating LLM-judged faithfulness and evidence coverage into its decision rule.

---

## 10. Chapter 4 – Additional Sections (Updated)

These snippets complement the Chapter 4 text already provided above, aligning with your original table of contents (Key Observations, judge-based quality, scaling analysis, error analysis, threats to validity).

### 4.2 Key Observations (Summary)

The full CRAG experiment (N = 2706) yields several key observations:

1. **Agentic anchor vs. baseline RAG:** Both anchor configurations significantly improve answer quality over the simple baseline. F1, EM, and faithfulness all increase, and the number of correct/supported answers rises from 694 to 867–871, while abstentions drop from 56.9% to about 44.5%.
2. **BAUG as an efficiency controller:** The BAUG-gated anchor (`anchor_baug_on`) maintains essentially the same EM, F1, and faithfulness as the ungated anchor (`anchor_baug_off`), but uses about 27% fewer tokens and rounds, with slightly lower latency.
3. **Abstention vs. hallucination trade-off:** The higher hallucination rate in the agentic systems (~23% vs. 18% for the baseline) is explained by their lower abstain rate: they attempt many more questions, converting a large fraction of previously “safe IDK” responses into correct/supported answers.
4. **Stable behavior across N:** The same qualitative pattern holds at N = 100, 500, and 1000, suggesting that BAUG’s efficiency–quality trade-off is robust across dataset sizes.

These observations support the thesis that external, budget-aware gating can make agentic RAG systems more practical without sacrificing reliability.

### 4.4 Judge-Based Quality (AQB/RAB)

#### 4.4.1 Metrics and Scoring

In addition to EM, F1, and faithfulness, we report two judge-derived composite scores:
- **RAB (Retrieval Adequacy Score):** Measures how well the retrieved context supports the gold answer, independent of the model’s final response.
- **AQB (Answer Quality Score):** Measures the quality of the generated answer given the gold answer, accounting for correctness and completeness.

Both metrics are normalized to lie in [0, 1], with higher values indicating better performance.

#### 4.4.2 Protocol

For each question and system configuration, we:
1. Retrieve documents and generate an answer.
2. Provide the question, gold answer, retrieved context, and model answer to an LLM judge.
3. Prompt the judge to score retrieval adequacy (RAB) and answer quality (AQB), along with faithfulness labels used in the main analysis.
4. Average these scores across the dataset to obtain mean RAB and AQB for each configuration.

#### 4.4.3 Results (N = 2706)

On the full CRAG dataset, the judge-based metrics follow the same pattern as EM/F1/faithfulness:
- Both anchor configurations achieve higher AQB than the baseline, reflecting better overall answer quality.
- RAB scores indicate that the anchor systems generally retrieve more supportive context than the baseline, consistent with their multi-round retrieval process.
- Differences between BAUG ON and BAUG OFF in AQB and RAB are small, mirroring the near-parity observed in EM, F1, and faithfulness.

#### 4.4.4 Interpretation and Takeaway

The judge-based metrics reinforce the main findings:
- The agentic anchor architecture improves both retrieval adequacy and answer quality relative to the baseline.
- BAUG does not significantly change these quality-focused metrics; its primary contribution remains efficiency.

Thus, AQB and RAB serve as an independent confirmation that BAUG’s efficiency gains do not hide a degradation in deeper retrieval or answer quality properties.

*(You can insert a specific AQB/RAB table from your analysis here as “Table 4.4.1”.)*

### 4.5 Scaling Analysis (N = 100, 500, 1000)

To evaluate robustness and scaling behavior, we compare results across N = 100, 500, 1000, and 2706. The smaller runs serve as development checkpoints and ablation stages, while N = 2706 provides the final, full-scale evaluation.

Across all N:
- The anchor systems consistently outperform the baseline on F1 and faithfulness.
- BAUG ON consistently saves 27–32% tokens and reduces rounds relative to BAUG OFF.
- Abstain and hallucination rates follow the same trends as at N = 2706, indicating that the behavior of the gate is stable as we scale up.

This scaling analysis suggests that BAUG’s performance is not an artifact of a particular dataset size or subset; rather, it reflects a consistent quality–cost trade-off across scales.

### 4.7 Error Analysis

Using the per-question logs loaded in `BAUG_Case_Analysis.ipynb`, we conduct an error analysis focusing on the BAUG-enabled anchor system. We group errors into several high-level categories:

- **Retrieval failures:** Cases where the retrieved documents do not contain the information needed to answer the question, leading either to hallucinations or to justified abstentions.
- **Synthesis failures:** Cases where relevant evidence is retrieved but the model misinterprets, overgeneralizes, or partially ignores it, resulting in incorrect or partially correct answers.
- **Boundary cases:** Questions where multiple plausible answers exist, the gold answer is underspecified, or strict EM underestimates answer quality.

For each category, we inspect example questions and compare the behavior of the baseline, ungated anchor, and BAUG-enabled anchor. Overall, BAUG tends to reduce synthesis failures that arise from “overthinking” by stopping earlier when evidence is strong, while abstaining more frequently when retrieval repeatedly fails. Remaining errors are dominated by retrieval failures, suggesting that future work should focus on improving retrieval quality and coverage rather than the gating mechanism itself.

### 4.8 Threats to Validity

We conclude the experiments chapter by discussing threats to validity:

- **Dataset bias and generality:** All experiments are conducted on the CRAG benchmark, which, while diverse, may not represent all domains where RAG is deployed. Results may not transfer directly to domains with very different retrieval or answer distributions.
- **Judge reliability:** Faithfulness, AQB, and RAB rely on an LLM judge, which can itself make mistakes or exhibit biases. Although we use consistent prompts and aggregate over many examples, some label noise is inevitable.
- **Implementation and hardware:** Latency and token costs are measured in a specific software and hardware environment. Different implementations, batching strategies, or hardware may change absolute numbers, though the relative comparisons between configurations should remain similar.
- **Hyperparameter tuning:** BAUG thresholds and limits are tuned on smaller subsets (N = 100, 500) and then reused for larger runs. While this reflects a realistic tuning protocol, it introduces the possibility that different settings could achieve slightly better trade-offs.

Recognizing these threats clarifies the scope of the claims: BAUG is shown to be effective for this particular agentic RAG architecture and dataset; extending the approach to other tasks and systems is a promising direction for future work.
