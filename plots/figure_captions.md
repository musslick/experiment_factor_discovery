# Figure Captions — AutoFactor Paper

> All figures currently use **mockup data**. Replace with real experimental results before submission. Captions are written for the final paper version and should be updated once real values are available.

---

## Figure 1 — System Overview

**AutoFactor: an automated pipeline for deriving experimental factors from behavioral data.**
The pipeline takes as input an observable trial matrix (base factors and behavioral outcomes) and iteratively discovers hidden derived factors through three coordinated phases. **(i) Seeding:** an LLM or combinatorial random sampler generates candidate factor descriptions specifying factor type (within-trial or window), factor class (discrete or continuous), levels, and input dependencies. **(ii) Synthesis and scoring:** for each candidate, the LLM synthesizes a `compute_factor()` predicate function, which is executed in a sandboxed subprocess and evaluated by participant-wise cross-validated log-likelihood improvement over the current null model; a novelty penalty discourages rediscovery of known factors. **(iii) Iterative refinement:** a genetic evolution step applies LLM-driven mutation, crossover, and repair operators to the highest-scoring candidates, repeating the score-evolve cycle until convergence. Candidates that pass a fixed held-out validation threshold are registered and their predicate functions are translated into SweetPea factor definitions, updating the null formula for the next discovery round. After each main factor is accepted, pairwise interaction terms among all discovered factors are enumerated and validated (Phase 2, effect search).

---

## Figure 2 — Synthetic Benchmark Recovery

**AutoFactor recovers hidden derived factors across a complexity gradient of synthetic benchmarks.**

**(A)** Precision, recall, and F1 for factor recovery on three synthetic benchmarks, ordered left-to-right by increasing complexity: Stroop-Simon (4 discrete factors), RDK Task-Switching (4 factors of mixed type), and Prospect Theory (7 factors, predominantly continuous). The number of ground-truth hidden factors is shown above each benchmark. Bars show mean ± SE across five independent seeds. Performance decreases with benchmark complexity, reflecting the growing difficulty of the factor space and the increasing proportion of continuous and window-type factors.

**(B)** Per-factor recovery scores for each ground-truth hidden factor, grouped by benchmark (horizontal rule separates groups) and ordered within each group by factor type (within-trial discrete → window discrete → within-trial continuous → window continuous). Bar length is the mean recovery score across seeds ± SE (error bars). For **discrete factors** (solid bars), the score is level recall — the proportion of ground-truth factor levels individually matched at ≥ 95% bijection agreement by any discovered factor level, averaged across seeds. For **continuous factors** (hatched bars), the score is mean |Spearman ρ| between the best-matching discovered factor and the ground-truth factor column, averaged across seeds. Bar color encodes factor type (legend, lower right). Factors consistently recovered appear near 1.0; missed factors appear near 0.0; partial level recovery produces intermediate values. A clear gradient is visible both across benchmarks and within each benchmark, with within-trial discrete factors recovered most reliably and window continuous factors least reliably.

**(C)** Aggregated 2 × 2 factor-type recall matrix. Rows indicate factor scope (within-trial vs. window); columns indicate factor class (discrete vs. continuous). Cell values are the mean recovery score across all factors of that type, all benchmarks, and all seeds. Color intensity encodes the recovery score (white = 0, dark blue = 1). The matrix reveals the predicted difficulty ordering: within-trial factors are consistently easier to discover than window factors, and discrete factors are consistently easier than continuous ones.

*Color code:* within-trial discrete (dark blue), window discrete (medium blue), within-trial continuous (red-orange, hatched), window continuous (salmon, hatched). Error bars: ±1 SE, five seeds.

---

## Figure 3 — Empirical Dataset Recovery

**AutoFactor recovers known derived factors in real behavioral datasets, with performance close to synthetic benchmarks of comparable complexity.**

**(A)** Precision, recall, and F1 for recovery of known ground-truth factors on [N] empirical behavioral datasets. The number of pre-specified ground-truth factors per dataset is shown above each bar group. Dashed gray lines indicate the F1 of the closest-complexity synthetic benchmark (Stroop-Simon for datasets with discrete sequential factors; RDK Task-Switching for datasets with mixed factor types), providing a reference for the degradation in performance attributable to real-data noise, individual differences, and missing trials. Bars show mean ± SE across five seeds.

**(B)** Per-factor recovery score for each known ground-truth factor, organized by dataset (bold labels, right side; horizontal rules separate datasets). Bar length and color follow the same conventions as Figure 2B: level recall for discrete factors (solid bars), |Spearman ρ| for continuous factors (hatched bars). A dotted vertical line marks the full-recovery threshold (0.95). Factors whose bar reaches or exceeds this threshold were consistently and fully recovered across seeds; shorter bars indicate partial or inconsistent recovery.

*Datasets shown:* Stroop congruency (N = 466 participants, 1 ground-truth factor); [Janker et al.] (N = [X] participants, [N] ground-truth factors); [Dataset 3] ([N] participants, [N] factors). Error bars: ±1 SE, five seeds.

---

## Figure 4 — Novel Factor Discovery on Empirical Datasets

**AutoFactor discovers novel derived factors with statistically supported effects on behavior.**

Each panel shows one novel factor discovered by AutoFactor in a real behavioral dataset — a factor that was *not* pre-specified as ground truth and was identified purely from the behavioral data. The **top subplot** shows the behavioral effect of the discovered factor: individual participant-level means are plotted as gray dots (jittered horizontally for visibility); group means ± 95% confidence intervals are shown as colored markers. The **bottom block** shows the machine-synthesized predicate code (`compute_factor()`) that defines the factor, together with its SweetPea factor definition (green block), which can be directly reused in the design of future experiments. The held-out log-likelihood gain (ΔLL, held-out 20% of participants) is reported below each panel; literature references are provided where an analogous factor has been reported previously.

**Ink Color Repetition** *(Stroop, N = 466).* Log response time is shorter when the ink color repeats from the previous trial than when it changes, independent of the trial's congruency (which is included in the null model). ΔLL = 13.2. *cf.* Mayr et al. (2003).

**Response Repetition Benefit** *([Janker et al.], N = [X]).* Accuracy is higher on trials where the required motor response repeats from the previous trial than on trials where it switches. ΔLL = 9.4.

**Task × Difficulty Interaction** *([Dataset 3], N = [X]).* The detrimental effect of high stimulus difficulty on accuracy is amplified on task-switch trials compared to task-repeat trials, consistent with a compounded cost of simultaneous reconfiguration and perceptual difficulty. Discovered as an interaction term by the Phase 2 effect search. ΔLL = 7.1.

All effects survive held-out validation (ΔLL > 0 on the 20% held-out participant set). [Placeholder panels marked "TBD" will be replaced with results from additional empirical datasets before submission.]

---

## Figure 5 — Ablation Study

**Each pipeline component contributes to factor recovery, with the largest gains on harder benchmarks.**

Factor recovery F1 under six system configurations, evaluated on Stroop-Simon (medium complexity, light blue) and RDK Task-Switching (high complexity, dark blue). The full system (top row) serves as baseline; dashed vertical lines mark the full-system F1 for each benchmark. Bars show mean ± SE across three seeds. Removing LLM seeding (replacing it with combinatorial random seeding) produces the largest drop on RDK Task-Switching, where the factor space is large and continuous factors are rare in random proposals. Removing evolution (a single seeding pass with no iterative refinement) reduces F1 on both benchmarks, with the larger deficit on RDK. The novelty bonus has a small positive effect on medium-complexity benchmarks and a negligible effect on the harder benchmark. Adding a hard complexity penalty reduces F1 on both benchmarks by penalizing legitimate multi-level and multi-dependency factors. Removing the Phase 2 interaction search reduces overall F1 by the amount attributable to discovered interaction terms.

*Conditions:* Full system; w/o LLM seeding (random seeder only); w/o evolution (no iterative refinement); w/o novelty bonus (novelty_weight = 0); + complexity penalty (complexity_exponent = 1.0); w/o interaction search (Phase 2 disabled). Error bars: ±1 SE, three seeds.

---

## Figure 6 — Search Efficiency

**LLM seeding reaches target factor recovery with fewer predicate synthesis calls than random seeding.**

Rolling-maximum factor F1 as a function of cumulative predicate synthesis calls, for three benchmarks (colors) and two seeding strategies (solid = LLM seeding, dashed = random seeding). Shaded ribbon = ±1 SE across five seeds. LLM seeding converges faster on all three benchmarks — particularly on RDK Task-Switching and Prospect Theory, where the factor space is large and continuous factors are rare under random sampling. The gap between LLM and random seeding widens with benchmark complexity, indicating that LLM seeding's domain knowledge advantage is most valuable when exhaustive search is infeasible. The final F1 achieved by random seeding (plateau of dashed lines) is consistently lower than that of LLM seeding, reflecting factors that random seeding never proposes.

*X-axis:* cumulative number of LLM predicate synthesis calls (does not include seeding calls or evolution proposals). *Y-axis:* best factor-level F1 achieved up to that call count (rolling maximum). Error bands: ±1 SE, five seeds.

---

## Notes for Submission

- All mockup numeric values in panels A/B (Figures 2 and 3), all data points (Figures 4–6), and all F1 values (Figure 5 ablation) must be replaced with experimental results before submission.
- Figure 4 discovery panels will be updated once all empirical datasets are collected and analyzed. Currently shows [Stroop], [Janker et al.], and [Dataset 3 — TBD].
- Figure 2B row ordering (within each benchmark) will be refined to match the actual observed recovery gradient from experimental results; factors may be reordered to highlight the clearest pattern.
- Interaction terms need to be added to the ground-truth statistical models before interaction F1 bars can be added to Figure 2A.
