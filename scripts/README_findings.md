## 11. Findings

Every claim below is traceable to a CSV under `results/`. Where a comparison
was not statistically significant after Holm correction, it is reported as "no
detectable difference" rather than as a win.

**Read Experiment 7 first.** Experiments 1–6 gave the neural model adaptive
capacity control (early stopping) while the meta-learners ran on fixed
defaults. Experiment 7 removes that asymmetry, and it overturns the headline
conclusion those experiments appeared to support. The subsections below are
ordered accordingly.

### 11.1 The headline: no reliable advantage under equal tuning budgets

Under an identical model-selection budget for all four estimators
(Experiment 7, 30 seeds), the neural model does **not** outperform a tuned
X-Learner on the synthetic benchmark:

| Scenario | X-Learner PEHE | NeuralRep PEHE | Δ | Holm verdict |
|---|---|---|---|---|
| γ = 0 | **0.4269** | 0.5775 | +0.1506 | NeuralRep significantly **worse** (p<0.0001) |
| γ = 2 | **0.5812** | 0.6096 | +0.0284 | NeuralRep significantly **worse** (p=0.0001) |
| γ = 3 | 0.6336 | 0.6232 | −0.0104 | **no detectable difference** (p=0.184) |
| 10% treated | 0.6819 | 0.7015 | +0.0196 | **no detectable difference** (p=0.262) |

On **absolute ATE error** the neural model is significantly worse than the
X-Learner in every synthetic scenario, and the gap widens with confounding:
+47% at γ=0, +72% at γ=2, +93% at γ=3 (all p≤0.024).

The one scenario where the neural model leads under fair selection is **IHDP**,
where it attains PEHE 1.402 ± 2.186 (median 0.717) against the X-Learner's
3.749 ± 6.847 (median 1.214), p<0.0001. IHDP is small (672 training units),
heavily confounded and ~18% treated. Its ATE-error advantage there is *not*
significant (p=0.109).

### 11.2 The apparent confounding crossover did not survive fair selection

Experiment 2 (fixed hyper-parameters) appeared to show a crossover: the neural
model overtaking the X-Learner on PEHE as confounding strength increased,
significantly better at γ=2 and γ=3. **That result does not replicate once both
models receive the same selection budget.** Under Experiment 7 the neural model
is significantly *worse* at γ=0 and γ=2 and merely indistinguishable at γ=3 —
there is no γ in the tested range at which it is significantly better.

The direction of travel is still visible: the X-Learner's PEHE advantage
narrows from +35.3% at γ=0 to a statistical tie at γ=3. But narrowing is not
crossing, and **no PEHE advantage under confounding is claimed here**. The
original crossover is best explained as an artifact of asymmetric tuning.

The confounding manipulation itself remains valid and is reported in
Experiment 2: with the true ATE held constant at 0.9956, maximum standardized
mean difference rises 0.106 → 1.031 and the share of units in the propensity
tails rises 0% → 45.5%. What Experiment 2 manipulates is **confounding
strength, treatment-selection bias and overlap/positivity degradation** — not
covariate shift as an independently controlled factor.

### 11.3 Sample size does not drive the result

| Benchmark | Regime | Relative PEHE improvement, 25% → 100% | Outcome at largest n |
|---|---|---|---|
| IHDP (168 → 672) | high γ, 18% treated | NeuralRep **−49%**, X-Learner −31% | NeuralRep better (fixed hyper-parameters) |
| Synthetic (2000 → 8000) | γ = 1.0, balanced | NeuralRep −26.7%, X-Learner **−35.6%** | X-Learner better, gap **widening** with n |

More data does not rescue the neural model in an easy regime; it lets the
X-Learner pull further ahead (Δ grows +0.037 → +0.048 as n goes 4000 → 8000).
Note both rows come from Experiments 4a/4b, which used fixed hyper-parameters;
they were not re-run under fair selection.

## 12. Experiment 7 — fair model selection

### 12.1 Motivation

In Experiments 1–6 the neural model chose its own effective capacity via early
stopping on a validation split, while S/T/X-Learners used fixed defaults with
no tuning at all. Any cross-model result under that design compares *these
particular configurations*, not the modelling strategies. Experiment 7 exists
to determine which conclusions survive when the budget is equalised.

### 12.2 Protocol

For each (scenario, seed), every estimator receives:

- the **same six candidate configurations** (`src/experiments/tuning.py`);
- the **same train/validation split**, derived from the same seed;
- the **same selection criterion**;
- a **refit of the selected configuration on the full training set**;
- **no adaptive early stopping** — the neural model's epoch budget (100/200/400)
  is a tuned hyper-parameter like any other.

The three meta-learners share one gradient-boosting grid, so differences among
them remain attributable to meta-learning strategy rather than candidate sets.

### 12.3 Why factual validation MSE, and why not PEHE

Selection uses the mean squared error of the predicted outcome under each
unit's **observed** treatment. PEHE cannot be used: it requires counterfactual
outcomes that no practitioner observes, so selecting on it would be oracle
selection and would leak the evaluation target into training. Factual outcome
error is the only criterion computable from observable data. It is an imperfect
proxy for CATE accuracy — imperfect *equally* for all four estimators, which is
what makes the comparison fair.

### 12.4 Validity correction

A first execution of Experiment 7 was discarded. Meta-learner hyper-parameters
were not reaching the underlying regressors: `build_learner` filtered the
overrides, and the meta-learner constructors did not forward them to the base
estimator. All six candidates were therefore identical, and only the neural
model was genuinely tuned — biasing the comparison in the direction the
experiment was designed to remove. The tell was that meta-learner results
matched the untuned Experiment 2 numbers to four decimal places.

Both defects were fixed, `tune_and_fit` now raises when every candidate scores
identically, and twelve regression tests cover the forwarding path. Experiments
1–6 were verified byte-identical after the fix (maximum change 0.00e+00), since
they never passed overrides to meta-learners. The results reported here come
solely from the corrected run.

### 12.5 Verification

- All 30 seeds present in every cell; identical seed sets across models within
  each scenario; zero failed runs (59/59 statistical checks).
- Selected configurations vary for **all four** models (5–6 distinct
  configurations each), confirming that selection is doing real work.
- Recorded `selected_config` values were re-instantiated and checked against
  the fitted estimators' actual attributes — `n_estimators`, `max_depth`,
  `learning_rate` for the meta-learners; `latent_dim`, `lr`, `weight_decay`,
  `epochs` for the neural model — all match, with `early_stopping=False`
  confirmed.
- Meta-learner results now differ from the untuned Experiment 2 values in all
  nine comparable cells.

### 12.6 Conclusion

The neural model's apparent advantage under confounding in Experiment 2 was an
artifact of asymmetric model selection. Given equal budgets it does not
outperform a tuned X-Learner on this synthetic DGP, on either PEHE or ATE
error. Its advantage on IHDP persists under fair selection.

## 13. Ablation: which component actually matters

Experiment 5, 30 seeds, paired by seed, Holm-corrected within each metric.
Positive Δ means removing the component made the model worse.

**PEHE change relative to the full model**

| Component removed | IHDP (n=672) | Synthetic (n=4000) |
|---|---|---|
| Treatment-specific heads | **+126.9%** (p<0.0001) | **+31.4%** (p<0.0001) |
| Learned representation | +6.3% (p=0.0020) | +16.2% (p<0.0001) |
| Regularization (dropout + weight decay) | −2.5% (p=0.477, **ns**) | +1.2% (p=0.339, **ns**) |

In this implementation, **treatment-specific outcome heads contribute far more
than the shared representation**. Removing the heads on IHDP more than doubles
PEHE and inflates ATE error by 404%; removing the encoder costs 6.3%. The
variant without separate heads is exactly a *neural S-Learner*, and its
collapse mirrors the S-Learner's poor showing in Experiment 1 — an internally
consistent cross-check.

The learned representation is a real but secondary contributor, and its
importance grows with data (+6.3% on 672 units, +16.2% on 4000). Regularization
is inert: not significant on any metric on either benchmark, which the
sensitivity sweep corroborates (weight decay is the least influential
hyper-parameter examined, 11.6% PEHE spread).

For a project premised on representation learning, this ordering is the most
informative result obtained: the architectural separation of treatment arms,
not the shared encoder, accounts for most of the model's behaviour.

Two caveats. Component contributions are metric-dependent — on synthetic data,
removing the representation *improved* ATE error by 34% (p=0.064, ns) while
tripling policy regret. And the ablation was run with fixed hyper-parameters;
it was not repeated under fair selection. Its internal comparison remains valid
because all four variants share an identical training protocol, including
early stopping.

## 14. Policy regret is not used as evidence here

The policy-value and policy-regret implementation was audited and is correct:
manual recomputation matches the module to 0.00e+00, regret equals oracle value
minus policy value, regret is non-negative, and the oracle dominates every
learned and constant policy. Thirteen unit tests cover decision direction, sign
inversion, threshold strictness, treatment-label consistency and baselines.

**The metric nevertheless lacks discriminating power in this DGP**, so no model
ranking is drawn from it:

- Only **~3.79%** of true treatment effects are negative, so treating everyone
  is close to optimal.
- The trivial **always-treat** policy already achieves regret **≈0.0082**.
- The neural model's learned policy treats **≈100%** of units, so it
  essentially reproduces always-treat, scoring 0.0083–0.0104.
- The X-Learner, estimating effects more accurately, withholds treatment from
  ~5% of units and is wrong often enough to score *worse* than the trivial
  baseline.

A low regret here therefore reflects a degenerate policy rather than better
treatment decisions. Estimating τ(x) and choosing whom to treat are different
problems: the policy depends only on sign(τ̂), not magnitude, so a model with
large but uniformly positive errors can match the best constant policy. **No
claim of superior policy performance is made for any model.** `evaluate_cate`
now reports the always-treat and never-treat baselines and the learned policy's
treated fraction so this degeneracy is visible directly in future results.

## 15. Error analysis and limitations

### 15.1 Error structure

Per-replicate IHDP errors are strongly right-skewed: under fair selection the
neural model's mean PEHE is 1.402 but its median is 0.717, because a few
realizations have very large outcome scales. All tables report mean, SD *and*
median, and all significance testing uses the rank-based Wilcoxon signed-rank
test rather than a t-test. Mean and median occasionally disagree about the
winner; both are reported.

### 15.2 Limitations

1. **Weak sign heterogeneity makes policy evaluation uninformative.** With only
   ~3.8% of effects negative, the decision problem is nearly trivial and
   always-treat is near-optimal. A DGP with substantially more sign
   heterogeneity would be required before policy regret could rank estimators.
   This is the most consequential limitation for the decision-making question.
2. **Two benchmarks only.** IHDP is a single small semi-synthetic dataset with
   well-documented quirks, and the synthetic DGP is one specific functional form
   (partially linear with a mild nonlinearity). ACIC and other benchmarks are
   supported by the framework but were not run.
3. **One base learner for all meta-learners.** S/T/X all wrap gradient
   boosting; conclusions about them may not transfer to other base regressors.
4. **No representation-balancing penalty.** The neural model is a plain
   shared-encoder architecture. CFR/IPM-style balancing — the mechanism the
   representation-learning literature actually proposes — was not implemented,
   so this project tests shared representations, not balanced ones.
5. **Experiments 1–6 retain the tuning asymmetry.** Experiment 7 corrects it for
   five key scenarios, but the sample-size, ablation and sensitivity results
   were not re-run under fair selection. Their cross-model comparisons should be
   read with that caveat; the ablation's internal comparison is unaffected.
6. **Experiment 6 uses 10 seeds**, not 30, and varies one hyper-parameter at a
   time, so hyper-parameter interactions are unexplored.
7. **Effect sizes are modest** where significant, so statistical significance
   should not be read as practical importance.
