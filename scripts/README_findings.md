## 11. Findings

Every claim below is traceable to a CSV under `results/`. Where a comparison
was not statistically significant after Holm correction, it is reported as "no
detectable difference" rather than as a win.

### 11.1 The headline: conditional usefulness, not superiority

Across 153 Holm-corrected paired comparisons spanning three metrics
(`results/tables/all_paired_comparisons.csv`), the neural model does **not**
dominate:

| Metric | Significant, favours NeuralRep | Significant, favours a meta-learner |
|---|---|---|
| PEHE | 39 | 5 |
| Absolute ATE error | 18 | 12 |
| Policy regret | 34 | 12 |

The pattern behind those counts is systematic, not noise. On the **synthetic**
benchmark, three different estimators win three different metrics:

- **NeuralRep** has the lowest PEHE — but only under design stress (see 11.2).
- **X-Learner** has lower ATE error than NeuralRep in *every* synthetic
  condition tested (all γ, all imbalance levels, all sample sizes; 12/12
  significant).
- **S-Learner** has lower policy regret than NeuralRep in *every* synthetic
  condition tested (12/12 significant).

On **IHDP**, by contrast, NeuralRep is never significantly beaten on any metric
in any condition. IHDP is small (672 training units), heavily confounded, and
18% treated — precisely the stressed regime identified in 11.2.

### 11.2 When representation learning helps, and when it does not

Experiment 2 produces a clean, well-powered **crossover** with the true ATE
held constant at 0.9956 across all conditions:

| γ | overlap (units with e<0.1 or e>0.9) | X-Learner PEHE | NeuralRep PEHE | Holm verdict |
|---|---|---|---|---|
| 0.0 | 0% | **0.431** | 0.474 | NeuralRep significantly **worse** (p<0.0001) |
| 1.0 | 3.6% | 0.502 | 0.494 | no detectable difference (p=0.164) |
| 2.0 | 27% | 0.580 | **0.538** | NeuralRep significantly **better** (p<0.0001) |
| 3.0 | 45.5% | 0.628 | **0.558** | NeuralRep significantly **better** (p<0.0001) |

Under randomization the shared representation is a liability; under strong
confounding and degraded overlap it is an advantage. Experiment 3 shows the
same shape for imbalance: NeuralRep beats X-Learner on PEHE only at the
strongest imbalance (10% treated, p=0.004), with no detectable difference at
25% or 50%.

**Effect sizes are modest.** The significant PEHE gaps versus X-Learner are
0.043–0.069 PEHE units on a base of roughly 0.5, i.e. 8–14% relative. These are
consistent and well-powered, not large.

### 11.3 Sample size does not drive the result — design stress does

The two sample-size experiments point in opposite directions, and the conflict
is informative rather than contradictory:

| Benchmark | Regime | Relative PEHE improvement, 25% → 100% | Outcome at largest n |
|---|---|---|---|
| IHDP (168 → 672) | γ high, 18% treated | NeuralRep **−49%**, X-Learner −31% | NeuralRep significantly better |
| Synthetic (2000 → 8000) | γ = 1.0, balanced | NeuralRep −26.7%, X-Learner **−35.6%** | X-Learner significantly better, gap **widening** with n |

More data does not rescue the neural model in an easy regime; it lets the
X-Learner pull further ahead (Δ grows +0.037 → +0.048 as n goes 4000 → 8000).
The determining factor across all experiments is the **difficulty of the
design** — confounding strength and treatment imbalance — not the amount of
data.

## 12. Ablation: which component actually matters

Experiment 5, 30 seeds, paired by seed, Holm-corrected within each metric.
Positive Δ means removing the component made the model worse.

**PEHE change relative to the full model**

| Component removed | IHDP (n=672) | Synthetic (n=4000) |
|---|---|---|
| Treatment-specific heads | **+126.9%** (p<0.0001) | **+31.4%** (p<0.0001) |
| Learned representation | +6.3% (p=0.0020) | +16.2% (p<0.0001) |
| Regularization (dropout + weight decay) | −2.5% (p=0.477, **ns**) | +1.2% (p=0.339, **ns**) |

Three conclusions replicate across both benchmarks:

1. **The treatment-specific heads are the dominant component**, not the learned
   representation. Removing them on IHDP more than doubles PEHE and inflates
   ATE error by 404%. The variant without them is exactly a *neural
   S-Learner*, and its collapse mirrors the S-Learner's poor showing in
   Experiment 1 — an internally consistent cross-check.
2. **The learned representation is a real but secondary contributor**, and its
   importance *grows with data* (+6.3% on 672 units, +16.2% on 4000). This
   coheres with the IHDP sample-size trend in 11.3.
3. **Regularization is inert.** Dropout and weight decay are not significant on
   any metric on either benchmark. Sensitivity analysis agrees: weight decay is
   the least influential hyper-parameter examined (11.6% PEHE spread).

**This partially contradicts the project's own framing.** The hypothesis was
that a shared representation is the mechanism behind the neural model's
behaviour. The evidence says the two-head architecture does most of the work,
and the representation adds a smaller, genuine increment that matters more as
data grows. That ordering is the single most important empirical result here.

One caveat against over-reading: component contributions are metric-dependent.
On synthetic data, removing the representation *improved* ATE error by 34%
(p=0.064, ns) while tripling policy regret (+223%, p<0.0001).

## 13. Sensitivity analysis

Experiment 6, IHDP, 10 seeds, one hyper-parameter varied at a time. Total PEHE
spread across each swept range:

| Hyper-parameter | Range swept | PEHE spread | Best value |
|---|---|---|---|
| Learning rate | 1e-4 → 1e-2 | **36.9%** | 0.005 |
| Latent dimension | 4 → 64 | 17.3% | 16 |
| Epochs | 25 → 400 | 15.5% | 200 |
| Weight decay | 0 → 1e-2 | 11.6% | 0.01 |

Learning rate matters roughly three times as much as weight decay. Performance
is flat between latent dimensions 8 and 64, so the representation's *width* is
not critical — consistent with the ablation finding that the encoder is a
secondary component. Results beyond 100 epochs are essentially flat, indicating
the early-stopping budget is adequate rather than binding.

The reported baseline configuration (lr=0.001, latent_dim=32) is **not** the
best setting found in this sweep (lr=0.005, latent_dim=16 were better). No
result in this project was produced with a tuned configuration, and the
baseline was fixed before the sweep was run.

## 14. Error analysis and limitations

### 14.1 Error structure

Per-replicate IHDP errors are strongly right-skewed: NeuralRep's mean PEHE is
2.008 but its median is 0.926, because a handful of realizations have very
large outcome scales. All tables therefore report mean, SD *and* median, and
all significance testing uses the rank-based Wilcoxon signed-rank test rather
than a t-test. Mean and median occasionally disagree about the winner — on
IHDP sample size, T-Learner has lower *median* ATE error than NeuralRep at
three of four sizes while NeuralRep wins on the mean. Both are reported.

### 14.2 Limitations

1. **Hyper-parameter fairness is asymmetric — the most important caveat.**
   NeuralRep receives adaptive model selection (early stopping on a 20%
   validation split); the S/T/X-Learners use fixed default hyper-parameters
   with no tuning. NeuralRep also therefore trains on 20% less data. Because of
   this, cross-model results should be read as comparing *these configured
   estimators*, not as isolating representation learning per se. The
   Experiment 5 ablation is unaffected — every variant shares identical early
   stopping — which is why the ablation, not the baseline comparison, carries
   the mechanism claim.
2. **No covariate-shift experiment exists.** Covariate shift between arms is a
   consequence of the confounding manipulation, never an independently
   manipulated factor. Nothing here demonstrates robustness to distribution
   shift as such.
3. **Unconfoundedness holds by construction** in both benchmarks; every
   confounder is observed. Nothing here speaks to hidden confounding, and no
   sensitivity analysis to unobserved confounding (e.g. Rosenbaum bounds) was
   performed.
4. **Two benchmarks only.** IHDP is a single small semi-synthetic dataset whose
   quirks are well documented, and the synthetic DGP is one specific functional
   form (partially linear with a mild nonlinearity). ACIC and other benchmarks
   are supported by the framework but were not run.
5. **Meta-learners use one base learner.** All three wrap gradient boosting;
   conclusions about S/T/X may not transfer to other base regressors.
6. **Experiment 6 uses 10 seeds**, not 30, and varies one hyper-parameter at a
   time, so interactions between hyper-parameters are not explored.
7. **Effect sizes are modest** where significant (8–14% relative PEHE), so
   statistical significance should not be read as practical importance.
