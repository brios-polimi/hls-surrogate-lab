# E5c results post-mortem

## Completion and provenance

The run at
`artifacts/results/e5c_honest_budget_attribution_v1_run_20260915` is complete.
All 35 architecture--draw bundles finished fitting and evaluation with zero
non-zero return codes. The expected 3,465 method fits, 3,465 evaluations, 2,240
neural checkpoints, 1,225 calibration artifacts, and seven zero-shot
evaluations are present. No failure signature was found in the worker logs.

The run records commit `832a7be` with a dirty worktree. Its bundled provenance
copy of `run_e5c.py` is therefore the authoritative executed analyzer. In
particular, it uses 98.75% per-claim curve intervals (97.5% family-wise coverage
over two claims) and 99.8% per-budget intervals (99% family-wise coverage over
five budgets). The checked-in analyzer and frozen protocol document have been
synchronized to those executed settings.

This is a late-added controlled follow-up. It was specified after the earlier
E5 query results had been inspected and reuses the same fixed query cohort. The
query was sealed during the complete E5c fit matrix, and implementation smoke
tests were outcome-blind, but the study is not fresh external confirmation.

## Registered claims

Positive effects favour the named treatment. The unit of generalization is the
architecture; initialization replicates are averaged before support draws, and
architectures receive equal weight.

| Claim | Curve-average effect | Adjusted interval | Wins | Exact p | Holm p | Decision |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| P1: source-head tuning over zero-shot | +25.77 | [17.25, 37.61] | 7/7 | 0.0156 | 0.0313 | passes |
| P2: pretrained over random encoder with a fresh head | -3.33 | [-7.47, 1.14] | 3/7 | 0.125 | 0.125 | fails |

P1 is strong and correctly scoped: under the fixed source checkpoint,
architecture cohort, and adaptation schedules, tuning a copy of the
source-trained head improves query SMAPE over exact total-label budgets from 4
to 64. It does not by itself identify which pretrained component caused the
gain.

P2 supplies that missing attribution and changes the thesis interpretation.
With the source head removed, learned encoder weights do not improve adaptation
on average over the registered curve. The previous pretrained-head-versus-full-
scratch contrast confounded encoder state, head initialization, and adaptation
interface and should not be described as encoder attribution.

## Budget-specific results

Source-head query SMAPE decreases monotonically from 46.90 at `k=4` to 13.38
at `k=64`, from a 55.39 zero-shot baseline. The registered zero-shot-minus-head
effects pass at every budget:

| k | Effect | 99% family-wise interval | Wins |
| ---: | ---: | --- | ---: |
| 4 | +8.49 | [2.19, 16.22] | 7/7 |
| 8 | +17.04 | [7.53, 30.50] | 7/7 |
| 16 | +26.49 | [16.54, 41.07] | 7/7 |
| 32 | +34.81 | [22.78, 53.70] | 7/7 |
| 64 | +42.01 | [28.84, 62.90] | 7/7 |

The fresh-head encoder contrast crosses over with label count:

| k | Random minus pretrained SMAPE | 99% family-wise interval | Pretrained wins |
| ---: | ---: | --- | ---: |
| 4 | -10.21 | [-20.26, 2.89] | 0/7 |
| 8 | -10.40 | [-19.05, -0.65] | 1/7 |
| 16 | -3.78 | [-11.85, 5.51] | 3/7 |
| 32 | +3.61 | [-2.35, 10.36] | 5/7 |
| 64 | +4.11 | [1.17, 7.42] | 7/7 |

Thus the pretrained encoder is not a generally better few-shot feature space
under the tested fresh-head optimizer. There is positive, architecture-
consistent evidence at `k=64`, which should be retained as a moderate-budget
result rather than generalized backward to `k=4`--32.

The optimization-independent residual-ridge control supports the same reading.
Its pretrained-versus-random feature contrast is inconclusive through `k=32`
and +1.37 [0.66, 2.12] at `k=64`, with 7/7 architecture wins. Agreement of the
neural and convex controls at 64 makes the late positive result more credible.

## Mechanism and operational comparisons

Source-head initialization is the dominant low-budget contribution. Relative
to a fresh head on the same pretrained encoder, retaining the source head gains
51.21, 45.82, 33.99, 20.89, and 5.80 SMAPE points as `k` increases. Every
architecture favours it at every budget, and all hierarchical 95% intervals
exclude zero. The effect shrinks with more labels, as expected if target data
eventually rebuild the discarded source mapping.

The practical ranking also converges with budget. At `k=32`, source-head tuning,
pretrained residual ridge, and scratch training obtain 20.58, 23.86, and 25.04
SMAPE. At `k=64`, they obtain 13.38, 14.23, and 14.61; paired source-head
differences from ridge and scratch include zero. Source-head tuning is therefore
the best tested low-budget interface, not a uniquely superior moderate-budget
method.

Full pretrained fine-tuning is poor under its registered schedule: 41.24 and
32.82 SMAPE at `k=32` and 64. Scratch is better by 16.20 [5.86, 30.90] and
18.22 [10.94, 28.00], respectively, with 7/7 wins. This diagnoses negative
transfer or optimization failure for the fixed `1e-5`, 200-epoch procedure. It
does not establish that all regularized or layer-wise fine-tuning must fail.

At `k=64`, source-head tuning improves all six targets on all seven
architectures. FF remains hardest at 31.60 SMAPE. Scratch is slightly better on
the four-resource aggregate (13.08 versus 13.81), while pretrained residual
ridge is better on timing (10.81 versus 12.52). These differences make method
selection target-dependent despite similar overall SMAPE.

## Defensible thesis conclusion

The useful positive claim is label-efficient reuse of the jointly trained
source encoder--head state. The experiment does not support the stronger claim
that the learned encoder alone causes the few-shot advantage. Source-head
initialization carries most of the benefit below 64 labels; learned features
show a smaller, replicated advantage only at the moderate 64-label endpoint.

The principal external-validity limits are the seven fixed exemplar
architectures, one source checkpoint, one Vitis-derived data contract, and reuse
of a previously inspected query cohort. The raw negative result is nevertheless
valuable: it identifies where the earlier causal story was too broad and gives
future work a precise target---learn representations that are usable by a newly
initialized low-data head, or design adaptation that explicitly preserves the
source head's predictive mapping.
