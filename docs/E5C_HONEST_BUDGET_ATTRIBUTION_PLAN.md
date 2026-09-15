# E5c: honest-budget adaptation and representation attribution

Status: frozen before E5c training. This is a late-added study designed after
inspection of the original E5 query results, so it is not represented as fresh
external validation. The inherited E5 query sets are unchanged. The production
orchestrator seals outcome evaluation until the complete registered fit matrix
succeeds. Before final freezing, temporary implementation smoke tests generated
predictions for shape, finiteness, and artifact-pairing checks on one architecture;
no error metric or treatment comparison from those predictions was inspected.

## Questions and permissible claims

P1 asks whether target adaptation helps: across exact total-target-label budgets
`k = {4, 8, 16, 32, 64}`, does tuning the pretrained source head reduce
equal-architecture query SMAPE relative to zero-shot transfer?

P2 asks what pretraining contributed: across the same budgets, does a fresh head
trained on source-standardized frozen pretrained features outperform an
identically initialized fresh head trained on source-standardized frozen random
features of the same architecture?

P2 isolates the effect of the learned encoder state under the tested source task,
architecture, optimizer, schedule, labels, support order, and head initialization.
It does not prove that every benefit came from a particular source example or
that the pretraining procedure is optimal. The exact defensible headline, if the
registered test passes, is:

> Source-pretrained encoder weights improve target-architecture adaptation over
> matched random encoder weights across total-label budgets from 4 to 64.

`k=4` through `k=32` form the few-shot regime. `k=64` is deliberately included
as a moderate-budget persistence test and is not relabeled as few-shot.

## Data and label accounting

- The seven E5 exemplar architectures and every original E5 query path remain
  fixed.
- The old adaptation-validation and support subsets are recombined into one
  adaptation pool. No query example enters it.
- Each reported `k` is exactly the number of target labels available to the
  method. There is no extra target validation set.
- Five deterministic support orders (`7, 42, 137, 271, 911`) produce nested
  supports at `k = 4, 8, 16, 32, 64`.
- The production query analysis runs only after all 3,465 registered fits
  complete. Pre-run smoke checks were outcome-blind as disclosed above.

## Methods

| Method | Purpose |
| --- | --- |
| Zero-shot source model | No-adaptation reference |
| Identity-regularized affine calibration | Low-variance output calibration |
| Pretrained-feature residual ridge | Convex use of the frozen representation |
| Random-feature residual ridge | Optimization-independent matched attribution check |
| Final-layer tuning | Minimal update of the pretrained source head |
| Source-head tuning | Pretrained encoder plus pretrained complete head |
| Fresh head, standardized pretrained features | Scale-controlled representation without source-head initialization |
| Fresh head, standardized random features | Scale-controlled encoder-state attribution control |
| Full fusion model from scratch (`k=32,64`) | Operational target-only baseline, not an attribution control |
| Full pretrained fusion-model tuning (`k=32,64`) | Standard practical adaptation baseline |

Affine and ridge penalties are selected by leave-one-out cross-validation using
only the `k` support labels, followed by refitting on all `k`. Neural methods use
a fixed, query-independent schedule: 40, 60, 80, 120, and 160 epochs for
`k=4,8,16,32,64`. The schedule was chosen from the scale of the old E5 learning
histories before E5c results. It is identical for paired attribution arms.
Scratch and pretrained full-model baselines use 200 fixed epochs; pretrained
full tuning retains E5's conservative `1e-5` learning rate.

Fresh-head comparisons use five paired replicates. For each replicate the head
initial state, minibatch seed, loss, optimizer, and epoch count are identical;
only the frozen encoder cache differs. Before fresh-head training, each encoder's
features are standardized using its own unlabeled source-test feature statistics
with a fixed `1e-3` standard-deviation floor. This prevents raw activation scale
from becoming an optimizer confound and uses no target labels or target-query
statistics. Completion artifacts record the initial-head, feature-cache, and
normalizer hashes. The random encoders use seeds 101, 202, 303, 404, and 505.

The same pretrained/random comparison is repeated with residual ridge. This
convex control distinguishes representation quality from a favorable neural-head
optimization trajectory; it is secondary rather than another primary claim.

## Estimands and inference

The unit of generalization is the architecture, not the design point. Query
SMAPE is computed pointwise with the stabilized benchmark denominator. For
replicated methods, initialization replicates are averaged first, then support
draws, then the seven architectures with equal weight.

Each of P1 and P2 has one primary curve estimand: the equal-weight mean paired
SMAPE advantage over all five log2-spaced budgets. Its uncertainty uses a
hierarchical bootstrap that resamples architectures and then support draws within
architecture. The two primary claims use 97.5% intervals and Holm-adjusted exact
sign-flip tests over the seven architecture-level curve effects.

A strong curve-level claim requires all three:

1. the adjusted interval excludes zero in the favorable direction;
2. Holm-adjusted `p < 0.05`;
3. at least six of seven architecture means favor the treatment.

All five budget-specific effects are also registered. Within each claim they use
Bonferroni 99% intervals. A budget-specific headline additionally requires at
least six of seven architecture means in the same direction. Ordinary 95%
intervals, architecture win counts, per-target metrics, and the complete raw
prediction table are reported regardless of outcome.

P3, source-head tuning versus a fresh head on the same pretrained encoder,
decomposes the benefit of source-head initialization. Affine, residual ridge,
final-layer tuning, and scratch training are operational secondary comparisons.

## Outcome interpretation, including negative results

- If both neural-head and residual-ridge attribution contrasts favor pretraining,
  the learned encoder state is useful under both nonlinear and convex adaptation.
- If only residual ridge favors pretraining, the representation contains useful
  information but the neural few-shot optimizer is unreliable. If only the
  neural head favors it, the advantage is not linearly accessible and should not
  be described as a generally superior feature space.
- If neither attribution contrast favors pretraining, E5c does not support an
  encoder-representation claim; any source-head gain should instead be attributed
  to source predictions or head initialization.
- If affine or residual calibration beats source-head tuning, the practical
  conclusion is that low-variance calibration is preferable at that budget.
- Pretrained full tuning versus head-only tuning at `k=32,64` determines whether
  updating the encoder is operationally justified. Scratch establishes whether
  transfer is useful at all under the same total-label accounting.

## Workload and time budget

There are 35 architecture-draw bundles. The registered matrix contains 2,100
small head fits (about 260,400 optimizer steps), 175 affine fits, 1,050 residual
ridge fits, 70 scratch fits, and 70 pretrained full-tuning fits. The two graph
methods contribute about 168,000 optimizer steps. Historical E5 and E5b timings
imply roughly 20--24 aggregate GPU-hours plus under one aggregate CPU-hour for
the extra ridge controls. Four concurrent workers on the RTX PRO 6000 are
expected to finish in 8--12 wall-clock hours, with a 14--16 hour pessimistic
allowance for contention, graph loading, cache creation, and evaluation.
