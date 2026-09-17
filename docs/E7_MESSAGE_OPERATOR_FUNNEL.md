# E7 hierarchical architecture funnel

## Decision and estimand

E7 first asks whether a better **local message/update operator** improves
exact-architecture transfer when the proven hierarchy is held fixed. Separate
performance, attention, and first-class-variable tracks then explore stronger
surrogates without pretending to isolate one local mechanism. The frozen
parts are input projection, pragma/operand injection, instruction-to-block and
block-to-function pooling, leaf-first call composition, callee injection, the
entry/reachable root readout, global/context fusion, resource/timing heads,
target transforms, optimizer, and checkpoint selection.

The primary confirmatory estimand is candidate minus canonical LLVM-Hier
six-target SMAPE on the E2 test architectures, paired by design and training
seed. Negative values favour the candidate. The screen uses the analogous E2
**validation** contrast and never evaluates test or exemplar labels.

"De-risk" is the right term for the screen: it removes broken, unstable, or
clearly unpromising mechanisms. It is not a small-data architecture contest and
its ranking is not a thesis result.

## Mechanism ladder

The first pool is deliberately a small factorial/ladder. Every candidate keeps
hidden dimension 64 and three instruction and block layers.

| ID | Change from canonical local layers | Direct comparison | Priority |
|---|---|---|---|
| `canonical` | Existing forward mean messages, sum, ReLU residual | Reference | required |
| `relation_mixer` | Concatenate relation messages and learn a low-rank correction to their sum | Does delayed learned relation mixing help without gates? | secondary |
| `receiver_gate` | Receiver-conditioned gate for each relation/flow | `receiver_gate - canonical`: gating main effect | high |
| `bidirectional` | Forward/reverse flows with shared relation projection and learned reverse modulation | `bidirectional - canonical`: direction main effect | high |
| `gate_bidir` | Receiver gates plus forward/reverse flows | Factorial interaction relative to the two main effects | high |
| `gate_bidir_meanmax` | Add relation/flow-specific mean and max aggregation | Aggregator effect conditional on `gate_bidir` | high |
| `gate_bidir_gru` | Replace residual update with a low-rank GRU-style update | Update effect conditional on `gate_bidir` | secondary |
| `full_operator` | Gate, bidirectionality, mean/max, and GRU-style update | Update effect conditional on `gate_bidir_meanmax` | secondary |
| `pna` | Rank-8 PNA aggregation | Multi-statistic/degree-scaling effect | high |
| `pna_gate_bidir` | Rank-8 PNA plus gates and reverse flow | Strong PNA mechanism bundle | secondary |
| `wide_receiver_gate` | Full-width receiver gates | Capacity-seeking gate model | performance |
| `wide_gate_bidir` | Full-width gates plus reverse flow | Capacity-seeking directional model | performance |
| `wide_gate_bidir_meanmax` | Full-width gates, reverse flow, and mean/max corrections | Strong residual-update model | performance |
| `wide_full_operator` | Full-width gates and mean/max plus standard GRU cells | Most expressive local operator | performance |
| `pna_wide` | Full-width PNA aggregation | Capacity-seeking PNA model | performance |
| `pna_wide_full` | Full-width PNA, gates, reverse flow, and standard GRU cells | Maximal PNA operator | performance |

This ladder distinguishes the proposed ingredients rather than testing only an
inseparable bundle. Reverse flow is tested at both instruction and block scale;
if it wins, a later localization pair (instruction-only versus block-only)
should be run before making a scale-specific claim. The same rule applies to
gating.

Full-width gates and GRU cells would turn the study into a capacity comparison.
The implementation therefore uses rank-8 gate, aggregate-correction, mixing,
and GRU-style maps. At the E2 width, candidate totals range from 240,848
(bidirectional, +0.5%) through 269,072 (gated bidirectional mean/max, +12.3%)
to 297,872 (the full bundle, +24.3%), versus 239,696 for canonical.

Parameter growth is not prohibited. The separate wide track contains models
from 314,000 through 907,280 parameters. Its purpose is the practical
accuracy--cost frontier, not clean mechanism attribution. PNA means
mean/max/min/std aggregation followed by identity, amplification, and
attenuation degree scalers. Average log-degree constants are computed only from
the selected training manifest and persisted in the run configuration.

## Attention track

The attention models keep containment, call composition, pooling, readout, and
heads fixed. Attention is quadratic only within one known group, and buckets
cap padded pair cost. Sinusoidal positions use the instruction/block order
already present in the frozen tensors.

| ID | Attention placement | Local operator | E2 parameters |
|---|---|---|---:|
| `attn_instruction` | full self-attention among instructions within each block | canonical-equivalent forward mean | 348,048 |
| `attn_block` | full self-attention among blocks within each function | canonical-equivalent forward mean | 348,048 |
| `attn_dual` | both containment scales | canonical-equivalent forward mean | 456,400 |
| `attn_dual_wide_operator` | both containment scales | `wide_full_operator` | 755,344 |

These are hierarchy/composition experiments, not message-operator mechanisms.
The old sequence, block-attention, and memory-dual runs neither promote nor
eliminate them because those probes changed several mechanisms and used a
different split and training contract. Unrestricted attention over every
instruction in a batched graph remains excluded: its quadratic cost ignores
the hierarchy already supported by E2.

The code also implements `forward_mean`, a generic-operator equivalence
reference with 239,696 parameters. It is tested against the canonical layer but
is not scheduled as a screen candidate because it asks no new question.

## First-class variable track

These models use the frozen `instruction -> defines -> variable` and
`variable -> operand -> instruction` edges. At each call depth, producer and
consumer instructions update recurrent variable states; the variables then
send independently gated feedback to consumers and producers before block
pooling. This is genuinely different from merely folding static variable
features into instruction inputs or using the derived instruction def-use edge.

| ID | Variable route | Local operator | E2 parameters |
|---|---|---|---:|
| `variable_route_all` | one exchange over every incident variable | canonical-equivalent forward mean | 322,640 |
| `variable_route_memory` | one exchange restricted by existing memory-like type slots | canonical-equivalent forward mean | 322,640 |
| `variable_route_all_2round` | two recurrent exchanges over all variables | canonical-equivalent forward mean | 405,584 |
| `variable_route_wide_operator` | one exchange over all variables | `wide_full_operator` | 621,584 |

Variable nodes have no explicit call-depth field. The implementation therefore
selects them only through edges incident to the current depth's instructions;
it does not invent ownership or mutate the tensor contract.

## Funnel

### Stage 0: engineering smoke test

Use 10% of training architecture groups, seed 42, and two epochs. This stage
checks finite forward/backward passes, complete gradients, memory, output
bundles, and resumability. It is never ranked.

### Stage 1: architecture screen

- Reuse the completed E6 H0 seed-7 and seed-42 50% runs under
  `artifacts/results/e6_architecture_scaling_v1`. Their prediction bundles
  include the complete E2 validation cohort. This avoids two redundant fits.
- Train candidates on those exact E6 family-stratified, whole-architecture 50%
  manifests. Candidate and canonical runs for a seed therefore use exactly the
  same subset.
- Keep all 2,297 E2 validation designs. Do not evaluate test or exemplar.
- Keep the E2 maximum epochs, patience, scheduler, loss, and target transforms.
  Reduced examples per epoch save compute; shortening the schedule would add a
  second training-budget change.
- Compare paired per-design SMAPE after averaging the six target errors. Use an
  architecture-cluster bootstrap for uncertainty and report resource, timing,
  target, family, runtime, memory, best epoch, and parameter count diagnostics.

Promotion is a robustness decision, not a significance test. Advance at most
five non-canonical candidates, with at least one slot reserved for a mechanism
model and one for a capacity/attention model when any model in that track meets
the rule. Ordinarily a candidate advances when both
seed/subset point estimates favour it and the mean improvement is at least 0.3
SMAPE. A candidate may also advance when one replicate improves by at least 1.0
and the other is no worse than +0.3. Reject numerical instability, a repeated
resource- or timing-scope regression above 1.0, or a runtime/memory increase
that is disproportionate to the effect. Freeze the shortlist and rules before
opening any E2 test predictions.

The 0.3 threshold is a practical smallest-interesting screen effect, not a
claim that smaller effects are zero. Borderline mechanisms can be retained for
scientific coverage, but must be labelled as such and count toward the limit of
five.

### Stage 2: full confirmation

- Rerun canonical and the frozen shortlist on 100% of E2 training data with
  seeds 7, 42, and 137 at the same code revision.
- Select checkpoints only on E2 validation SMAPE. Evaluate test once the full
  fit matrix is complete. Exemplar is a secondary stress diagnostic.
- Report each seed separately with architecture-cluster intervals. Treat the
  across-seed mean/range as descriptive; three seeds do not estimate a stable
  training-run population interval.
- Make one primary candidate-versus-canonical contrast. Treat ladder contrasts
  as separately trained interventions, not additive component attribution.

## Matching and capacity controls

The screen and primary confirmation are **training-contract matched** and
**representation-interface matched**. They are not wall-clock matched: early
stopping and operator cost can differ. They are not parameter matched: gates,
direction modulation, aggregate corrections, and GRU-style updates add active
capacity.
Every table must say this and include active parameters, time to best
validation, total fit time, and peak memory.

The reused E6 H0 screen controls were fitted at an earlier recorded revision
whose worktree was marked dirty. Subsequent committed changes did not alter the
canonical structured forward path, so reuse is appropriate for de-risking, but
the screen is explicitly a cross-revision comparison. Canonical is rerun at the
same clean revision as every finalist for full confirmation.

If the selected candidate has materially more parameters (more than 5% of the
canonical total), add two capacity diagnostics at full data:

1. a widened canonical model with the nearest attainable total parameter count;
2. a narrowed candidate with the nearest attainable canonical parameter count.

These are capacity sensitivity analyses, not replacements for the same-width
mechanism comparison. Padding either model with inactive parameters is not a
valid capacity match.

## Interpretation boundaries

- A screen result supports promotion only, never a test-set conclusion.
- A bundled candidate beating canonical does not identify its components;
  only the registered adjacent ladder contrasts do.
- Same-width comparisons isolate practical architecture choice, not parameter
  count.
- Validation reuse across many candidates creates selection pressure. The
  untouched test set is therefore the confirmatory endpoint, and the number of
  screened variants and all promotion decisions must be retained in metadata.
- Family/target slices are guardrails and heterogeneity descriptions unless a
  corresponding hypothesis was registered before test evaluation.

## Explicit future pool

These frozen-tensor-compatible ideas are intentionally not bundled into the
first screen. They are ordered roughly by expected information value.

1. Relation-specific sparse multi-head edge attention, separately localized to
   instruction relations and block CFG edges.
2. GPS-style parallel local message passing and containment-bounded attention,
   with a learned merge at instruction and block scales.
3. Virtual block and function tokens that recurrently exchange state with
   their contained nodes, without replacing the existing multi-statistic pool.
4. Call-site cross-attention over completed callee states, testing a richer
   replacement for the current projected mean injection.
5. Jumping-knowledge connections over instruction and block depths, followed
   by a deeper pre-norm residual stack.
6. Learned attention/set-transformer pooling as a separate pooling study.
7. Hierarchical Perceiver latents for a deliberately high-capacity surrogate.
8. Resource/timing-specific encoders or conditional mixture-of-experts. This
   changes representation routing or heads and should not be reported as a
   message-operator result.
9. Graphormer-style shortest-path or structural biases computed on the fly
    from existing edges. This requires no tensor regeneration but has enough
    engineering and compute cost to justify a later study.

## Expected compute

Twenty-three scheduled candidates at two 50%-data fits each cost roughly
twenty-three full-data-fit equivalents before early-stopping and architecture-cost
differences because the two canonical fits already exist. A canonical-plus-five
shortlist over three full-data seeds costs another eighteen full fits. The
attention and wide models will cost more per example, so GPU-hours and peak
memory—not fit counts alone—must accompany the accuracy frontier.

## Commands

The default screen schedules all twenty-three implemented candidates and reuses
E6 H0 automatically:

```bash
/home/brend/anaconda3/bin/conda run -n pipeline-env --no-capture-output \
  python scripts/run_e7_message_operators.py \
  --stage screen \
  --tensor-dir /home/brend/projects/data/tensors \
  --output-dir artifacts/results/e7_message_operator_screen_v1 \
  --devices 0 1 2 3 \
  --dry-run
```

Remove `--dry-run` only after checking the 46-job index. Analyze validation
predictions without touching test outputs:

```bash
/home/brend/anaconda3/bin/conda run -n pipeline-env --no-capture-output \
  python scripts/analyze_e7_screen.py \
  --results-dir artifacts/results/e7_message_operator_screen_v1
```
