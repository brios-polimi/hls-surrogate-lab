# E7 message-operator architecture funnel

## Decision and estimand

E7 asks whether a better **local message/update operator** improves
exact-architecture transfer when the proven hierarchy is held fixed. The frozen
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
| `relation_mixer` | Concatenate relation messages and learn their update mixture | Does delayed learned relation mixing help without gates? | secondary |
| `receiver_gate` | Receiver-conditioned gate for each relation/flow | `receiver_gate - canonical`: gating main effect | high |
| `bidirectional` | Separate learned forward and reverse flows | `bidirectional - canonical`: direction main effect | high |
| `gate_bidir` | Receiver gates plus forward/reverse flows | Factorial interaction relative to the two main effects | high |
| `gate_bidir_meanmax` | Add relation/flow-specific mean and max aggregation | Aggregator effect conditional on `gate_bidir` | high |
| `gate_bidir_gru` | Replace residual update with a GRU update | Update effect conditional on `gate_bidir` | secondary |
| `full_operator` | Gate, bidirectionality, mean/max, and GRU update | GRU effect conditional on `gate_bidir_meanmax` | secondary |

This ladder distinguishes the proposed ingredients rather than testing only an
inseparable bundle. Reverse flow is tested at both instruction and block scale;
if it wins, a later localization pair (instruction-only versus block-only)
should be run before making a scale-specific claim. The same rule applies to
gating.

Sparse receiver-conditioned edge attention is the first expressive extension
after this ladder. Within-block instruction attention and within-function block
attention are separate hierarchy/composition experiments, not message-operator
variants. They should enter only after E7 identifies a strong local operator,
using that operator as their control. Full-graph attention is low priority:
its quadratic cost, weak structural prior, and confounding of hierarchy make it
a poor first response to the current evidence. The old sequence, block-attention,
and memory-dual runs neither promote nor eliminate these ideas because they
changed several mechanisms and used a different split and training contract.

## Funnel

### Stage 0: engineering smoke test

Use 10% of training architecture groups, seed 42, and two epochs. This stage
checks finite forward/backward passes, complete gradients, memory, output
bundles, and resumability. It is never ranked.

### Stage 1: mechanism screen

- Use family-stratified, whole-architecture 50% subsets of the frozen E2 train
  split for seeds 7 and 42. Candidate and canonical runs for a seed use exactly
  the same subset.
- Keep all 2,297 E2 validation designs. Do not evaluate test or exemplar.
- Keep the E2 maximum epochs, patience, scheduler, loss, and target transforms.
  Reduced examples per epoch save compute; shortening the schedule would add a
  second training-budget change.
- Compare paired per-design SMAPE after averaging the six target errors. Use an
  architecture-cluster bootstrap for uncertainty and report resource, timing,
  target, family, runtime, memory, best epoch, and parameter count diagnostics.

Promotion is a robustness decision, not a significance test. Advance at most
three non-canonical candidates. Ordinarily a candidate advances when both
seed/subset point estimates favour it and the mean improvement is at least 0.3
SMAPE. A candidate may also advance when one replicate improves by at least 1.0
and the other is no worse than +0.3. Reject numerical instability, a repeated
resource- or timing-scope regression above 1.0, or a runtime/memory increase
that is disproportionate to the effect. Freeze the shortlist and rules before
opening any E2 test predictions.

The 0.3 threshold is a practical smallest-interesting screen effect, not a
claim that smaller effects are zero. Borderline mechanisms can be retained for
scientific coverage, but must be labelled as such and count toward the limit of
three.

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
reverse projections, aggregate compressors, and GRU cells add active capacity.
Every table must say this and include active parameters, time to best
validation, total fit time, and peak memory.

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

## Expected compute

Eight candidates (including canonical), two 50%-data fits each, cost roughly
eight full-data-fit equivalents before early-stopping differences. A
canonical-plus-three shortlist over three full-data seeds costs another twelve
full fits. This is substantially more informative than dozens of 25%-data,
single-seed rankings and substantially cheaper than immediately giving every
idea a three-seed E2 campaign.
