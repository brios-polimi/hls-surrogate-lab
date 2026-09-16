# E2 hierarchy ablation

## Registered questions

The primary contrast is `hierarchy_orderless - h0`, reported separately for
training seeds 7, 42, and 137. It tests whether structured block/function/call
composition improves exact-architecture transfer beyond intact instruction
control/def-use propagation, the same raw node and pragma inputs, and the same
239,696 active parameters.

Two secondary, pre-specified contrasts localize the primary effect:

- `no_block_cfg - h0` removes only block-CFG messages.
- `no_callee - h0` removes only callee-state injection at call sites.

Positive SMAPE differences favor LLVM-Hier. Architecture-cluster bootstrap
intervals condition on each fitted pair. The mean, standard deviation, and
range across the three training seeds are descriptive and are not a confidence
interval over training randomness.

The orderless control retains graph-global hierarchy counts because it retains
the registered global-feature branch. Its estimand is therefore the effect of
learned structured composition, not the effect of removing every
hierarchy-derived scalar input.

The thesis claim is conditional on the observed contrasts. If the primary
delta is positive with a cluster interval above zero for all three seeds, the
defensible claim is that structured block/function/call composition improves
exact-architecture transfer beyond instruction-local propagation and an
active-capacity-matched orderless encoder. The two secondary contrasts can
localize evidence to block-CFG propagation or callee injection, but their
effects are not assumed to be additive. If the primary intervals include zero,
the existing local-adjacency result remains valid but the thesis should not
claim an independently demonstrated hierarchy benefit.

## Run

Use the frozen E2 manifest and tensor vocabulary. The dry run must list exactly
nine jobs before training is started.

```bash
cd /home/brend/projects/hls-surrogate-lab

/home/brend/anaconda3/bin/conda run -n pipeline-env --no-capture-output \
  python scripts/run_e2.py \
  --tensor-dir /home/brend/projects/data/tensors \
  --vocab /home/brend/projects/data/tensors/vocab.json \
  --manifest /home/brend/projects/hls-surrogate-lab/artifacts/releases/vitis-a31-coarsearch1-hierarchy2-vocab-2026-09-04/e2_structural_v1/architecture_grouped_structural_v1.json \
  --output-dir /home/brend/projects/hls-surrogate-lab/artifacts/results/e2_hierarchy_ablation_v1 \
  --models hierarchy_orderless no_block_cfg no_callee \
  --seeds 7 42 137 \
  --devices 0 1 2 3 \
  --dry-run
```

Remove `--dry-run` to train. Use `--resume` only when the runner reports a
partial neural run with a backup checkpoint.

The closest completed runs took 9.27--11.69 GPU-hours for LLVM-Hier and
7.43--9.02 GPU-hours for the no-local control. Nine fits therefore budget about
80--90 GPU-hours. Dynamic scheduling should put this near 20--25 wall-clock
hours on four comparable GPUs or 27--31 hours on three, subject to early
stopping and hardware differences.

## Analyze

The analysis reuses the previously produced, same-seed LLVM-Hier predictions.
It validates the manifest membership, labels, model identities, training seed,
split hash, and sibling resolved configurations before computing contrasts.

```bash
/home/brend/anaconda3/bin/conda run -n pipeline-env --no-capture-output \
  python scripts/analyze_e2_hierarchy.py \
  --manifest /home/brend/projects/hls-surrogate-lab/artifacts/releases/vitis-a31-coarsearch1-hierarchy2-vocab-2026-09-04/e2_structural_v1/architecture_grouped_structural_v1.json \
  --results-dir /home/brend/projects/hls-surrogate-lab/artifacts/results/e2_hierarchy_ablation_v1 \
  --output-dir /home/brend/projects/hls-surrogate-lab/artifacts/results/e2_hierarchy_ablation_v1/analysis \
  --h0-prediction 7=/home/brend/projects/hls-surrogate-lab/artifacts/results/e2_replication_v1/seed7/e2_h0_structural_seed7/predictions.csv \
  --h0-prediction 42=/home/brend/projects/hls-surrogate-lab/artifacts/results/e2_structural_v1/e2_structural_v1/results/seed42/e2_h0_structural_seed42/predictions.csv \
  --h0-prediction 137=/home/brend/projects/hls-surrogate-lab/artifacts/results/e2_replication_v1/seed137/e2_h0_structural_seed137/predictions.csv
```

The durable headline table is `analysis/seed_specific_hierarchy_contrasts.csv`.
`analysis/cross_seed_descriptive.csv` contains only descriptive aggregation.
Each seed directory contains family- and target-scope diagnostics plus hashes
of the exact prediction and configuration files used. The suite analysis also
refuses to compare a run when its recorded parameter count differs from H0.

After results exist, the implementation chapter needs the registered orderless,
no-block-CFG, and no-callee definitions; the methodology chapter needs the
pre-specified primary/secondary roles and conditional seed-wise inference; and
the results chapter needs the three seed-specific contrasts plus the explicitly
descriptive cross-seed summary. No thesis text should be changed before those
outputs are available.
