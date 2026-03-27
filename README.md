# Adaptive Recourse Routing Artifact

This repository is the public artifact for the manuscript **Adaptive recourse routing under realistic actionability constraints**.

## Contents

- `scripts/`
  - experiment runners for adaptive routing, DiCE baselines, and tree-MILP baselines
  - analysis and plotting code used for the reported summary tables and figures
- `exp/`
  - processed experiment outputs used in the manuscript
  - merged multi-seed summaries for the guarded router, DiCE family, exact tree-MILP baseline, and scaling studies
- `figures/`
  - exported figures used in the manuscript, including the external quality-speed frontier

## Public artifact scope

This repository intentionally includes only:

- code required to run the public experiments
- derived result tables and summary CSV files
- generated figures needed to reproduce the reported claims

This repository intentionally excludes:

- manuscript source and build files
- local notes and draft reports
- caches and temporary files
- legacy reference project materials not part of the new artifact

## Main experiment groups

- `exp/adaptive_routing_guarded_multi3_merged/`
  - guarded adaptive routing main results over Adult, German Credit, and Bank Marketing
- `exp/dice_baseline_multi3_q20/`
  - DiCE random matched-subset baseline
- `exp/dice_genetic_multi3_q20/`
  - DiCE genetic matched-subset baseline
- `exp/tree_milp_multi3_q20_v2/`
  - exact tree-constrained mixed-integer baseline
- `exp/adaptive_routing_scaling_depth/`
  - runtime scaling with tree depth
- `exp/adaptive_routing_scaling_rows_merged/`
  - row-budget scaling summaries

## Reproduction

The experiments were run with Python and the following main dependencies:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `dice-ml`
- `pulp`

Install the minimal dependencies with:

```bash
pip install -r requirements.txt
```

Representative commands:

```bash
python scripts/run_adaptive_routing_experiments.py --datasets adult german bank --seeds 7 13 19 --depths 3 --max-rows 600 --exp-name adaptive_routing_guarded_multi3
```

```bash
python scripts/run_dice_baselines.py --datasets adult german bank --seeds 7 13 19 --depths 3 --max-rows 600 --max-queries-per-cell 20 --exp-name dice_baseline_multi3_q20 --dice-method random
```

```bash
python scripts/run_tree_milp_baseline.py --datasets adult german bank --seeds 7 13 19 --depths 3 --max-rows 600 --max-queries-per-cell 20 --exp-name tree_milp_multi3_q20_v2
```

## Data

The manuscript uses standard public tabular benchmark datasets: Adult, German Credit, and Bank Marketing. This artifact releases processed experiment outputs and derived figures rather than redistributing raw benchmark data.
