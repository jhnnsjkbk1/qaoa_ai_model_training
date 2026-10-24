# QAOA AI model training

> [!NOTE]
> This repository is associated with a [research publication](https://arxiv.org/abs/2610.23296) and the code here is not actively maintained. It is not officially supported IBM Quantum software.


This repository contains the code to train and evaluate machine learning models that predict QAOA parameters for weighted MaxCut problems. Given a problem graph, a model outputs the angles

```text
[beta_1, ..., beta_p, gamma_1, ..., gamma_p]
```

for QAOA depth `p`.

Depending on the architecture, a model uses graph-level features, the graph structure, node features and edge weights. All available models are listed in `MODEL_REGISTRY` in [`ml_models/models.py`](ml_models/models.py), and each model class documents which inputs it uses.

## Installation

Requires Python 3.11+. Clone the repository and install the package into a fresh `venv`:

```bash
git clone git@github.com:jhnnsjkbk1/qaoa_ai_model_training.git
cd qaoa_ai_model_training
pip install -e .
```

This covers the `mlp`, `edge_transformer`, `diffusion_transformer`, and `graph_neural_network` models.

The `gcn`, `graph_isomorphism_network`, and `graph_transformer` models need `torch-geometric` as well. Install it after the base package when `torch` is already present:

```bash
pip install -e ".[graph]" --no-build-isolation
```

`--no-build-isolation` ensures `torch-scatter`, `torch-sparse` and `torch-cluster` are built against the environment you installed `torch` into.

The tutorial notebook needs Jupyter, which is included in the `notebooks` extra:

```bash
pip install -e ".[notebooks]"
```

### Julia (for QAOA depth `p > 1`)

Runs with `p = 1` do not need Julia. For `p > 1`, the pipeline uses `PPEvaluator` from [`qaoa_training_pipeline`](https://github.com/qiskit-community/qaoa_training_pipeline), which relies on [PauliPropagation.jl](https://github.com/MSRudolph/PauliPropagation.jl) through `juliacall`. Install the `julia` extra, then run the setup script:

```bash
pip install -e ".[julia]"
python link-julia-python.py
```

If Julia is not installed, `juliacall` downloads a compatible version into the virtual environment on first use. An existing Julia installation, for example from [juliaup](https://github.com/JuliaLang/juliaup), is used otherwise. The setup script installs `PauliPropagation.jl` into the Julia environment and checks that the Python–Julia bridge works.

## Quickstart

To check that the installation works, you can run the full pipeline on a tiny synthetic dataset. The labels are random, non-optimized angles, so the resulting model is only useful as a smoke test. Each step is explained in more detail in [Data](#data), [Training](#training) and [Testing](#testing).

1. Create the dataset. This writes 12 small random graphs, their max-cut values and made-up labels to `/tmp/qaoa_smoke_test`:
   ```bash
   python -m scripts_and_notebooks.make_smoke_test_data --out_dir /tmp/qaoa_smoke_test
   ```
2. Split the graphs into train, validation and test sets (see [Train, validation and test splits](#4-train-validation-and-test-splits)). The first command splits all graphs and computes normalization statistics. The second keeps only the graphs that have labels at depth `p = 1`:
   ```bash
   python -m scripts_and_notebooks.testset_generation --graphs_repo /tmp/qaoa_smoke_test --instance_sets smoke_test
   python -m scripts_and_notebooks.build_per_p_split --graphs_repo /tmp/qaoa_smoke_test --target_p 1
   ```
3. Train a model with the settings in [`smoke_test_config.yaml`](configs/sample_configs/smoke_test_config.yaml):
   ```bash
   python -m training.training_script --config configs/sample_configs/smoke_test_config.yaml
   ```
4. Test the trained checkpoint. The results are written to `runs/smoke_test/testing/<timestamp>/`:
   ```bash
   python -m training.testing_script --config configs/sample_configs/smoke_test_config.yaml \
     --ckpt_path checkpoints/smoke_test/smoke_test/last.ckpt
   ```

To work with your own data, follow [Data](#data), [Configuration](#configuration), [Training](#training) and [Testing](#testing) in order.

## Data

**This repository contains code only.** The graph dataset used in the accompanying paper is not included. To train or evaluate a model, you need to provide your own graph data and QAOA optimization results.

The pipeline expects the directory specified by `data.graph_dir` to follow this structure (individual paths can also be overridden in the configuration):

```text
<graph_dir>/
  instances/            # graph instances used as model inputs
  max_cuts/             # max/min cut values, needed for approximation-ratio metrics
  fixed_angle_cuts/     # optional fixed-angle baseline results, under p_<P>/<instance_set>/
  results_p<P>/         # best_param.json with the label angles for depth P
  splits/               # train, validation and test splits
```

The required data can be generated with [`qaoa_training_pipeline`](https://github.com/qiskit-community/qaoa_training_pipeline), which is installed with this package. The steps below show how.

### 1. Graph instances

Store graph instances under

```text
<graph_dir>/instances/<instance_set>/
```

For example:

```text
<graph_dir>/instances/random_regular/
<graph_dir>/instances/erdos_renyi/
```

Each graph is stored as a JSON edge list with per-edge weights:

```json
{
  "edge list": [
    {"nodes": [0, 1], "weight": 1},
    {"nodes": [1, 2], "weight": 1}
  ]
}
```

For example, a random 3-regular graph can be written with

```python
import json
import networkx as nx
from qaoa_training_pipeline.utils.graph_utils import graph_to_dict

graph = nx.random_regular_graph(3, 10, seed=0)
with open("<graph_dir>/instances/random_regular/rr3_n10_s0.json", "w") as f:
    json.dump(graph_to_dict(graph), f)
```

Every instance ID must be unique across all instance sets that are combined in one dataset.

### 2. Max-cut values

Maximum and minimum cut values are needed to compute approximation ratios and `reach` during testing, and during validation for `p = 1`. Training itself does not need them, since the loss is computed on the angles, but without them these metrics are `NaN` and a warning reports how many files are missing.

Store one file per instance, named like the instance file, under

```text
<graph_dir>/max_cuts/<instance_set>/<instance_id>.json
```

Each file needs the keys `max_cut` and `min_cut`:

```json
{"max_cut": 16.0, "min_cut": 0.0}
```

They can be computed with `solve_max_cut` from `qaoa_training_pipeline`, which requires CPLEX (`pip install cplex`). For example:

```python
import json
from pathlib import Path
from qaoa_training_pipeline.utils.graph_utils import graph_to_operator, load_graph, solve_max_cut

out_dir = Path("<graph_dir>/max_cuts/random_regular")
out_dir.mkdir(parents=True, exist_ok=True)
for f in Path("<graph_dir>/instances/random_regular").glob("*.json"):
    max_cut, min_cut, _ = solve_max_cut(graph_to_operator(load_graph(f), pre_factor=-0.5))
    (out_dir / f.name).write_text(json.dumps({"max_cut": max_cut, "min_cut": min_cut}))
```

### 3. QAOA optimization results

The supervised labels are optimized QAOA parameters for each graph instance and QAOA depth.

You can use any QAOA parameter optimization strategy to produce them. Raw optimization results are kept under `<graph_dir>/results_p<P>/<instance_set>/`, but the pipeline does not rely on them. They are only used as input for building `best_param.json` (see below), so keeping them is optional.

The only file training and testing read from `results_p<P>/` is `best_param.json`, which holds the best angle and energy pair for every instance:

```text
{
  <instance_set>: {
    <instance_id>: {
      <p>: {
        "energy": ...,
        "qaoa_angles": ...,
        "result_file_name": ...
      }
    }
  }
}
```

`<p>` is the depth as a string, for example `"3"`, and `qaoa_angles` is ordered `[beta_1, ..., beta_p, gamma_1, ..., gamma_p]`.

#### Generating results with qaoa_training_pipeline

[`configs/qaoa_methods/fixed_start_recursion_SV.json`](configs/qaoa_methods/fixed_start_recursion_SV.json) optimizes the angles with an exact statevector simulation. It starts from the 3-regular `p = 1` fixed angles, extends them by recursion to depth `p` and refines them. Run it for every instance and depth:

```bash
for p in 1 3; do
  for f in <graph_dir>/instances/random_regular/*.json; do
    python -m qaoa_training_pipeline.train \
      --input "$f" \
      --config configs/qaoa_methods/fixed_start_recursion_SV.json \
      --pre_factor -0.5 \
      --component_kwargs0 reps:$p \
      --save --save_dir <graph_dir>/results_p$p/random_regular/ \
      --save_file "$(basename "$f")"
  done
done
```

`--pre_factor -0.5` defines the MaxCut cost operator and `--component_kwargs0 reps:<p>` sets the depth. Statevector simulation scales exponentially with the number of nodes, so use another evaluator of `qaoa_training_pipeline` for larger graphs.

Then collect the best angles per instance and depth with

```bash
python -m scripts_and_notebooks.build_best_param \
  --results <graph_dir>/results_p1 <graph_dir>/results_p3 \
  --graphs_repo <graph_dir>
```

This reads all result files under the given folders, keeps the angles with the highest energy over all pipeline stages for each instance and depth, and writes `results_p<P>/best_param.json` for every depth found. The instance set of each instance is looked up under `<graph_dir>/instances/`, and instances without an instance file are skipped.

### 4. Train, validation and test splits

Once the graph instances are available, generate the master split with

```bash
python -m scripts_and_notebooks.testset_generation \
  --graphs_repo <graph_dir> \
  --instance_sets random_regular erdos_renyi
```

This also generates the normalization statistics used by the models. By default, 15% of the instances are held out for testing (`--test_fraction`) and 90% of the rest are used for training (`--train_fraction`).

For a specific QAOA depth, filter the split to instances that have optimization results at that depth:

```bash
python -m scripts_and_notebooks.build_per_p_split \
  --graphs_repo <graph_dir> \
  --target_p 3
```

This creates

```text
splits/p_<P>/train_val_split.json
splits/p_<P>/test_split.json
```

Training and testing pick up these files and the normalization statistics stored in `train_val_split.json` automatically from `data.graph_dir` and `data.p`. To use split files from another location, set `data.split_file` and `data.test_file` in the configuration.

### 5. Fixed-angle baseline (optional)

If no fixed-angle result is available for an instance, training and testing still work and `fixed_angle_approx_ratio` is left empty.

To reproduce our fixed-angle baseline:

1. Download `angles_regular_graphs.json` from [danlkv/fixed-angle-QAOA at commit `956be14`](https://github.com/danlkv/fixed-angle-QAOA/blob/956be1450c9d1f1957d3189a6603264d13163088/angles_regular_graphs.json):
   ```bash
   curl -LO https://raw.githubusercontent.com/danlkv/fixed-angle-QAOA/956be1450c9d1f1957d3189a6603264d13163088/angles_regular_graphs.json
   ```
2. For each instance set, run:
   ```bash
   python -m scripts_and_notebooks.compute_fixed_angle_cuts \
     --angles_file angles_regular_graphs.json \
     --instance_dir <graph_dir>/instances/<instance_set> \
     --instance_set <instance_set> \
     --depth <P>
   ```
   This looks up the published angles for each instance's degree, evaluates the cut value, and writes one result file per instance under `fixed_angle_cuts/p_<P>/<instance_set>/`.

You can use the same output format to plug in fixed angles from any other source instead.

A fixed-angle result has the following format:

```json
{
  "fixed_angle_cut": 9.886767290492752,
  "betas": [0.3926720292447629],
  "gammas": [0.615533629093832]
}
```

`fixed_angle_cut` is the cut value at the given angles. `betas` and `gammas` must each have length `p`.

## Configuration

We provide a sample config, [`configs/sample_configs/sample_config.yaml`](configs/sample_configs/sample_config.yaml). You can use it to get started and build your own configs on top of it.

The main settings to change are:

- **`data.graph_dir`**: root directory containing the graph data
- **`data.p`**: QAOA depth. Start with `p = 1` if you want to run the pipeline without Julia
- **`data.instance_set`**: instance sets used for training
- **`data.norm_stats`**: optional feature normalization statistics. If omitted, the statistics stored in the train/val split file are used
- **`model.model_type`**: model architecture from `MODEL_REGISTRY`

Multiple instance sets can be listed to train one model across different graph families.

[`scripts_and_notebooks/datamodule_tutorial.ipynb`](scripts_and_notebooks/datamodule_tutorial.ipynb) provides a walkthrough of the datamodule, normalization statistics and data splits.

## Training

Run training with:

```bash
python -m training.training_script \
  --config configs/sample_configs/sample_config.yaml
```

Checkpoints are written to

```text
<ckpt_dir>/<run_name>/<model_type>-p<p>-<YYYYMMDD_HHMMSS>-<epoch>-<metric>.ckpt
```

TensorBoard logs are written to

```text
<log_dir>/<run_name>/
```

Both locations are configured in the `trainer` section of the configuration file.

## Testing

Testing is run separately from training because QAOA evaluation can be expensive and may be performed on different hardware.

Test a checkpoint with:

```bash
python -m training.testing_script \
  --config configs/sample_configs/sample_config.yaml \
  --ckpt_path checkpoints/p_1/qaoa_graph_neural_network_p1/last.ckpt
```

To evaluate several checkpoints, run `training.testing_script` once for each configuration and checkpoint pair.

Checkpoints are loaded with `strict=True`, so incompatible model architectures fail during loading instead of silently loading a partially initialized model.

Each run writes

```text
test_metrics.csv
test_per_instance.csv
reach_distribution.png
```

to a timestamped folder `<log_dir>/<run_name>/testing/<YYYYMMDD_HHMMSS>/`, so repeated runs do not overwrite each other.

### Analyzing results

`test_per_instance.csv` contains one row for each evaluated graph and is the recommended basis for further analysis.

| Column | Meaning |
|---|---|
| `instance_id`, `instance_set` | Graph instance and source instance set |
| `graph_type` | Same value as `instance_set`, e.g. `random_regular` |
| `node_count`, `num_edges`, `depth_p` | Graph size and QAOA depth |
| `reach` | Model approximation ratio divided by the optimal-angle approximation ratio |
| `approx_ratio` | Approximation ratio obtained from the predicted angles |
| `opt_approx_ratio` | Approximation ratio obtained from the reference angles |
| `energy` | QAOA energy obtained from the predicted angles |
| `fixed_angle_approx_ratio` | Fixed-angle baseline when available |
| `beta_<j>`, `gamma_<j>` | Predicted QAOA angles for layer `j = 1, ..., p` |

A `reach` of `1.0` means that the predicted angles match the approximation ratio obtained from the reference optimized angles.

For example:

```python
import pandas as pd

df = pd.read_csv("runs/<run_name>/testing/<YYYYMMDD_HHMMSS>/test_per_instance.csv")

df.groupby("instance_set")["reach"].agg(["mean", "std", "min"])
df.groupby("node_count")["reach"].mean()
(df["reach"] > 0.99).mean()
df.nsmallest(10, "reach")
```

The models do not receive `instance_set` or `graph_type` as input. These fields are only stored for analysis.

## Repository layout

| Path | Contents |
|---|---|
| `datamodule/` | Dataset loading, feature computation, normalization and splits |
| `ml_models/` | Model architectures and `MODEL_REGISTRY` |
| `training/` | Training, testing and QAOA evaluation |
| `configs/sample_configs/` | Sample and smoke-test configurations |
| `configs/qaoa_methods/` | `qaoa_training_pipeline` config for generating labels |
| `scripts_and_notebooks/` | Data preparation utilities and tutorial |
| `tests/` | Model, configuration and repository tests |

See [`training/README.md`](training/README.md) for how the modules in `training/` fit together.

The main training components are:

- `training/training_script.py` runs model training
- `training/testing_script.py` evaluates a trained checkpoint
- `training/eval_step.py` contains the shared validation and test evaluation logic
- `training/qaoa_model.py` defines the Lightning model, training step, MSE loss and Adam optimizer

The data preparation utilities are:

| File | Purpose |
|---|---|
| `datamodule_tutorial.ipynb` | Walkthrough of the datamodule, normalization and splits |
| `build_best_param.py` | Build `results_p<P>/best_param.json` from `qaoa_training_pipeline` results |
| `testset_generation.py` | Generate the master train, validation and test split |
| `build_per_p_split.py` | Filter a split to instances available at a given QAOA depth |
| `compute_fixed_angle_cuts.py` | Compute the fixed-angle baseline |
| `make_smoke_test_data.py` | Generate a tiny synthetic dataset for smoke tests |

## Citation

If you use this code, please cite:

```bibtex
@article{jakubik2026learning,
  title   = {Learning angles for quantum approximate optimization},
  author  = {Jakubik, Johannes and Wittmann, Isabelle and Gaffoor, Zaheed and Baloyi, Gciniwe Simphiwe and Mahlasi, Craig and Vos, Etienne and Brunschwiler, Thomas and Egger, Daniel J.},
  journal = {arXiv preprint arXiv:2610.23296},
  year    = {2026},
  doi     = {10.48550/arXiv.2610.23296}
}
```

## Acknowledgements

This project builds on [`qaoa_training_pipeline`](https://github.com/qiskit-community/qaoa_training_pipeline) for QAOA evaluation and training utilities and on [PauliPropagation.jl](https://github.com/MSRudolph/PauliPropagation.jl) for efficient evaluation at QAOA depth `p > 1`.

## License

See [LICENSE](LICENSE).
