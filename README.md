# reverse_engineering

A small research codebase for **reverse engineering / auditing training data influence** in text classification by:
1) selecting a **seed subset** of training data,
2) fine-tuning a base model into an **expert**,
3) creating **pseudo-expert models** along a base→expert path (via interpolation / merging),
4) computing a per-example **alignment matrix** using **last-layer gradients**.

The code is structured into modular components:

- `data.py` — dataset indexing + dataloader creation (seed subset and fine-tuning subset)
- `finetuning.py` — fine-tune base → expert and optionally checkpoint intermediate models
- `models.py` — model interpolation / merge utilities (includes mergekit config helpers)
- `alignment.py` — compute alignment matrix `M` from last-layer gradients
- `domain_distribution.py` — experiment entrypoint that wires everything together using a JSON config

---

## Quickstart

### Install dependencies

Use either pip or conda:

```bash
pip install -r requirements.txt
```

or

```bash
conda env create -f enviroment.yml
conda activate <env-name>
```

This project expects common ML libs like:
- `torch`
- `transformers`
- `datasets`
- `numpy`
- `tqdm`
- `pyyaml`
- (optional) `mergekit` (see below)

### Run an experiment

The modular entrypoint expects a JSON config:

```bash
python domain_distribution.py path/to/config.json
```

Outputs are written to:

- `output_dir = config.experiment_name`

---

## How the modular pipeline works

### 1) Configuration (`domain_distribution.py`)

`domain_distribution.py`:
- loads a JSON config file from `sys.argv[1]`
- constructs a `Config` object and derives:
  - `lambdas = np.linspace(lambda_min, lambda_max, K)`
  - `device = cuda if available else cpu`
- creates the output directory `config.experiment_name`
- loads the dataset: `datasets.load_dataset(config.dataset)` and uses `dataset["train"]`

### 2) Dataset subsets & dataloaders (`data.py`)

`data.py` defines a `Dataset` helper class that:
- computes `valid_indices` by filtering out samples with `label < 0` (useful for datasets like SNLI)
- selects a seed subset **D** of size `n_seed` (`n_total` in config)
- selects a fine-tuning subset **D'** of size `n_finetune` with a desired class distribution `proportionArr`
- provides dataloaders used for:
  - fine-tuning
  - per-example gradient computations on the seed subset

> Notes:
> - `proportionArr` is checked to sum to 1.
> - The code supports choosing the fine-tuning source via `finetuning_source` (e.g., "select" vs "original" depending on your intended experiment design).

### 3) Fine-tuning (`finetuning.py`)

`finetuning.py` defines `Finetuning`, which:
- trains a `BertForSequenceClassification` on the fine-tuning loader
- periodically checkpoints intermediate models (`model_<k>.pt`) based on a computed `batch_interval`
- can evaluate accuracy by sampling from a provided evaluation set (superset)

Typical artifacts you may save:
- `accuracy_arr.pkl` (if you store it from the caller)
- intermediate checkpoints `model_0.pt ... model_{K-1}.pt` (if enabled/used)

### 4) Pseudo-experts / interpolation / merges (`models.py`)

`models.py` defines a `Model` class that contains helpers for creating pseudo-expert models.
It includes YAML config generation for merge methods like:
- **SLERP**
- **TIES**
(and likely other mergekit-based approaches in the remainder of the file)

It also attempts to import mergekit by adding a local `mergekit` directory to `sys.path`.

> If mergekit is not available, the code prints a warning and merge-based interpolation paths may not work.

### 5) Alignment matrix computation (`alignment.py`)

`alignment.py` defines `Alignment`, which computes an alignment matrix `M` of shape:

- `M.shape == (n_seed, K)`

The key idea:
- for each pseudo-expert (indexed by `k`, corresponding to `lambda_k`),
- compute per-example gradients for **only the classifier / last layer** (parameters with `'classifier' in name`),
- compare those gradients to a “direction” vector derived from last-layer parameter differences between models,
- store per-example scores into the matrix.

This matrix can then be used for downstream analysis (e.g., identifying which examples are most aligned with certain interpolation points).

---

## Outputs

The entrypoint currently saves (or prepares to save) artifacts like:

- `dataset_info.json`
  - `indices_D` (seed subset)
  - `indices_D_prime` (fine-tune subset)
  - `n_total`, `n_finetune`

Other typical outputs (depending on what’s enabled in your run / scripts):
- intermediate model checkpoints: `model_<k>.pt`
- base/expert model directories (HF format) if you save them in your run
- alignment matrices: `alignment_matrix_<method>.npy` (if written by the caller)

---

## Example config (template)

Here’s a minimal template showing the kinds of fields used by `domain_distribution.py`:

```json
{
  "experiment_name": "my_experiment",
  "dataset": "ag_news",
  "model_name": "bert-base-uncased",
  "num_labels": 4,

  "batch_size": 16,
  "max_length": 256,
  "learning_rate": 2e-5,
  "num_epochs": 4,
  "optimizer": "Adam",

  "n_total": 5000,
  "n_finetune": 2500,
  "finetuning_source": "original",
  "proportionArr": [0.25, 0.25, 0.25, 0.25],

  "K": 15,
  "lambda_min": 0.05,
  "lambda_max": 0.95
}
```

---

## Notes / Caveats

- Some modules appear to be mid-refactor (e.g., `Alignment.generate_alignment_matrix()` references `self.get_interpolated_model(...)` which must exist either later in the file or via composition with `Model`).
- Mergekit integration depends on a local `mergekit/` folder being present and importable.
- If you intend `domain_distribution.py` to be the “main” entrypoint, consider renaming it to something more descriptive like `run_experiment.py`.

---

## License

No license file is currently included. Add one if you plan to share/distribute this repository.