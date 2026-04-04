# hack_model (reverse engineering training-data influence)

Research codebase for **sample-level hacking / reverse engineering** of training-data influence in text(-classification) fine-tuning trajectories.

At a high level, the pipeline:
1. Selects a **seed subset** \(D\) of training data.
2. Builds a **fine-tuning subset** \(D'\) with a controlled class distribution.
3. Fine-tunes a base model into an **expert** (saving intermediate checkpoints along the way).
4. Constructs **pseudo-expert models** along the base → expert path (via interpolation / merge).
5. Computes a per-example **alignment matrix** \(M\) using **last-layer gradients**.

This repository currently contains two experiment implementations:
- `bert/` — BERT sequence-classification experiments
- `gpt2/` — GPT-2 sequence-classification experiments

---

## Repository structure

Top-level utilities:

- `data.py` — dataset indexing + dataloader creation.
  - filters invalid labels (e.g. SNLI’s `label == -1`)
  - selects the seed subset \(D\)
  - constructs the fine-tuning subset \(D'\) with a target class distribution (`proportion_arr`)
  - provides PyTorch `DataLoader`s used for fine-tuning + per-example gradient computation

Notebooks:
- `experiment.ipynb` — exploratory / interactive experiment driver
- `Baselines.ipynb` — baseline comparisons
- `Visualizations.ipynb` — plotting / visualization utilities
- `edit_plots.ipynb` — plot editing / polishing

Model-specific pipelines:

### `bert/`
- `bert/bert_domain_distribution.py` — **main runner** for the BERT pipeline.
  - loads a JSON config
  - prepares \(D\) and \(D'\) using `data.py`
  - fine-tunes to create the expert model (and intermediate checkpoints)
  - computes alignment matrices for each interpolation method in `config.interpolations`

- `bert/bert_finetuning.py` — fine-tunes BERT base → expert and optionally saves intermediate checkpoints `model_<k>.pt`.

- `bert/bert_models.py` — pseudo-expert creation utilities.
  - linear + quadratic interpolation
  - mergekit-based merges (if available): **SLERP**, **TIES**, **DELLA**

- `bert/bert_alignment.py` — computes the alignment matrix \(M\) using **per-example last-layer gradients**.

### `gpt2/`
- `gpt2/gpt2_domain_distribution.py` — **main runner** for the GPT-2 pipeline.
  - same overall steps as the BERT runner, but using GPT-2 classification models

- `gpt2/gpt2_finetuning.py` — fine-tunes GPT-2 base → expert and saves intermediate checkpoints.

- `gpt2/gpt2_models.py` — pseudo-expert creation utilities for GPT-2.
  - linear + quadratic interpolation
  - mergekit-based merges (if available): **SLERP**, **TIES**, **DELLA**

- `gpt2/gpt2_alignment.py` — alignment matrix computation for GPT-2 using **last-layer (`score`) gradients**.

Legacy:
- `old_code/` — older scripts kept for reference (AG News class/sample hacking variants).

---

## Quickstart

### Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using conda:

```bash
conda env create -f enviroment.yml
conda activate <env-name>
```

> Note: advanced interpolation methods (SLERP/TIES/DELLA) rely on `mergekit`. The repo’s `requirements.txt` includes a mergekit editable install; if mergekit import fails, those methods will be unavailable.

---

## Running an experiment

Both pipelines are driven by a JSON config.

### BERT

```bash
python bert/bert_domain_distribution.py path/to/config.json
```

### GPT-2

```bash
python gpt2/gpt2_domain_distribution.py path/to/config.json
```

Outputs are written to:
- `output_dir = config.experiment_name`

Typical artifacts:
- `{experiment_name}/dataset_info.json` — indices for \(D\) and \(D'\)
- `{experiment_name}/theta_base_model.pt` — base weights
- `{experiment_name}/theta_exp_model.pt` — expert weights
- `{dataset}_{interpolation}_{proportionArr}/alignment_matrix_<interpolation>.npy` — alignment matrices
- `{dataset}_{interpolation}_{proportionArr}/lambda_statistics.json` — per-λ summary stats

---

## Example config (template)

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
  "lambda_max": 0.95,

  "interpolations": ["linear", "quadratic", "slerp", "ties", "della"]
}
```

---

## Notes / caveats

- The code is research-grade and some parts may be mid-refactor; if you hit a runtime error, open an issue with the traceback and config.
- `enviroment.yml` is spelled as-is in the repo.

---

## License

No license file is currently included. Add one if you plan to share/distribute this repository.