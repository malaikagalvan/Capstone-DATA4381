# Franklin's class-weighting comparison — DATA 4381

**Purpose:** Compare the team's three-class **original-data** NumPy neural network with ordinary cross-entropy versus inverse-frequency class-weighted cross-entropy. The goal is to test whether weighting reduces the tendency to predict Happy and improves Angry/Sad recall. Better performance is **not** guaranteed.

## What is included

- `losses.py`: full standalone version of softmax, normal/weighted cross-entropy, correct weighted gradient, and a class-weight calculator.
- `training.py`: full standalone copy of the team's training/evaluation loop, with optional class weights. `evaluate` uses **unweighted** loss for fair comparisons.
- `experiment.py`: loads **original unaugmented** FER2013 images, splits 70/15/15, trains two identical scratch neural networks, prints class-level metrics, and writes output files.
- `test_weighting.py`: tests the weight formula, unweighted compatibility, and weighted gradient correctness.
- `__init__.py`: enables imports from this folder as a Python package.

## Where to place these files

Extract the package at the **root** of a cloned `Capstone-DATA4381` repository so the new folder becomes:

```
Capstone-DATA4381/
  capstone_data/image_loader.py                 # EXISTING; don't edit
  capstone_nn/losses.py                         # EXISTING; don't edit
  capstone_nn/training.py                       # EXISTING; don't edit
  capstone_nn/model.py                          # EXISTING; don't edit
  capstone_nn/layers.py                         # EXISTING; don't edit
  capstone_nn/activations.py                    # EXISTING; don't edit
  capstone_nn/franklin_class_weighting/         # NEW
    __init__.py
    losses.py
    training.py
    experiment.py
    test_weighting.py
    README.md
```

**IMPORTANT:** `capstone_nn/franklin_class_weighting/losses.py` is a **new** file. It does NOT replace `capstone_nn/losses.py`. Same for `training.py`. GitHub can store both because they live in different folders.

## Setup on your computer

Open the repository root in VS Code / terminal and use the Python environment you already have for the capstone.

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

This experiment uses NumPy and the team's image-loader code (Pillow/tqdm from requirements); it does not need scikit-learn.

Put the **original cleaned 48x48 FER2013** images under `data/final/{angry,happy,sad}`, or pass a different directory whose class folders are `angry`, `happy`, `sad` (also accepts `angry_cleaned`, `happy_cleaned`, etc.). **Do not give this script the augmented dataset** for the original-data comparison.

Verify the class-weighting math first, without any images:

```bash
python -m unittest capstone_nn.franklin_class_weighting.test_weighting -v
```

Run the actual experiment:

```bash
python -m capstone_nn.franklin_class_weighting.experiment --data-dir data/final
```

Or if your original data is somewhere else:

```bash
python -m capstone_nn.franklin_class_weighting.experiment --data-dir "C:/Users/YourName/Downloads/cleaned_data"
```

Quick functional run (not enough epochs for reporting results):

```bash
python -m capstone_nn.franklin_class_weighting.experiment --data-dir data/final --epochs 1
```

Options: `--epochs` (default 10), `--batch-size` (64), `--learning-rate` (0.01), `--seed` (42), `--output-dir` (default `class_weighting_results`). All model hyperparameters and initial weights are the same between runs. `--final-test` evaluates the test split only when you are ready for final reporting.

## What results mean

The experiment writes these new files to `class_weighting_results/`:

- `comparison.json`: class weights, training counts, validation accuracy, balanced accuracy, macro F1, per-class precision/recall/F1, Happy false-positive rate, and confusion matrices.
- `normal_training_history.csv` and `weighted_training_history.csv`: epoch-by-epoch training and validation numbers.
- `normal_val_confusion_matrix.csv` and `weighted_val_confusion_matrix.csv`: rows=actual Angry/Happy/Sad; columns=predicted Angry/Happy/Sad.

**Happy false-positive rate** = the percentage of actual Angry or Sad images incorrectly predicted as Happy. Compare it alongside macro F1 and per-class recall; focusing only on this one rate can hide other mistakes.

The loss scales minority-class errors using `w[c] = N_train / (3 * count_of_class_c)`. The backward gradient is weighted consistently. The model architecture is Flatten → Dense(2304,128) → ReLU → Dense(128,3), matching the team's scratch NN adjusted to three classes.

## Important fair-comparison limitation

This independent script creates a **new NumPy stratified 70/15/15 split with seed 42**, by calling the current `capstone_data.image_loader.stratified_split_indices`. Melissa's augmentation notebooks use **scikit-learn** `train_test_split` before creating augmented images, which is *not guaranteed to select the same actual images* as this script even with the same ratios and seed. So do **not** claim a definitive head-to-head comparison against the augmented models until the group shares **identical original training/validation/test image IDs or split manifests**. Never split the already-augmented images again: related original and augmented images can leak across the train/test boundary.

Also confirm the original source images have been deduplicated and the same image hasn't been included multiple times across partitions. Final model selection should be based on validation, not repeated testing on the held-out test split.

## GitHub contribution

Create a new branch, copy the **new folder** into the existing repo, and push it as a pull request. No shared files need to be edited.

```bash
git checkout -b franklin/class-weighting
git add capstone_nn/franklin_class_weighting
git commit -m "Add independent class-weighting comparison for scratch NN"
git push -u origin franklin/class-weighting
```

GitHub > Pull requests > New pull request. Your team can review it and merge it, and you'll have attributable code contribution.
