# Class Weighting Experiment

## Overview

For this experiment, I wanted to explore class weighting as another way to address the class imbalance in our facial expression recognition dataset.

Our group has already been working on augmentation to balance the dataset. I thought it would be useful to compare that approach with something simpler, like class weighting, to see whether it can improve Angry and Sad predictions and reduce the model's tendency to predict Happy.

I also discussed this with the professor. He believes augmentation will likely perform better, but agreed that comparing both approaches could be a useful experiment.

## How It Works

Class weighting gives more importance to classes with fewer training images by increasing their contribution to the loss function.

The weights are calculated as:

**Class Weight = Total Training Images / (Number of Classes * Images in That Class)**

For example, if Angry has fewer images than Happy, mistakes on Angry images receive a higher weight during training.

Weights are calculated using only the training data.

## Files Included

- **`losses.py`** – Calculates normal and class-weighted cross-entropy loss and gradients.
- **`training.py`** – Trains the neural network with or without class weighting.
- **`experiment.py`** – Runs both models and compares their performance.
- **`test_weighting.py`** – Checks that the weight calculations and gradients are correct.
- **`__init__.py`** – Supports importing the experiment files.

These files are separate from the team's existing neural network implementation, so the original code remains unchanged.

## Experiment Setup

The experiment uses the original, unaugmented FER2013 dataset with three classes: Angry, Happy, and Sad.

Both models use the same neural network architecture:

**Flatten → Dense(2304, 128) → ReLU → Dense(128, 3)**

| Setting | Value |
|---|---|
| Dataset Split | 70% train / 15% validation / 15% test |
| Epochs | 10 |
| Batch Size | 64 |
| Learning Rate | 0.01 |
| Random Seed | 42 |

Two versions are compared: one with normal cross-entropy and another with class-weighted cross-entropy.

## Running the Experiment

From the repository root, install dependencies:

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
```

Check the calculations:

```bash
python -m unittest capstone_nn.class_weighting.test_weighting -v
```

Run the experiment:

```bash
python -m capstone_nn.class_weighting.experiment --data-dir data/final
```

For a quick test, add `--epochs 1`. Use `--final-test` only when ready for final test evaluation.

## Evaluating Results

The experiment compares:

- **Accuracy:** Overall correct predictions.
- **Recall:** How well each expression is recognized, especially Angry and Sad.
- **Macro F1:** Overall performance giving equal importance to each class.
- **Confusion matrix:** Which expressions are confused.
- **Happy false-positive rate:** How often Angry or Sad images are incorrectly predicted as Happy.

Results are saved to `class_weighting_results/` as JSON and CSV files.

## Comparing With Augmentation

This experiment tests class weighting on the original dataset, while our group's augmentation work uses additional training images.

The purpose isn't to replace augmentation but to see whether a simpler adjustment to the loss function could help with imbalance.

One limitation is that this experiment creates its own 70/15/15 split, which may not contain the exact same images as Melissa's split. We would need matching original-image splits before making a direct comparison with the augmented models.

## Goal

I want to see whether class weighting improves Angry and Sad recall without significantly reducing overall performance.

Even if augmentation performs better, this experiment should help us understand how much of the imbalance can be addressed through the loss function alone and give us another result to discuss in our final project.
