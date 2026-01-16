# Behavior-Conditioned Personalization from EdNet-KT1 Logs

This repository contains code and artifacts for a research project on extracting **behavioral user profiles** from educational interaction logs (EdNet-KT1) in a **self-supervised** manner and converting them into **interpretable personalization controls** for adaptive content generation.

## Project goal
Build a reproducible pipeline that:
1) preprocesses raw event logs into sequential training data,
2) trains a sequence model for **next-step prediction**,
3) extracts **user embeddings** as compact behavioral representations,
4) discovers behavioral groups via clustering,
5) translates behavioral signals into **controls** (pacing, support level, difficulty shift) for personalization.

## Data
EdNet-KT1 is available here: [https://github.com/riiid/ednet.git](https://github.com/riiid/ednet.git). The dataset is not stored in this repository.

## Pipeline 
Raw logs → preprocessing (sorting, sessions, time features, bucketing) → windowing + user-level split → GRU next-step prediction → user embeddings → clustering → cluster interpretation → controls → generation modes.

## Preprocessing (fixed parameters)
- Session split rule: new session if `delta_sec > 30 min` (1800 s) or (if used separately) `SESSION_GAP_SEC = 30*60`
- “Long pause” definition: `long_gap = (delta_sec > 1800 s)`
- Bucketing:
  - `elapsed_bins = [5, 15, 30, 60, 120, 300]` (seconds)
  - `delta_bins   = [10, 60, 300, 1800, 86400]` (seconds)
- Missing/abnormal handling:
  - `user_answer` → `"not_choose"`
  - missing buckets → `-1`
  - rows with missing `timestamp` or `question_id` are dropped

## Dataset preparation / splitting
- Windowing: `SEQ_LEN = 100`, `stride = 50`
- Split strategy: **user-level split** (train/val/test by users) to avoid leakage and to ensure non-empty val/test windows.

## Model (self-supervised)
Task: **next-step prediction** (predict next `qid` from history)
Input per step: `qid + ans + elapsed_b + delta_b`

Hyperparameters:
- GRU with embeddings: `d_model = 64`, hidden size `hidden = 128`
- Optimizer: AdamW, `lr = 1e-3`
- Batch size: `64`
- Epochs: `15`

## Evaluation
- Sequence prediction metrics: Recall@K, MRR@K on val/test (K=5,10)
- Ablation: with time features vs without time features
- Baselines:
  - Frequency baseline
  - Markov baseline

## Main results (final run)
- Best clustering: **K = 2**, silhouette ≈ **0.2167**
- Cluster differences are statistically significant for key behavioral metrics (tempo/activity/long pauses).
- Controls distributions are non-degenerate for `target_pacing`, `hint_level`, `difficulty_shift` (v2 rules).

## Repository artifacts (results_run/)
The `results_run/` folder contains the main outputs used in the paper/report:
- `Table1_model_df_sample20.csv` — sample of preprocessed sequential table
- `Table5_split_sizes.csv` — split parameters and window counts
- `Table6_training_history.csv` — GRU training history
- `Table7_gru_metrics.csv` — GRU metrics (Recall@K / MRR@K)
- `Table8_baselines.csv` — baselines (frequency / Markov)
- `Table12_silhouette_vs_k.csv` — silhouette vs number of clusters
- `Table2_cluster_profile.csv` — cluster-level behavior profiles
- `Table10_cluster_stability.csv` — clustering stability (silhouette/DBI/CH)
- `Table11_cluster_differences_stats.csv` — Mann–Whitney + effect size
- `Table9_generation_modes_by_controls.csv` — generation modes by controls (no prompts)
- Embeddings and model:
  - `user_embeddings.npy`, `user_ids.npy`
  - `v2_seq100_stride50_gru_best.pt`
  - `vocab.json`
- Figures:
  - `v2_seq100_stride50_fig*.png` (EDA, training curve, silhouette, cluster plots, controls distributions)

## How to reproduce
Recommended: run via Google Colab: https://colab.research.google.com/drive/11kpOrzY0thHSjvR0fKd-PtAjIrjQkuHI?usp=sharing#scrollTo=TtDcnB_oarrV.

## Limitations
- Selection bias: embeddings are more reliable for users with longer histories; short histories provide limited signal.
- Moderate cluster separability (silhouette ~0.22 in the final run).
- Controls are derived from rule-based mapping; a learned mapping is a natural next step.
- The effect of personalization is demonstrated through proxies; a full evaluation requires integration and controlled comparison.

## Contact
Author: <Shuvalova Ksenia>
Email: <shuvalova.kk@icloud.com>
