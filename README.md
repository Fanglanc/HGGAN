# HGGAN: A Hierarchical Graph Generative Adversarial Framework for Urban Land Use Planning

This anonymous repository contains the code for **HGGAN**, a hierarchical graph generative adversarial framework for conditional urban land use planning. HGGAN models an urban grid as a spatial graph and follows a coarse-to-fine generation pipeline: a coarse generator captures macro-scale structure, while a fine generator refines local allocation using coupled convolutional and graph-based reasoning.

![HGGAN framework](./HGGAN_framework.png)

## Overview

Urban land use planning is a multi-objective problem that must balance accessibility, livability, sustainability, and spatial coherence. Prior generative approaches are largely raster-based, which makes it harder to model long-range spatial dependencies and often leaves evaluation overly focused on distribution matching. HGGAN addresses these issues by combining:

- **Graph-native spatial modeling** over a discretized urban grid
- **Hierarchical coarse-to-fine generation** for macro structure and local refinement
- **Dual-stream fine refinement** with convolutional and functional graph reasoning
- **Comprehensive evaluation** using distribution metrics, rule-based urban planning dimensions, and uncertainty-aware LLM assessment

## Main Ideas

### 1. Graph-native conditional urban planning
The target region is represented as a grid graph, where each cell is a node and node features represent POI composition. This lets the model explicitly capture non-local relationships through message passing instead of relying only on local convolutional texture.

### 2. Hierarchical generation
HGGAN uses a two-stage generator:

- **Coarse generator**: predicts macro-scale signals including global intensity, road probability, and latent zoning structure
- **Fine generator**: produces detailed POI allocation using a dual-stream design that combines local spatial refinement and functional graph interactions

### 3. Multi-layer evaluation
The evaluation protocol combines three complementary views:

- **Distribution-based metrics** such as KL / JS / Hellinger / Cosine / Wasserstein distributional distances
- **Rule-based urban planning scores** summarized into 6 selected dimensions:
  - Spatial Coherence
  - Development Compactness
  - Healthy Environment
  - Land Use Balance
  - Community Convenience
  - Urban Resilience
- **Uncertainty-aware LLM evaluation** with repeated runs to quantify semantic assessment variability

## Data

The dataset is not included in this repository at this stage. A later update will provide the releasable data package and instructions for organization.

## Environment
A Conda environment specification is provided. A typical setup is:

```bash
conda create -n hggan python=3.10 -y
conda activate hggan
pip install -r requirements.txt
```

## Training

Example training command for the anchor-based functional backend:

```bash
python train.py \
  --data_dir ./data \
  --output_dir ./result/func_anchors \
  --func_backend anchors \
  --anchor_m 32 \
  --anchor_key_dim 32
```

This stage trains the hierarchical generator and saves checkpoints under the specified output directory.

## Generation

After training, generate plans from the trained checkpoint:

```bash
python generate.py \
  --ckpt ./result/func_anchors/best_model.pt \
  --data_dir ./data \
  --out_npz ./result/func_anchors/generated/generated_testset.npz \
  --batch_size 8 \
  --func_mode cached
```

## Evaluation

### 1) Distribution-based metrics + urban planning 6 dimensions

```bash
python evaluate_generated_plans.py \
  --baseline_dir ./result \
  --models func_anchors \
  --data_dir ./data \
  --tag testset \
  --presence_mode argmax \
  --dimension_profile planning_6dimension \
  --do_quant \
  --save_all
```

This computes quantitative distribution metrics and the six selected urban planning dimensions.

### 2) LLM evaluation with uncertainty

```bash
python robust_llm_evaluator.py \
  --provider gemini \
  --model gemini-2.5-pro \
  --generated_dir ./result/func_anchors/generated \
  --n_sample 30 \
  --n_runs 5 \
  --temperature 0.7 \
  --output ./result/llm_evaluation/gemini25pro_uncertainty.json \
  --verbose
```

This samples generated plans, evaluates each plan multiple times, and aggregates uncertainty-aware semantic scores across the six selected planning dimensions.


