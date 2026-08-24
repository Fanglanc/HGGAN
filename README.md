# HGGAN: A Hierarchical Graph Generative Adversarial Framework for Urban Land Use Planning

This repository contains the implementation of **HGGAN**, a hierarchical graph generative adversarial framework for instruction-guided urban land use planning. HGGAN formulates land use configuration as a conditional graph generation problem and adopts a coarse-to-fine architecture to jointly model local spatial structure and long-range functional dependencies.

![HGGAN framework](./HGGAN_architecture.png)

## Overview

Urban land use planning requires balancing multiple and often competing objectives, including spatial accessibility, functional balance, environmental quality, and urban resilience. Existing deep generative approaches commonly represent urban layouts as raster grids and rely primarily on convolutional architectures. While effective at capturing local spatial regularity, these approaches provide limited support for modeling long-range relationships such as functional complementarity, transportation connectivity, and land use compatibility across distant locations. Evaluation presents a second challenge. Distributional similarity alone does not determine whether a generated plan is coherent from a planning perspective: two plans with similar land use proportions may differ substantially in spatial organization, service accessibility, environmental exposure, functional balance, or resilience. 

HGGAN addresses these limitations through:

- **Region graph formulation** of urban land use configuration
- **Hierarchical coarse-to-fine generation** for macro-scale structure and detailed allocation
- **Dual-stream fine refinement** for local spatial regularity and long-range functional dependencies
- **Multi-objective training** integrating local, global, structural, and adversarial supervision
- **Comprehensive evaluation protocol** combining distributional metrics, interpretable planning indicators, and uncertainty-aware LLM assessment

## Architecture

### Coarse Graph Generator

The coarse generator uses graph message passing and positional encoding to predict:

- Development intensity
- Road probability
- Latent zoning structure

These representations establish the global planning structure before fine-scale generation.

### Fine Dual-Stream Generator

The fine generator combines two branches:

- **Spatial stream:** convolutional refinement for local neighborhood consistency
- **Functional stream:** anchor-based graph message passing for long-range dependencies between functionally similar areas

Bidirectional cross-attention and a mixture gate fuse the two streams before generating node-level land use distributions.

### Conditional Discriminator

A conditional Wasserstein discriminator encourages generated plans to be spatially realistic and consistent with the surrounding urban context and planning requirements.

## Training Objective

HGGAN combines four losses:

- **Reconstruction loss** for local land use assignment
- **KL regularization** for global land use composition
- **Road prior** for circulation structure
- **Adversarial loss** for higher-order spatial realism

The adversarial term uses learnable uncertainty weighting to reduce unstable critic feedback during training.

## Evaluation

HGGAN uses three complementary evaluation layers:

1. **Distributional fidelity**
   - KL divergence
   - JS divergence
   - Hellinger distance
   - Cosine distance
   - Wasserstein distance

2. **Six planning-quality dimensions**
   - Spatial Coherence
   - Development Compactness
   - Healthy Environment
   - Land Use Balance
   - Community Convenience
   - Urban Resilience

3. **Uncertainty-aware LLM evaluation**
   - Repeated semantic assessment
   - Dimension-level scores
   - Confidence and inter-run variability

## Data

The dataset is not included in this repository at current stage. A later update will provide the releasable data package and instructions for organization.


## Environment

```bash
conda create -n hggan python=3.10 -y
conda activate hggan
pip install -r requirements.txt
```

## Training

```bash
python train.py \
  --data_dir ./data \
  --output_dir ./result/func_anchors \
  --func_backend anchors \
  --anchor_m 32 \
  --anchor_key_dim 32
```

## Generation

```bash
python generate.py \
  --ckpt ./result/func_anchors/best_model.pt \
  --data_dir ./data \
  --out_npz ./result/func_anchors/generated/generated_testset.npz \
  --batch_size 8 \
  --func_mode cached
```

## Evaluation

### Distributional + Planning Metrics

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

### LLM Evaluation

```bash
python robust_llm_evaluator.py \
  --provider gemini \
  --model gemini-2.5-pro \
  --generated_dir ./result/func_anchors/generated \
  --n_sample 30 \
  --n_runs 10 \
  --temperature 0.7 \
  --output ./result/llm_evaluation/gemini25pro_uncertainty.json \
  --verbose
```


