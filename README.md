# GRPO (Group Relative Policy Optimization)

A PyTorch implementation of Group Relative Policy Optimization for training language models with reward functions.

## Overview

This repository implements GRPO, a policy optimization algorithm that uses group-based advantage estimation and relative rewards to train language models. The implementation includes:

- GRPO algorithm implementation
- Policy model wrapper for language models
- Multiple reward functions
- Training utilities

## Installation

1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```


## Components

### GRPO Algorithm

The core GRPO implementation (`grpo.py`) provides:
- Group-based advantage estimation
- KL-divergence constrained policy updates
- Clipped policy gradient optimization

### Policy Model

The policy model (`policy.py`) wraps Hugging Face transformers models and provides:
- XML-formatted response generation
- Special token handling
- Response formatting utilities

### Training

Example usage for training: see `example.py`
