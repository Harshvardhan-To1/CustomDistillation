# Custom Two-Stage LLM Distillation

This project introduces a custom two-step method for distilling large language models (LLMs) designed to optimize memory usage. The method involves saving teacher model logits in the first stage, enabling users with limited VRAM to perform distillation efficiently in the second stage.

## Overview

Distillation is a widely-used technique to transfer knowledge from a large teacher model to a smaller student model. However, conventional distillation methods often require significant VRAM, which can be a barrier for users with limited resources. This project addresses the problem by decoupling the distillation process into two stages:

1. **Logit Saving (Stage 1):**
   - The teacher model generates and saves logits from the training data.
   - This step can be performed on machines with higher VRAM.

2. **KL-Divergence Distillation (Stage 2):**
   - The student model is trained using the saved teacher logits, significantly reducing VRAM requirements.

## Features

- **Memory Efficient:** Suitable for users with low VRAM capacity.
- **Two-Stage Process:** Decouples the resource-intensive logit generation from the distillation step.
- **KL-Divergence Loss:** Implements knowledge transfer using KL-divergence between student predictions and saved teacher logits.

## How It Works

### Stage 1: Logit Saving
1. Load the teacher model.
2. Generate logits for the training data using the teacher model.
3. Save the generated logits to disk.

### Stage 2: Student Training
1. Load the student model.
2. Load the saved teacher logits.
3. Train the student model using KL-divergence as the loss function.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Harshvardhan-To1/CustomDistillation.git
   cd CustomDistillation
