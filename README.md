# HybridFAS

HybridFAS is a hybrid CNN–Transformer architecture for Face Anti-Spoofing, designed to learn complementary cues for live and spoof face detection.

The model was trained and evaluated on the CelebA-Spoof dataset.

## Key Features
- Hybrid CNN + Transformer backbone
- Dedicated Spoof and Real feature branches
- Learnable fusion gate for adaptive feature weighting
- Robust to JPG ↔ PNG domain shifts
- Evaluation using AUC, APCER, BPCER, EER, and HTER

## Architecture
- Patch-based CNN embedding
- Spoof branch: edge, reflection, noise modeling
- Real branch: skin, illumination, structure modeling
- Transformer encoder for global reasoning

## Results (CelebA-Spoof Test Set)
- AUC: 0.93
- EER: 0.155
- APCER ≈ BPCER (balanced performance)

Confusion matrix and evaluation plots are available in the `assets/` directory.

## Dataset
This project uses the CelebA-Spoof dataset.
The dataset is NOT included due to licensing restrictions.

## Usage
### Training
