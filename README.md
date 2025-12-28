# HybridFAS

HybridFAS is a hybrid **CNN–Transformer** architecture for **Face Anti-Spoofing (FAS)**, designed to learn complementary representations for distinguishing **live** and **spoofed** faces under challenging real-world conditions.

The project focuses on **generalization**, robustness, and biometric-specific evaluation rather than inflated accuracy scores.

---

## 🔍 Project Overview

Face Anti-Spoofing is a critical component of biometric security systems (e.g., KYC, access control, identity verification).  
HybridFAS explores a **dual-branch learning strategy** combined with **global reasoning** to capture both local spoof cues and holistic facial structure.

The model is trained and evaluated on the **CelebA-Spoof** dataset using official splits.

---

## ✨ Key Features

- Hybrid **CNN + Transformer** backbone
- Dedicated **Spoof** and **Real** feature branches
- **Learnable fusion gate** for adaptive feature weighting
- Patch-based representation for local cue modeling
- Robustness to **JPG ↔ PNG domain shifts**
- Evaluation using biometric metrics:
  - **AUC**
  - **APCER**
  - **BPCER**
  - **EER**
  - **HTER**

---

## 🧠 Architecture Summary

- **Patch-based CNN embedding** for local texture extraction  
- **Spoof branch**: models edge artifacts, reflections, and noise patterns  
- **Real branch**: captures skin texture, illumination consistency, and facial structure  
- **Transformer encoder** for global contextual reasoning  
- **Fusion gate** dynamically balances real vs spoof evidence  

This design encourages the model to learn **complementary and discriminative cues** rather than relying on a single appearance-based signal.

---

## 📊 Results (CelebA-Spoof)

| Metric | Performance |
|------|------------|
| Validation Accuracy | ~94% |
| Test Accuracy | 76–86% |
| AUC | ~0.93 |
| EER | ~0.155 |
| APCER / BPCER | Balanced |

While strong validation performance is achieved, a **notable performance drop on the test set** is observed.  
This behavior highlights the **severe domain shift and generalization challenges** intentionally present in the CelebA-Spoof benchmark.

Confusion matrices and evaluation plots are available in the `assets/` directory.

---

## 📉 Discussion: Generalization Challenge

Despite iterative architectural improvements (v1.0 → v2.5), test performance consistently lags behind validation results.  
This reflects:
- Unseen attack types in the test split
- Identity-disjoint protocols
- Real-world degradations (blur, compression, lighting)
- Limited reliance on appearance-only cues

Such behavior is expected for CelebA-Spoof and emphasizes the importance of **robust evaluation over optimistic validation metrics**.

---

## ⚠️ Limitations

- Single-frame RGB input only
- No external datasets or pretraining used
- No frequency-domain supervision
- Evaluated on a single benchmark dataset

---

## 🚀 Future Improvements

- Frequency-domain regularization (FFT / DCT features)
- Stronger artifact-aware data augmentation
- Cross-dataset evaluation (e.g., CASIA-FASD)
- Temporal modeling for video-based anti-spoofing
- Metric-driven optimization (AUC / EER)

---

## 📁 Dataset

This project uses the **CelebA-Spoof** dataset.  
The dataset is **not included** in this repository due to licensing restrictions.

---

## 🛠 Usage

### Training
```bash
python training/train.py --config configs/train.yaml
