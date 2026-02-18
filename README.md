# Indonesian Traditional Food Image Classification

## Overview

Proyek ini bertujuan untuk melakukan klasifikasi citra terhadap 15 jenis makanan tradisional Indonesia menggunakan dataset hasil scraping Google Images.

Dataset training tidak memiliki label, sehingga diperlukan proses pelabelan mandiri sebelum model dapat dilatih. Pendekatan yang digunakan dalam proyek ini menggabungkan pseudo-labeling, fine-tuning model pretrained modern, serta representasi berbasis self-supervised embedding.

Kompetisi dievaluasi menggunakan metrik Accuracy karena distribusi kelas seimbang.

---

## Dataset Description

Dataset terdiri dari citra digital 15 jenis makanan tradisional Indonesia.

### Data Summary
- Total train images: 4,257
- Total test images: 2,057
- Jumlah kelas: 15
- Tidak terdapat class imbalance
- Evaluation metric: Accuracy

### Files

train folder
Berisi gambar makanan tradisional tanpa label.
Peserta harus melakukan pelabelan secara mandiri sebelum training.

test folder
Berisi gambar makanan tradisional tanpa label untuk prediksi.

test.csv
Contoh format submission.

---

## Problem Framing

- Task: Multi-class Image Classification
- Number of classes: 15
- Evaluation metric: Accuracy
- Tidak terdapat class imbalance sehingga tidak diperlukan weighted loss.

---

## Approach 1: Fine-Tuning Pretrained CNN Models

Pendekatan pertama menggunakan transfer learning dengan fine-tuning model pretrained modern.

### Model Pipeline

MODEL_PIPELINE = {
    "EfficientNetV2S": {
        "build_fn": build_efficientnet_v2s, "config": config_effnet_v2s
    },
    "ConvNeXtTiny": {
        "build_fn": build_convnext_tiny,    "config": config_convnext
    },
    "DenseNet169": {
        "build_fn": build_densenet,"config": config_densenet
    }
}

### Workflow

1. Pseudo-labeling pada data train
2. Sanity check manual untuk memperbaiki label yang keliru
3. Fine-tuning model pretrained
4. Evaluasi dan model selection berdasarkan validation performance

Pendekatan ini menekankan kualitas label melalui verifikasi manual setelah pseudo-labeling.

### Results (Approach 1)

- Public Accuracy: 91.1%
- Private Accuracy: 92.4%

---

## Approach 2: DINOv2 Embedding + Iterative Pseudo-Labeling

Pendekatan kedua menggunakan representasi berbasis self-supervised learning.

### Feature Extraction

- Menggunakan DINOv2 embedding berukuran 768 dimensi
- Model tidak dilatih end-to-end pada pixel space
- Menggunakan embedding sebagai representasi fitur tetap

### Strategy

1. Ekstraksi embedding 768 dimensi untuk seluruh gambar
2. Pseudo-labeling tanpa sanity check manual
3. Hanya sebagian data train digunakan (tidak seluruhnya dilabeli)
4. Dilakukan 2 iterasi pseudo-label refinement

Pendekatan ini lebih efisien dan lebih scalable dibanding manual relabeling.

### Results (Approach 2)

- Public Accuracy: 91.3%
- Private Accuracy: 94.0%

Pendekatan embedding menunjukkan generalisasi yang lebih baik pada private leaderboard.

---

## Key Insights

1. Embedding self-supervised (DINOv2) memberikan representasi fitur yang sangat kuat bahkan tanpa fine-tuning penuh.
2. Fine-tuning CNN modern tetap kompetitif, terutama jika kualitas label dijaga dengan sanity check manual.
3. Iterative pseudo-labeling efektif meningkatkan performa meskipun tidak semua data train digunakan.
4. Karena kelas seimbang, Accuracy menjadi metrik yang cukup representatif.

---

## Comparison of Approaches

Approach 1
- Fine-tuning pretrained CNN
- Manual sanity check labeling
- Lebih berat secara komputasi
- Private Accuracy: 92.4%

Approach 2
- DINOv2 768-d embedding
- Iterative pseudo-labeling tanpa manual relabel
- Lebih ringan dan efisien
- Private Accuracy: 94.0%

---

## Conclusion

Pendekatan berbasis embedding self-supervised dengan DINOv2 dan iterative pseudo-labeling memberikan performa terbaik pada kompetisi ini.

Hasil menunjukkan bahwa representasi visual modern berbasis self-supervised learning mampu mengungguli fine-tuning CNN konvensional dalam skenario semi-supervised labeling.

Proyek ini menunjukkan efektivitas kombinasi pseudo-labeling dan pretrained representation dalam membangun classifier yang robust meskipun data awal tidak berlabel.