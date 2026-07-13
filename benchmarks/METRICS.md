# Final Empirical Evaluation Metrics

## 1. Evaluation Methodology

**Objective (Why):**
To empirically validate the performance of the final `SignLanguageGRU` architecture on continuous real-world data and address the evaluation gaps identified in the FYP dissertation.

**Dataset (What):**
A high-throughput HDF5-compiled dataset consisting of **37,960 sequence samples** distributed across **89 distinct Indian Sign Language (ISL) classes**.

**Evaluation Domains (How/Whom):**
The dataset was partitioned into isolated training/validation clusters and a strictly separated **unseen test cluster (`webcam` domain)**. This design ensured that model evaluation occurred on data independent from the training distribution, simulating real-world deployment scenarios involving previously unseen users and environmental conditions.

**Assumptions:**

* The evaluation assumes a fixed vocabulary of **89 ISL classes**, in addition to an `idle` / `__reject__` class used to represent background noise and non-signing states.
* Metrics reflect isolated single-sequence classification performance using **Top-1 prediction accuracy**, prior to the application of any higher-level NLP sequence modelling, grammar correction, or language reconstruction techniques.

---

## 2. Pipeline Configuration & Architecture

The evaluated system utilised the canonical checkpoint (`models/model.pth`) configured using the following architecture and training parameters:

* **Feature Space:** 506 dimensions per frame consisting of:

  * 126 raw landmark coordinates
  * 126 face-relative spatial coordinates
  * 1 proximity scalar
  * Frame-to-frame velocity derivatives for all extracted features

* **Sequence Length:** 20 temporal frames per prediction window

* **Model Topology:**
  3-layer Bidirectional GRU (hidden size: 64) incorporating a dedicated Proximity Attention mechanism

* **Training Configuration:**

  * Epochs: 50
  * Learning Rate: 0.0003
  * Label Smoothing: 0.05
  * Class Weighting applied to reduce majority-class bias

---

## 3. Empirical Results

The final model demonstrated strong performance across both validation and isolated evaluation domains.

Interestingly, the unseen webcam domain achieved higher performance than the validation split (**98.13% vs 91.34%**). This may indicate that the isolated webcam sequences contained lower noise levels and more temporally consistent signing patterns compared with the validation dataset despite remaining outside the training distribution. While these results suggest strong transfer capability, broader cross-user and cross-environment evaluation would be necessary to confirm generalisation at larger scales.

| Metric                           | Measured Score | Description                                                                                                                 |
| :------------------------------- | :------------: | :-------------------------------------------------------------------------------------------------------------------------- |
| **Validation Accuracy**          |     91.34%     | Performance on held-out validation data within the training domain                                                          |
| **Unseen Data Accuracy (Top-1)** |   **98.13%**   | Performance on isolated webcam sequences outside the training distribution                                                  |
| **Macro F1 Score**               |     96.61%     | Equal-weight average F1 score across all classes, indicating consistent performance across majority and minority categories |
| **Weighted F1 Score**            |     97.73%     | F1 score adjusted according to class support sizes                                                                          |
| **Balanced Accuracy**            |     98.20%     | Mean recall across all classes, reducing the influence of class imbalance                                                   |

---

## 4. Confusion Matrix Analysis (Top-10 Misclassifications)

Analysis of the dominant confusion pairs within unseen evaluation data highlights the system's remaining limitations. Most errors occurred between gestures exhibiting highly similar hand configurations, spatial trajectories, or facial anchor relationships.

| Ground Truth Sign | Predicted Sign | Error Count | Architectural Analysis                                                                                                                                                             |
| :---------------- | :------------- | :---------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nice`            | `good`         |     408     | **Semantic / Visual Overlap:** These signs exhibit highly similar hand configurations and movement trajectories across certain ISL variations, resulting in substantial ambiguity. |
| `hard`            | `idle`         |      7      | **Motion Attenuation:** Low-motion gestures were occasionally interpreted as background or resting states.                                                                         |
| `blind`           | `female`       |      6      | **Anchor Proximity:** Both signs involve similar facial interaction regions with overlapping spatial characteristics.                                                              |
| `blind`           | `idle`         |      6      | **Confidence Thresholding:** Subtle motion patterns caused prediction confidence to decrease below the commit threshold.                                                           |
| `happy`           | `idle`         |      6      | **Transition Ambiguity:** Fluid movement transitions were occasionally interpreted as non-gesture states.                                                                          |
| `ugly`            | `beautiful`    |      6      | **Directional Similarity:** These signs contain similar spatial structures but differ primarily in movement direction.                                                             |
| `happy`           | `expensive`    |      5      | **Temporal Overlap:** Shared repetitive two-handed movement patterns increased confusion.                                                                                          |
| `poor`            | `strong`       |      5      | **Spatial Similarity:** Similar chest-level movement dynamics and bounding region characteristics caused overlap.                                                                  |
| `0`               | `idle`         |      4      | **Static Gesture Filtering:** Static number gestures were occasionally interpreted as idle hand states.                                                                            |
| `deaf`            | `idle`         |      4      | **Temporal Window Limitation:** The brief facial interaction may have occurred partially outside the 20-frame temporal window.                                                     |

---

## 5. Limitations

Although the model demonstrated strong performance under isolated evaluation conditions, several limitations remain.

* The evaluation was constrained to a fixed vocabulary of **89 ISL signs**.
* Classification was performed on isolated sequence windows rather than continuous sentence-level signing.
* Evaluation across larger signer populations and broader environmental diversity remains necessary.
* NLP-based sequence modelling and grammar reconstruction techniques were not integrated into the reported results.

Future work may focus on extending the architecture toward continuous sign language recognition and sentence-level semantic interpretation.

---

## 6. Summary Conclusion

The empirical evaluation demonstrates the effectiveness of the proposed **506-dimensional face-relative feature representation**. By achieving **98.13% unseen data accuracy** and a **96.61% Macro F1 Score**, the system demonstrated reliable discrimination across **89 ISL sign classes** in real-time conditions. The results suggest that the face-relative feature design improved spatial consistency while enabling robust recognition performance across isolated real-world sign sequences.
