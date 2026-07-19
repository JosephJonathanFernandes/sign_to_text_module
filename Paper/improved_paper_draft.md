**Title**
Edge-Optimized Perception Engine for 300-Class Indian Sign Language Translation

**Author1:** [Last Name], [First Name], [Affiliation];
**Author2:** [Last Name], [First Name], [Affiliation];
**Author3:** [Last Name], [First Name], [Affiliation];

**Abstract**
Communication barriers severely restrict the societal participation of the Deaf and Hard-of-Hearing (DHH) community, largely because Indian Sign Language (ISL) is a spatial, multidimensional language rarely understood by the broader hearing population. Existing computational translation systems often face restrictive computational overhead or lack real-world stability due to inter-sign movement epenthesis. This paper presents a low-latency, edge-optimized spatial architecture designed to facilitate real-time, continuous ISL translation. The system ingests a standardized 506-dimensional spatio-temporal vector extracted via MediaPipe landmark topology tracking. This input is dynamically routed through a dual-branch feature fusion pipeline that combines a Spatial Graph Neural Network (GNN) and a Conv1D frontend. Sequential modeling is handled by a 3-layer Bidirectional Gated Recurrent Unit (BiGRU), optimized with a domain-adversarial branch to enforce signer-invariance. Compiled via ONNX FP32 optimization and stabilized by a recurrent 2-of-3 momentum window filter, the engine achieves a 98.33% unseen test accuracy and a 97.84% Macro F1-score across 300 distinct ISL classes. Operating at just 6.22 ms per frame sequence (160.7 FPS) on consumer-grade CPUs, the architecture effectively suppresses high-frequency movement epenthesis noise, providing a highly scalable, real-time foundation for bidirectional assistive communication tools.

**Keywords:** Indian Sign Language, Edge Computing, Graph Neural Networks, Real-Time Translation, Assistive Technology, Motion Epenthesis

**Introduction/Background**
While Indian Sign Language (ISL) serves as the primary language for millions within India's Deaf and Hard-of-Hearing (DHH) community, the lack of widespread fluency among the hearing population creates systemic accessibility barriers in education, healthcare, and public services. Recent advancements in deep learning have catalyzed the development of automated Sign Language Translation (SLT) systems. However, a significant gap remains between theoretical accuracy and real-world deployability. Existing architectures often rely on restrictive, compute-heavy transformers or fail to adequately process continuous spatial grammar. Consequently, these systems suffer from cognitive lag and prediction instability, disrupting natural conversational flow and limiting their viability as real-time assistive devices.

**Methods**
To achieve real-time, signer-invariant translation without relying on cloud-based GPU infrastructure, this system implements a highly optimized, localized perception engine. Raw webcam inputs are parsed to extract high-fidelity 3D hand and face landmarks, which are mathematically mapped to a posture-invariant 506-dimensional spatial vector.

This spatial data is buffered into a strict 20-frame sliding window and routed through a dual-branch neural architecture. The first branch utilizes a Spatial GNN to explicitly model the anatomical topology of the hands, while the second branch applies point-wise and depth-wise 1D convolutions for temporal feature extraction. The fused features are processed by a 3-layer BiGRU equipped with a proximity-weighted attention bias (derived from hand-to-face distance) and a Domain-Adversarial Neural Network (DANN) branch to enforce robust generalization across diverse users.

*Accessibility Note for Document Formatting:*

> **[Insert Figure 1 Here]**
> *Alt-Text:* A linear flowchart illustrating the end-to-end perception pipeline. It begins with the extraction of a 506-dimensional spatio-temporal vector, which splits into a dual-branch fusion consisting of a Spatial GNN and a Conv1D Frontend. These branches merge into a 3-layer BiGRU with proximity-aware attention. The entire neural network is enclosed in a bounding box labeled "ONNX FP32 CPU Inference: 6.22ms". The final output routes through a 2-of-3 momentum window filter before yielding the stabilized translated text.

**Results**
The model was trained and empirically evaluated on a robust dataset comprising 93,798 samples across an expanded 300-class ISL vocabulary, compiled using an optimized HDF5 data engine. The architecture achieved:

* **Unseen Test Data Accuracy:** 98.33%
* **Macro F1-Score:** 97.84%
* **Inference Speed:** 6.22 ms per frame sequence (sustaining 160.7 FPS) via ONNX FP32 optimization.

This 6.22 ms computational footprint utilizes only 3.1% of a standard 200 ms real-time interaction budget. During live execution, the temporal post-processor—configured with a strict 2-of-3 momentum commit strategy (`momentum_commit_count=2`, `momentum_window=3`) and a dual-stage filtering heuristic (`confidence_threshold=0.12`, `momentum_min_avg_conf=0.60`)—successfully suppressed inter-sign transition flicker and hallucinated outputs.

**Discussion**
Scaling sign recognition vocabularies typically introduces severe latency degradation and catastrophic forgetting. However, the integration of ONNX FP32 optimization demonstrated that high-throughput inference (160.7 FPS) is highly sustainable on standard CPU hardware, even at a massive 300-class scale. Furthermore, the recurrent momentum filtering proved critical in bridging the gap between isolated sign classification and continuous real-world communication. By mathematically rejecting the non-semantic motion epenthesis that commonly triggers false positives in traditional sliding-window architectures, the system ensures a stable, highly readable output.

**Conclusion/Implications for the AT field**
This research demonstrates that complex spatio-temporal neural architectures can be successfully optimized for extreme edge deployment without sacrificing vocabulary scale or predictive accuracy. By eliminating the dependency on high-end hardware accelerators, this perception engine democratizes AI accessibility. It establishes a scalable, ultra-low-latency framework that can be embedded directly into consumer laptops, mobile devices, and public smart kiosks, dramatically expanding the inclusive communication infrastructure available to the DHH community.

**References**
[1] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand, and Victor Lempitsky. 2016. Domain-Adversarial Training of Neural Networks. *Journal of Machine Learning Research* 17, 59 (2016), 1–35.
[2] Akhil Sridhar, Rajeev G. Ganesan, Pratyush Kumar, and Mitesh Khapra. 2020. INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition. In *Proceedings of the 28th ACM International Conference on Multimedia (MM '20)*. Association for Computing Machinery, New York, NY, USA, 1366–1375.
[3] Sijie Yan, Yuanjun Xiong, and Dahua Lin. 2018. Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition. In *Thirty-Second AAAI Conference on Artificial Intelligence*.

**Contact information of the communicating author**
**Title:** [Mr./Ms./Dr.]
**Name:** [Insert Full Name]
**Designation:** [Insert Academic or Professional Title, e.g., Student/Researcher]
**Affiliation:** [Insert University/Organization Name]
**Email ID:** [Insert Email Address]
**Contact number:** [Insert Phone Number]
