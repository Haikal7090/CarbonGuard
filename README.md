# CarbonGuard
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)

CarbonGuard v5.1: Hybrid Unsupervised ML for Real-Time Subsea CCUS Leak Detection
Location Focus: Tangguh UCC Project — West Papua, Indonesia.

CarbonGuard is an unsupervised machine learning (ML) framework designed to detect micro-leaks in subsea CO2 injection wells in real-time. Since actual field leakage data (anomalies) are extremely scarce, the system trains on normal operational profiles using a combination of multimodal sensor data: PDG (Permanent Downhole Gauge) for pressure data and DAS (Distributed Acoustic Sensing) for acoustic data.

Key Features (v5.1 Updates):
1. Hybrid Dual-Model: Integrates Deep Autoencoders (AE) for feature extraction and denoising with Isolation Forest (iForest) for structural anomaly isolation within the latent space.
2. Multimodal Fusion & Cross-Validation: Alarms are triggered only if a temporal correlation of anomalies is detected across both pressure (PDG) and acoustic (DAS) modalities, significantly suppressing the False Alarm Rate (FAR).
3. Youden’s Index Optimization: Automatically determines optimal alarm thresholds to achieve the most effective balance between sensitivity and specificity.
4. TTD (Time-to-Detection) Tracking: Explicitly calculates the system's response time from the leak onset to the first valid detection.
5. Physical Realism: Incorporates simulations of water hammer effects, Joule-Thomson cooling, and broadband turbulent emissions, all calibrated to industrial sampling rates.
