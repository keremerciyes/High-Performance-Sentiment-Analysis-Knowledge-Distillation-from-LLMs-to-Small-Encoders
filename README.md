# High-Performance Sentiment Analysis: Knowledge Distillation from LLMs to Small Encoders

## Project Overview
This project explores the efficiency-accuracy trade-offs in deploying Natural Language Processing (NLP) models for sentiment analysis. We investigate **Knowledge Distillation (KD)** techniques to transfer the capabilities of Large Language Models (LLMs) and Large Encoders into compact, production-ready models.

The core objective is to take a massive, general-purpose model (like Mistral-7B or RoBERTa-Large) and compress its knowledge into a small, fast student model (DistilBERT) without significant loss of accuracy.

## Executive Summary of Results
*   **Efficiency:** Distilled models achieved a **2.89x speedup** over RoBERTa and a massive **128x speedup** over Mistral-7B.
*   **Performance:** The RoBERTa-distilled student retained **87% accuracy** while reducing the model size by **~20x**.
*   **The LLM Bottleneck:** While Mistral-7B achieves high accuracy (95%), its latency (522ms/sample) makes it unsuitable for real-time classification, validating the need for distillation.

## Installation & Requirements
The project is designed to run in a **Kaggle Notebook** environment (Dual T4 GPUs recommended) or a local environment with at least 16GB VRAM. Ensure you have the necessary libraries installed (PyTorch, Transformers, Datasets, Accelerate, PEFT, BitsAndBytes, Scikit-Learn, Pandas, Numpy).

## Project Structure & Scripts

### `01_baseline.py`
**Goal:** Establish a performance baseline.
*   Implements a **TF-IDF Vectorizer** feeding into a **Multinomial Naive Bayes** classifier.
*   **Purpose:** Serves as a "sanity check." If a deep learning model cannot significantly beat this in accuracy, the computational cost is not justified.

### `02_finetune_roberta.py`
**Goal:** Train a high-performance "Teacher" (Encoder).
*   Fine-tunes `roberta-large` (355M params) on the IMDB dataset.
*   **Output:** Saves the model and generates **Soft Labels (Logits)** to be used for distillation.

### `03_finetune_mistral.py`
**Goal:** Adapt a Generative LLM for Sentiment Analysis.
*   Fine-tunes `Mistral-7B-v0.1` using **QLoRA** (Quantized Low-Rank Adapters).
*   **Optimization:** Uses 4-bit quantization and Gradient Checkpointing to fit on T4 GPUs.
*   **Prompt Strategy:** Formats inputs as `[INST] Review... [/INST] Sentiment:`.

### `04a_distill_from_roberta.py` (White-Box Distillation)
**Goal:** Transfer knowledge from RoBERTa to DistilBERT.
*   **Method:** Uses a hybrid loss function:
    1.  **Logit Loss (KL Divergence):** Matches the probability distribution of the teacher.
    2.  **Feature Loss (MSE):** Aligns hidden states (using a projection layer to match 768-dim to 1024-dim).
    3.  **Hard Label Loss:** Standard Cross-Entropy against ground truth.

### `04b_distill_from_mistral.py` (Black-Box Distillation)
**Goal:** Transfer knowledge from Mistral to DistilBERT.
*   **Method:** Since architectures differ (Decoder vs Encoder), we cannot match hidden states. We use Mistral to generate probability scores for "positive" vs "negative" tokens and train DistilBERT to mimic these soft labels.

### `05_benchmark.py`
**Goal:** Rigorous comparison of all models.
*   Measures **Latency** (ms/sample), **Throughput** (samples/sec), **Model Size**, and **Accuracy**.
*   Includes aggressive memory cleaning to allow sequential testing on limited GPU memory.

## Experimental Results

| Model | Type | Accuracy | Latency (ms) | Throughput (samp/s) | Speedup (x) | Size (MB) | Params (M) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (TF-IDF)** | Traditional | **0.84** | 1.33 | 4135.32 | 12.84 | 28.80 | 0.00 |
| **Teacher (RoBERTa-L)** | Teacher | **0.96** | 17.11 | 18.41 | 1.00 | 5432.01 | 355.36 |
| **Student (from RoBERTa)** | Student | **0.87** | 5.91 | 103.00 | **2.89** | 256.33 | 66.96 |
| **Student (from Mistral)** | Student | **0.51** | 4.07 | 115.05 | **4.21** | 1022.69 | 66.96 |
| **Teacher (Mistral-7B)** | Teacher | **0.95** | 522.36 | 1.15 | 0.03 | 15000.00 | 7000.00 |

## Discussion & Analysis

### 1. The Success of Encoder Distillation
The **Student distilled from RoBERTa** was the most balanced model. It achieved a **2.89x speedup** over its teacher while maintaining **87% accuracy**. This proves that "White-Box" distillation (access to internal weights/features) is highly effective. The drop from 96% to 87% suggests that further hyperparameter tuning (specifically the weight of the Feature Loss) could close the gap further.

### 2. The Challenge of Generative Distillation
The **Student distilled from Mistral** failed to converge (0.51 accuracy is equivalent to random guessing). This highlights the difficulty of "Black-Box" distillation.
*   **Cause:** The mismatch between the generative tokenizer (Mistral) and the classifier tokenizer (DistilBERT) likely caused alignment issues when extracting logits for specific words like "positive".
*   **Insight:** While LLMs are powerful teachers, extracting their knowledge into a classifier requires precise calibration of the "verbalizer" (mapping words to class labels).

### 3. The Cost of Generative AI
The benchmark reveals a critical engineering reality: **Mistral-7B is ~30x slower than RoBERTa-Large**.
While Mistral achieved high accuracy (95%), its throughput (1.15 samples/sec) is too low for high-volume batch processing. This confirms that for specific, narrow tasks like Sentiment Analysis, **Encoder models (RoBERTa/BERT) remain the superior engineering choice** over Generative LLMs regarding cost and latency.

### 4. An Addition for the Baseline 

The TF-IDF baseline achieved **84% accuracy** with negligible latency (1.33ms). This serves as a reminder that for many business use cases, simple statistical methods are "good enough" and vastly cheaper than Deep Learning solutions.

