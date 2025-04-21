# 🧠 Grammar and Spelling Error Detection and Correction System

This is a multi-model **error correction system** that detects and corrects **both grammatical and spelling mistakes** using:

- ✅ Rule-based corrections
- ✅ N-gram predictions
- ✅ Transformer-based rewriting with T5 (Text-to-Text Transfer Transformer)

The system provides a GUI for users to input text and see corrected outputs using different models.

---

## 📌 Project Overview

This project implements an integrated correction system with three approaches:

### 1. 🧾 Rule-Based Correction
Uses custom **NLP grammar and spelling rules** to manually correct common mistakes such as:
- Subject-verb agreement (e.g., *"She do"* → *"She does"*)
- Auxiliary verb usage (e.g., *"He did went"* → *"He did go"*)
- Common spelling errors (e.g., *"recieve"* → *"receive"* using a spell checker)

### 2. 🔁 N-Gram Based Correction
Predicts the most probable word(s) based on surrounding context using a trained **N-gram language model**. Helps with:
- Collocation and fluency
- Fixing misplaced or missing words
- Context-aware spelling fixes

### 3. 🔄 Transformer-Based Correction (T5)
Uses a pre-trained **T5 model** from HuggingFace Transformers to **rephrase the sentence**, correcting both grammar and spelling holistically.

---


