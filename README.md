# 🏷️ Shallow Parsing and Named Entity Recognition using BiLSTM-CRF

This repository contains the implementation of a **Bidirectional Long Short-Term Memory (BiLSTM) network combined with a Conditional Random Fields (CRF) layer** for the joint tasks of Shallow Parsing (Chunking) and Named Entity Recognition (NER).

The project is structured as a mini-project focusing on sequence labeling tasks, utilizing deep learning to extract structured information from raw text.

## 🚀 Project Overview

The core objective is to process English sentences and output two types of labels for each word:
1.  **Named Entities (NER):** Identifying real-world entities (e.g., PERSON, ORGANIZATION, LOCATION).
2.  **Phrase Boundaries (Shallow Parsing):** Identifying grammatical structures (e.g., Noun Phrase - NP, Verb Phrase - VP).

### Architecture Flow

The model follows a standard sequence labeling architecture:

**Input Text** → **Preprocessing/Embeddings** → **BiLSTM Layers** (Contextual Encoding) → **CRF Layer** (Optimal Sequence Tagging) → **Tagged Output**

### Dataset

* **Corpus:** CoNLL-2003 Named Entity Recognition Dataset (W-P-T-C format).
* **Size:** Approximately 300,000 tokens and 20,000 sentences.

## ⚙️ Setup and Installation

### Prerequisites

You need Python 3.6+ and the following libraries.

```bash
pip install -r requirements.txt