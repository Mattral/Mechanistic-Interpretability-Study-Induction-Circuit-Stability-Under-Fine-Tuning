# Induction Circuit Stability: A Mechanistic Interpretability Study

[![CI Status](https://img.shields.io/badge/CI-failing-red?style=flat-square)](#-current-status--limitations)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](pyproject.toml)
[![Status](https://img.shields.io/badge/Status-Active_Research-yellow?style=flat-square)](#-current-status--limitations)

> **TL;DR:** We are tracking what happens to the "brain wiring" (induction heads) of a simple Transformer when you force it to learn Python code. 


> **A mechanistic interpretability study tracking the survival and evolution of induction heads in a 2-layer attention-only transformer during fine-tuning.**


## 🧠 What is this?

When language models learn, they develop specialized circuits (like **induction heads**, which help them copy patterns). But what happens to these fragile structures when we fine-tune the model on a completely new domain? Do they break? Do they adapt? 

This repository contains a reproducible mechanistic interpretability study that maps the exact training steps where a 2-layer attention-only transformer alters its internal circuitry while being fine-tuned on Python.

*If you are interested in AI safety, alignment, or how neural networks actually "think" under the hood, this project is for you.*

---

## 📖 Overview

Do structural circuits learned during pre-training survive domain adaptation? 

This repository houses a rigorous mechanistic interpretability study that investigates how **induction head circuits** adapt when a 2-layer attention-only transformer is fine-tuned on Python code. By meticulously tracking the network across exact training steps, we aim to map the precise moments structural changes and circuit formations occur.

### 🎯 Key Research Questions
1. Do pre-existing induction heads degrade, adapt, or remain static when exposed to a new, highly structured domain (Python)?
2. At what exact training step do noticeable structural phase changes occur in the attention patterns?
3. Can we mathematically map the intermediate circuitry during the fine-tuning transition?

---

## ⚠️ Current Status & Honest Limitations

*Transparency is our core value. Here is exactly where the project stands today:*

- **The Good:** Our core baseline models, config systems (`experiments/configs/`), and visualization pipelines (`src/`) are built. The repository is architected for strict reproducibility, complete with an Architecture Decision Record (`decisions/`).
- **The Bad (Current Blockers):** As indicated by our CI badge, our automated testing pipeline is currently failing. We are actively debugging edge cases in our circuit-patching tests.
- **The Limitations:** This study is strictly limited to a **2-layer attention-only model**. Our findings *may not* scale linearly to massive MLPs in frontier models like Llama-3 or GPT-4. We are starting small to ensure mathematical rigor.

---

## 🚀 Quickstart

We use `environment.yml` and `pyproject.toml` to ensure you can replicate our environment exactly.

```bash
# 1. Clone the repo
git clone https://github.com/Mattral/Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning.git
cd Mechanistic-Interpretability-Study-Induction-Circuit-Stability-Under-Fine-Tuning

# 2. Set up the exact environment
conda env create -f environment.yml
conda activate mech-interp

# 3. Install the source package locally
pip install -e .

```

*Head over to the `notebooks/` directory to see our interactive replication scripts.*

---

## 🗺️ Repo Tour

We hate messy research code. This repository is structured to be read like a book:

* 📂 **`/paper`** — The actual LaTeX drafts and references. Read this for the theory.
* 📂 **`/decisions`** — Our research diary. Every major methodological pivot is documented here.
* 📂 **`/experiments`** — The raw configs. You can reproduce any of our fine-tuning runs using these files.
* 📂 **`/notebooks`** — Visual dashboards and scratchpads. Start here if you want to see pretty attention graphs.
* 📂 **`/src`** — The core, tested interpretability tools doing the heavy lifting.

---

## 🤝 How to Get Involved

This is an open science project.

1. **⭐ Star the repo** if you want to follow along with our findings (it helps us know people care about this niche!).
2. **🐛 Open an Issue** if you have ideas on how to fix our failing CI or improve our patching methodology.
3. **📖 Read the Docs** in our `notebooks` and share your thoughts.

As this is an ongoing research study, the primary goal of this repository is transparency and reproducibility. If you spot an error in our circuit analysis or have suggestions for the fine-tuning methodology, please open an Issue or review our decisions/ log.

---

**Authors:** [Mattral](https://github.com/Mattral) | **License:** Apache 2.0

```
