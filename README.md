# Prodigy InfoTech Internship — Generative AI Track (`PRODIGY_GA`)

This repository contains the complete implementation for all 5 tasks under the **Generative AI (GA)** internship track provided by Prodigy InfoTech.

---

## 📁 Repository Structure

```text
prodigy-genai-tasks/
├── PRODIGY_GA_01/
│   └── text_generation_gpt2.py         # Task 01: GPT-2 Text Generation & Fine-Tuning
├── PRODIGY_GA_02/
│   └── image_generation_sd.py          # Task 02: Text-to-Image Generation with Stable Diffusion
├── PRODIGY_GA_03/
│   └── markov_chain_generator.py       # Task 03: Statistical N-Gram Markov Chain Text Generator
├── PRODIGY_GA_04/
│   └── pix2pix_cgan.py                 # Task 04: Image-to-Image Translation with Pix2Pix cGAN
├── PRODIGY_GA_05/
│   └── neural_style_transfer.py        # Task 05: Gatys et al. Neural Style Transfer using VGG-19
├── requirements.txt                    # Project Dependencies
└── README.md                           # Documentation
```

---

## 🚀 Tasks Overview & Execution Instructions

### 1. `PRODIGY_GA_01`: Text Generation with GPT-2
- **Description**: fine-tunes / generates contextually relevant text using Hugging Face `transformers` and `GPT2LMHeadModel`.
- **Run Command**:
  ```bash
  python PRODIGY_GA_01/text_generation_gpt2.py --prompt "Artificial intelligence in modern software development" --max_length 100
  ```

### 2. `PRODIGY_GA_02`: Image Generation with Pre-trained Models
- **Description**: Generates high-quality images from text prompts using Stable Diffusion pipelines (`diffusers`).
- **Run Command**:
  ```bash
  python PRODIGY_GA_02/image_generation_sd.py --prompt "A majestic golden castle on a floating island in space" --output castle.png
  ```

### 3. `PRODIGY_GA_03`: Text Generation with Markov Chains
- **Description**: Pure Python n-gram statistical model that calculates transition probability matrices over text corpora.
- **Run Command**:
  ```bash
  python PRODIGY_GA_03/markov_chain_generator.py --prompt "Generative Artificial Intelligence" --n 2 --length 50
  ```

### 4. `PRODIGY_GA_04`: Image-to-Image Translation with cGAN (Pix2Pix)
- **Description**: PyTorch implementation of Conditional GAN with U-Net Generator and PatchGAN Discriminator.
- **Run Command**:
  ```bash
  python PRODIGY_GA_04/pix2pix_cgan.py --demo
  ```

### 5. `PRODIGY_GA_05`: Neural Style Transfer
- **Description**: Transfers artistic style from a style image onto a content image using pre-trained VGG-19 Gram matrices.
- **Run Command**:
  ```bash
  python PRODIGY_GA_05/neural_style_transfer.py --steps 100 --output stylized_artwork.png
  ```

---

## 📌 Submission & GitHub Repository Naming Rules
Per Prodigy InfoTech internship instructions:
- Each completed task should be pushed to a public GitHub repository named:
  - Task 1: `PRODIGY_GA_01`
  - Task 2: `PRODIGY_GA_02`
  - Task 3: `PRODIGY_GA_03`
  - Task 4: `PRODIGY_GA_04`
  - Task 5: `PRODIGY_GA_05`
- Share your work on LinkedIn upon task completion, tagging **Prodigy InfoTech** and describing your learnings.

---
*Created as part of the Prodigy InfoTech Generative AI Internship Program.*
