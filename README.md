# ai_image_generator
Text to image generation 
# 🎨 AI-Powered Image Generator (Stable Diffusion)

## 🌟 Project Overview

This project develops an open-source, text-to-image generation system using the Stable Diffusion v1.5 model. It is designed to run efficiently on GPU-accelerated environments like Google Colab or Hugging Face Spaces, providing a user-friendly web interface (Gradio) for converting text descriptions into high-quality, adjustable images.

The core objective is to provide a hands-on experience with diffusion models, deep learning frameworks, and practical prompt engineering techniques.
https://www.google.com/search?q=https://nandinnandu-ai-image-generator.hf.space/<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/81d95fdc-1fa8-49d9-bb94-36b44c3d7c36" />

### Architecture

The application follows a standard ML demo architecture:
**User Interface:** Gradio web interface.
**Model Backend:** PyTorch framework leveraging the `diffusers` library.
**Model:** Stable Diffusion v1.5 (runwayml/stable-diffusion-v1-5).
**Deployment:** Continuous integration via **GitHub Actions** syncing to **Hugging Face Spaces**.

---

## 💻 Setup and Installation

This project is designed for deployment on Google Colab or Hugging Face Spaces.

### 1. Hardware Requirements

| Configuration | Recommended Hardware | Generation Time (Approx.) | Note |
| :--- | :--- | :--- | :--- |
| **Preferred (GPU)** | NVIDIA Tesla T4 (Colab Free Tier) | 10 - 30 seconds / image | Uses FP16 and `xformers` optimization. |
| **Fallback (CPU)** | 2+ vCPU, 16GB RAM (HF Free Tier) | 5 - 10 minutes / image | Extremely slow; intended for demonstration only. |

### 2. Dependencies

The following packages are required and are listed in the `requirements.txt` file:

```bash
torch
diffusers
transformers
accelerate
gradio
ftfy
bitsandbytes
