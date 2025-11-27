# ai_image_generator
Text to image generation 
# 🎨 AI-Powered Image Generator (Stable Diffusion)

## 🌟 Project Overview

This project develops an open-source, text-to-image generation system using the Stable Diffusion v1.5 model. It is designed to run efficiently on GPU-accelerated environments like Google Colab or Hugging Face Spaces, providing a user-friendly web interface (Gradio) for converting text descriptions into high-quality, adjustable images.

The core objective is to provide a hands-on experience with diffusion models, deep learning frameworks, and practical prompt engineering techniques.
https://5890988915ecc5e594.gradio.live/
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/81d95fdc-1fa8-49d9-bb94-36b44c3d7c36" />


1. Project Overview and Architecture

The project is an AI-powered image generator built using Stable Diffusion v1.5. It is deployed on Hugging Face Spaces.

Architecture includes:

- Text encoder that converts prompts into embeddings
- UNet denoising model for removing noise in stages
- Variational autoencoder (VAE) for decoding hidden images
- Scheduler for controlling diffusion steps

A Gradio UI acts as the front end for user interaction.

Automated deployment through GitHub Actions ensures ongoing integration and updates.

2. Setup and Installation Steps (Including Model Download)

Clone the repository.

```
git clone https://github.com/<your-repo>.git
cd <your-repo>
```

Install dependencies.

```
pip install -r requirements.txt
```

Download Stable Diffusion v1.5.

Log in to Hugging Face.

Accept the model license.

Run:

```
huggingface-cli login
```

The model will download automatically at runtime through the diffusers pipeline.

Launch the app.

```
python app.py
```

3. Hardware Requirements (GPU/CPU Specifications)

Recommended:

- NVIDIA GPU with 8GB+ VRAM
- CUDA support for faster generation

Minimum (runs slowly):

- CPU-only system
- Hugging Face free-tier hardware

Optimizations:

- FP16 precision for quicker inference
- xFormers for memory-efficient attention

4. Usage Instructions with Example Prompts

Open the Gradio interface.

Enter:

- Prompt
- Optional negative prompt
- CFG scale
- Number of inference steps

Click generate image.

Example prompts:

“A cyberpunk city with neon lights, ultra-detailed, 4K”

“A watercolor painting of a quiet forest village”

“A realistic portrait of a robot reading a book”

5. Technology Stack and Model Details

Programming language: Python

Frameworks/libraries:

- Hugging Face diffusers
- PyTorch
- Gradio
- xFormers

Model used: Stable Diffusion v1.5

Deployment: Hugging Face Spaces and GitHub Actions (CI/CD)

Precision: FP16 for optimized generation

6. Prompt Engineering Tips and Best Practices

Use clear and descriptive prompts, such as lighting, style, atmosphere, and camera type.

Combine artistic keywords like:

“cinematic lighting”, “highly detailed”, “volumetric fog”

Add negative prompts to remove unwanted artifacts.

Ideal parameters include:

- CFG scale: 7 to 11
- Inference steps: 20 to 40

Avoid vague or generic prompts. Specificity improves output quality.

7. Limitations and Future Improvements

Limitations:

- Slower inference on CPU or low VRAM GPUs
- Free-tier deployments may restart or sleep
- High-resolution images require more memory
- Occasionally produces small artifacts

Future improvements:

- Fine-tuning on custom datasets
- Adding style transfer and LoRA-based personalization
- High-resolution upscaling using ESRGAN
- Batch image generation and image-to-image features
- Preset styles for quick image generation
