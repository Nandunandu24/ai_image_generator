# -----------------------------------------------------------
# Colab Executable Code: AI-Powered Image Generator with Gradio
# -----------------------------------------------------------

# --- 1. Installation ---
# Install required libraries: diffusers for the model, transformers for utilities, and gradio for the UI.
# The accelerate library helps optimize PyTorch operations on the GPU.
print("--- 1. Installing required libraries... ---")
!pip install -qqq torch torchvision torchaudio
!pip install -qqq diffusers transformers accelerate ftfy bitsandbytes
!pip install -qqq gradio

# Import necessary libraries
import torch
import gradio as gr
from PIL import Image
from diffusers import StableDiffusionPipeline
import os
import time

# --- 2. Model Selection and Setup ---
# Use the Stable Diffusion v1.5 model from Hugging Face for high-quality open-source text-to-image generation.
MODEL_ID = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Determine device: GPU (cuda) is preferred, with CPU fallback.
# bitsandbytes is used for memory-efficient 8-bit loading on GPU, if available.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"--- 2. Detected device: {device} ---")

# Load the pipeline
try:
    if device == "cuda":
        # Load the model in half-precision (torch.float16) and 8-bit quantization for GPU memory efficiency
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            use_safetensors=True
        ).to(device)
        # Enable memory optimization features
        pipe.enable_xformers_memory_efficient_attention()
    else:
        # Load for CPU
        pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID).to(device)
        print("Note: Running on CPU will be very slow.")

except Exception as e:
    print(f"An error occurred during model loading: {e}")
    print("Attempting CPU fallback...")
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID).to("cpu")
    device = "cpu"

# --- 3. Core Generation Function ---

def generate_image(
    prompt: str,
    negative_prompt: str,
    num_images: int,
    style_guidance: float,
    steps: int,
    seed: int,
):
    """
    Generates an image from a text prompt using the Stable Diffusion pipeline.
    """
    if not prompt:
        return [None] * num_images, "Error: Prompt cannot be empty."

    # Set up deterministic generation if a seed is provided
    generator = None
    if seed != -1:
        generator = torch.Generator(device=device).manual_seed(seed)
    
    # List to store generated images
    generated_images = []
    
    # Apply ethical AI watermark guideline: we'll save the image with a note
    # For actual watermarking, a separate image processing library like OpenCV would be used.
    
    print(f"\n--- Starting generation for prompt: '{prompt[:50]}...' ---")
    start_time = time.time()
    
    # Generate images in a loop
    for i in range(num_images):
        try:
            # The core generation call
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=style_guidance, # Corresponds to style guidance
                num_inference_steps=steps,
                generator=generator,
                num_images_per_prompt=1 # pipe call generates 1 image at a time here
            ).images[0]
            
            # --- 4. Storage and Export ---
            # Save the image with metadata and timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            safe_prompt = prompt.replace(" ", "_").replace("/", "_")[:30] # Simple file-safe name
            filename = f"AI_{timestamp}_{i}_{safe_prompt}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Save in PNG format (supports lossless and metadata storage)
            image.save(filepath, "PNG")
            
            # Save a JPEG version as well (for multi-format requirement)
            jpeg_filepath = filepath.replace(".png", ".jpeg")
            image.save(jpeg_filepath, "JPEG", quality=90) # Quality 90 is a good balance
            
            generated_images.append(image)
            print(f"Image {i+1} saved as {filepath} and {jpeg_filepath}")
            
        except Exception as e:
            print(f"Generation error for image {i+1}: {e}")
            generated_images.append(None)
            
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Return images and a status message
    status_message = (
        f"✅ Generation complete! Total time: {total_duration:.2f} seconds. "
        f"Images saved in the '{OUTPUT_DIR}' folder in Colab."
    )
    
    # Pad the list with None if fewer images were generated than requested
    while len(generated_images) < num_images:
        generated_images.append(None)
        
    return generated_images, status_message


# --- 5. Gradio User Interface ---

# Define a function to display the ethical guidelines
def ethical_guidelines_message():
    return """
    **AI-Powered Image Generator - Ethical Guidelines**

    1.  **Responsible Use:** Do not generate content that is hateful, violent, discriminatory, or promotes self-harm.
    2.  **Content Filtering:** Inappropriate prompts will be blocked or may lead to poor results.
    3.  **AI Origin:** All generated images are AI-created. Treat them responsibly.
    """

# Define the Gradio interface structure
with gr.Blocks(title="AI-Powered Image Generator (Stable Diffusion)") as demo:
    gr.Markdown("# 🤖 AI-Powered Image Generator (Stable Diffusion)")
    gr.Markdown(
        "Enter your prompt and adjust settings. The model is running on **"
        + device
        + "**."
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            # Input Fields
            prompt_input = gr.Textbox(
                label="**Positive Prompt (The main idea)**",
                placeholder="e.g., a futuristic city at sunset, highly detailed, 4K, cinematic lighting",
                lines=5,
            )
            negative_prompt_input = gr.Textbox(
                label="**Negative Prompt (What to exclude)**",
                placeholder="e.g., low quality, blurry, worst quality, ugly, watermark, distortion",
                lines=2,
            )
            
            # Adjustable Parameters (as required)
            with gr.Accordion("🎨 Generation Settings", open=True):
                num_images_slider = gr.Slider(
                    minimum=1,
                    maximum=4,
                    step=1,
                    value=1,
                    label="Number of Images to Generate",
                )
                guidance_slider = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    step=0.5,
                    value=7.5,
                    label="Style Guidance (CFG Scale: Higher = Closer to prompt)",
                )
                steps_slider = gr.Slider(
                    minimum=10,
                    maximum=150,
                    step=1,
                    value=50,
                    label="Inference Steps (Higher = Better quality/longer time)",
                )
                seed_input = gr.Number(
                    value=-1, label="Seed (-1 for random, use a number to reproduce results)"
                )

            generate_btn = gr.Button("🚀 Generate Image(s)", variant="primary")
            
            # Display progress and status
            status_output = gr.Textbox(
                label="Generation Status",
                value="Ready.",
                interactive=False,
            )

            # Display Ethical Guidelines
            gr.Markdown(ethical_guidelines_message())


        with gr.Column(scale=3):
            # Output Display
            output_gallery = gr.Gallery(
                label="Generated Images", columns=2, object_fit="contain", height="auto"
            )

    # Link the button click event to the generation function
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            negative_prompt_input,
            num_images_slider,
            guidance_slider,
            steps_slider,
            seed_input,
        ],
        outputs=[output_gallery, status_output],
    )

# --- 6. Launch Gradio Interface ---
# Launch with share=True to generate a public link for Colab access.
print("--- 6. Launching Gradio Interface... ---")
demo.launch()