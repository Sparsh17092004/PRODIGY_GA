"""
PRODIGY_GA_02: Image Generation with Pre-trained Models
Author: Prodigy InfoTech Intern
Description: Text-to-Image Generation pipeline using Stable Diffusion / HuggingFace Diffusers.
"""

import argparse
import sys
import os


def generate_image_sd(
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, ugly",
    output_path: str = "generated_output.png",
    model_id: str = "runwayml/stable-diffusion-v1-5",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    seed: int = 42
):
    """
    Generate an image from a textual prompt using Stable Diffusion pipeline.
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print("[!] Diffusers or PyTorch library not installed.")
        print("[!] Please run: pip install torch diffusers transformers accelerations")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Using device: {device.upper()}")
    print(f"[+] Loading model pipeline '{model_id}'...")

    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)

    # Set random generator seed
    generator = torch.Generator(device=device).manual_seed(seed) if seed else None

    print(f"[+] Generating image for prompt: '{prompt}'...")
    print(f"[+] Steps: {num_inference_steps}, Guidance: {guidance_scale}")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=generator
    ).images[0]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    image.save(output_path)
    print(f"[+] Success! Image saved to: {os.path.abspath(output_path)}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="PRODIGY_GA_02: Image Generation with Stable Diffusion")
    parser.add_argument("--prompt", type=str, default="A futuristic cyberpunk city with neon lights at night, digital art", help="Text prompt")
    parser.add_argument("--output", type=str, default="cyberpunk_city.png", help="Output file path")
    parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print("=" * 60)
    print("PRODIGY_GA_02: Image Generation with Pre-trained Models")
    print("=" * 60)

    print(f"\n[+] Task: Text-to-Image Generation")
    print(f"[+] Prompt: {args.prompt}")
    print(f"[+] Output File: {args.output}\n")

    generate_image_sd(
        prompt=args.prompt,
        output_path=args.output,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
