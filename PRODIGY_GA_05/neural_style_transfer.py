"""
PRODIGY_GA_05: Neural Style Transfer
Author: Prodigy InfoTech Intern
Description: PyTorch implementation of Gatys et al. Neural Style Transfer using pre-trained VGG-19.
"""

import argparse
import sys
import os


def gram_matrix(tensor):
    """
    Compute Gram Matrix to measure correlation between feature map channels.
    G = F * F^T
    """
    b, c, h, w = tensor.size()
    features = tensor.view(b * c, h * w)
    gram = torch.mm(features, features.t())
    return gram.div(b * c * h * w)


def run_neural_style_transfer(
    content_img_path: str = None,
    style_img_path: str = None,
    output_path: str = "stylized_output.png",
    num_steps: int = 150,
    style_weight: float = 1e6,
    content_weight: float = 1.0
):
    """
    Perform Neural Style Transfer optimization loop using VGG19.
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import models, transforms
        from PIL import Image
    except ImportError:
        print("[!] PyTorch / Torchvision / PIL library not installed.")
        print("[!] Please run: pip install torch torchvision pillow")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Device selected: {device.upper()}")

    # Image transformations
    imsize = 256
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    unloader = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
    ])

    # Load or generate content & style tensors
    if content_img_path and os.path.exists(content_img_path):
        content_img = Image.open(content_img_path).convert("RGB")
        content_tensor = loader(content_img).unsqueeze(0).to(device)
    else:
        print("[+] Content image not provided/found. Generating synthetic content tensor...")
        content_tensor = torch.rand(1, 3, imsize, imsize).to(device)

    if style_img_path and os.path.exists(style_img_path):
        style_img = Image.open(style_img_path).convert("RGB")
        style_tensor = loader(style_img).unsqueeze(0).to(device)
    else:
        print("[+] Style image not provided/found. Generating synthetic style tensor...")
        style_tensor = torch.rand(1, 3, imsize, imsize).to(device)

    # Output image initialized to copy of content image
    target_tensor = content_tensor.clone().requires_grad_(True).to(device)

    # Load pre-trained VGG-19
    print("[+] Loading pre-trained VGG-19 feature network...")
    vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

    # Freeze VGG parameters
    for param in vgg.parameters():
        param.requires_grad = False

    # Define content & style layers
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_features(image, model):
        features = {}
        x = image
        i = 0
        for name, layer in model._modules.items():
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                i += 1
                layer_name = f"conv_{i}"
                if layer_name in content_layers or layer_name in style_layers:
                    features[layer_name] = x
        return features

    # Extract target features and style gram matrices
    content_features = get_features(content_tensor, vgg)
    style_features = get_features(style_tensor, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    optimizer = optim.Adam([target_tensor], lr=0.03)

    print(f"[+] Starting optimization for {num_steps} iterations...")
    for step in range(1, num_steps + 1):
        target_features = get_features(target_tensor, vgg)

        # Compute content loss
        c_loss = torch.mean((target_features['conv_4'] - content_features['conv_4']) ** 2)

        # Compute style loss across layers
        s_loss = 0
        for layer in style_layers:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            layer_style_loss = torch.mean((target_gram - style_gram) ** 2)
            s_loss += layer_style_loss / (target_features[layer].shape[1] ** 2)

        total_loss = content_weight * c_loss + style_weight * s_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 30 == 0 or step == num_steps:
            print(f"    Step [{step}/{num_steps}] | Total Loss: {total_loss.item():.4f} | Content: {c_loss.item():.4f} | Style: {s_loss.item():.4f}")

    print(f"[+] Neural Style Transfer completed!")

    # Save output image
    final_img = target_tensor.cpu().clone().squeeze(0)
    final_img = torch.clamp(final_img, 0, 1)
    pil_img = transforms.ToPILImage()(final_img)
    pil_img.save(output_path)
    print(f"[+] Saved output image to: {os.path.abspath(output_path)}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="PRODIGY_GA_05: Neural Style Transfer")
    parser.add_argument("--content", type=str, default=None, help="Path to content image")
    parser.add_argument("--style", type=str, default=None, help="Path to style image")
    parser.add_argument("--output", type=str, default="stylized_output.png", help="Output file path")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    args = parser.parse_args()

    print("=" * 60)
    print("PRODIGY_GA_05: Neural Style Transfer with VGG-19")
    print("=" * 60 + "\n")

    run_neural_style_transfer(
        content_img_path=args.content,
        style_img_path=args.style,
        output_path=args.output,
        num_steps=args.steps
    )


if __name__ == "__main__":
    main()
