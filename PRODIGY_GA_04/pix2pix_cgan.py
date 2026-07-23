"""
PRODIGY_GA_04: Image-to-Image Translation with cGAN (Pix2Pix)
Author: Prodigy InfoTech Intern
Description: Conditional GAN (Pix2Pix) architecture with U-Net Generator and PatchGAN Discriminator.
"""

import argparse
import sys
import os


def build_pix2pix_model():
    """
    Construct Pix2Pix Generator and Discriminator neural network architectures in PyTorch.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("[!] PyTorch library not installed. Please run: pip install torch torchvision")
        sys.exit(1)

    class UNetBlock(nn.Module):
        """U-Net Encoder/Decoder block with skip connections."""
        def __init__(self, in_channels, out_channels, submodule=None, outermost=False, innermost=False, dropout=False):
            super().__init__()
            self.outermost = outermost

            downconv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            downrelu = nn.LeakyReLU(0.2, True)
            downnorm = nn.BatchNorm2d(out_channels)
            uprelu = nn.ReLU(True)
            upnorm = nn.BatchNorm2d(in_channels)

            if outermost:
                upconv = nn.ConvTranspose2d(out_channels * 2, in_channels, kernel_size=4, stride=2, padding=1)
                down = [downconv]
                up = [uprelu, upconv, nn.Tanh()]
                model = down + [submodule] + up
            elif innermost:
                upconv = nn.ConvTranspose2d(out_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
                down = [downrelu, downconv]
                up = [uprelu, upconv, upnorm]
                model = down + up
            else:
                upconv = nn.ConvTranspose2d(out_channels * 2, in_channels, kernel_size=4, stride=2, padding=1, bias=False)
                down = [downrelu, downconv, downnorm]
                up = [uprelu, upconv, upnorm]
                if dropout:
                    up.append(nn.Dropout(0.5))
                model = down + [submodule] + up

            self.model = nn.Sequential(*model)

        def forward(self, x):
            if self.outermost:
                return self.model(x)
            else:
                return torch.cat([x, self.model(x)], 1)

    class UNetGenerator(nn.Module):
        """U-Net Generator architecture."""
        def __init__(self, in_channels=3, out_channels=3, num_downs=7, ngf=64):
            super().__init__()
            # Construct U-Net recursively
            unet_block = UNetBlock(ngf * 8, ngf * 8, innermost=True)
            for _ in range(num_downs - 5):
                unet_block = UNetBlock(ngf * 8, ngf * 8, submodule=unet_block, dropout=True)
            unet_block = UNetBlock(ngf * 4, ngf * 8, submodule=unet_block)
            unet_block = UNetBlock(ngf * 2, ngf * 4, submodule=unet_block)
            unet_block = UNetBlock(ngf, ngf * 2, submodule=unet_block)
            self.model = UNetBlock(in_channels, ngf, submodule=unet_block, outermost=True)

        def forward(self, input_img):
            return self.model(input_img)

    class PatchGANDiscriminator(nn.Module):
        """70x70 PatchGAN Discriminator architecture."""
        def __init__(self, in_channels=6, ndf=64):
            super().__init__()
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)
            )

        def forward(self, input_img, target_img):
            # Concatenate input and target along channels
            x = torch.cat([input_img, target_img], dim=1)
            return self.model(x)

    return UNetGenerator, PatchGANDiscriminator


def run_demo():
    import torch
    import torch.nn as nn

    UNetGenerator, PatchGANDiscriminator = build_pix2pix_model()

    print("[+] Instantiating Pix2Pix Generator and Discriminator...")
    generator = UNetGenerator(in_channels=3, out_channels=3)
    discriminator = PatchGANDiscriminator(in_channels=6)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator.to(device)
    discriminator.to(device)

    # Synthetic batch of images (Batch size=2, Channels=3, H=256, W=256)
    dummy_source = torch.randn(2, 3, 256, 256).to(device)
    dummy_target = torch.randn(2, 3, 256, 256).to(device)

    print(f"[+] Device: {device.upper()}")
    print(f"[+] Dummy Input Batch Tensor Shape:  {list(dummy_source.shape)}")

    # Forward pass Generator
    fake_target = generator(dummy_source)
    print(f"[+] Generator Output Tensor Shape:  {list(fake_target.shape)}")

    # Forward pass Discriminator (Real pair vs Fake pair)
    pred_real = discriminator(dummy_source, dummy_target)
    pred_fake = discriminator(dummy_source, fake_target)
    print(f"[+] PatchGAN Real Pred Tensor Shape: {list(pred_real.shape)}")
    print(f"[+] PatchGAN Fake Pred Tensor Shape: {list(pred_fake.shape)}")

    # Loss computation demonstration
    criterion_GAN = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
    loss_L1 = criterion_L1(fake_target, dummy_target)
    lambda_L1 = 100.0
    total_g_loss = loss_GAN + lambda_L1 * loss_L1

    print(f"\n[+] Loss Computations:")
    print(f"    - Adversarial GAN Loss: {loss_GAN.item():.4f}")
    print(f"    - L1 Reconstruction Loss: {loss_L1.item():.4f}")
    print(f"    - Total Generator Loss: {total_g_loss.item():.4f}")
    print("\n[+] Pix2Pix cGAN Architecture test completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="PRODIGY_GA_04: Pix2Pix cGAN Image-to-Image Translation")
    parser.add_argument("--demo", action="store_true", default=True, help="Run architecture verification demo")
    args = parser.parse_args()

    print("=" * 60)
    print("PRODIGY_GA_04: Image-to-Image Translation with cGAN (Pix2Pix)")
    print("=" * 60 + "\n")

    run_demo()


if __name__ == "__main__":
    main()
