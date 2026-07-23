"""
Prodigy InfoTech Generative AI Track — Interactive Master Demo Launcher
Run this script to test and see any of the 5 tasks in action!
"""

import os
import sys
import subprocess

TASKS = {
    "1": ("PRODIGY_GA_01", "Text Generation with GPT-2", "PRODIGY_GA_01/text_generation_gpt2.py"),
    "2": ("PRODIGY_GA_02", "Image Generation with Stable Diffusion", "PRODIGY_GA_02/image_generation_sd.py"),
    "3": ("PRODIGY_GA_03", "Text Generation with Markov Chains", "PRODIGY_GA_03/markov_chain_generator.py"),
    "4": ("PRODIGY_GA_04", "Image-to-Image Translation with cGAN (Pix2Pix)", "PRODIGY_GA_04/pix2pix_cgan.py"),
    "5": ("PRODIGY_GA_05", "Neural Style Transfer (VGG-19)", "PRODIGY_GA_05/neural_style_transfer.py")
}

def display_menu():
    print("\n" + "=" * 65)
    print(" 🚀 PRODIGY INFOTECH GENERATIVE AI TRACK — DEMO LAUNCHER ")
    print("=" * 65)
    for key, (code, name, path) in TASKS.items():
        print(f" [{key}] {code} — {name}")
    print(" [A] Run All Tasks")
    print(" [Q] Quit")
    print("=" * 65)

def run_task(key):
    if key in TASKS:
        code, name, script_path = TASKS[key]
        print(f"\n[+] Running {code} ({name})...\n")
        cmd = [sys.executable, script_path]
        subprocess.run(cmd, check=False)
    elif key.upper() == "A":
        for k in ["3", "4", "5", "1", "2"]:
            run_task(k)
    else:
        print("[!] Invalid option.")

def main():
    if len(sys.argv) > 1:
        run_task(sys.argv[1])
        return

    while True:
        display_menu()
        choice = input("Enter choice (1-5, A, Q): ").strip()
        if choice.upper() == "Q":
            print("\n[+] Exiting Demo Launcher. Goodbye!")
            break
        run_task(choice)

if __name__ == "__main__":
    main()
