"""
PRODIGY_GA_01: Text Generation with GPT-2
Author: Prodigy InfoTech Intern
Description: Text generation and fine-tuning pipeline using Hugging Face Transformers GPT-2.
"""

import argparse
import sys


def generate_text_gpt2(
    prompt: str,
    model_name: str = "gpt2",
    max_length: int = 100,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.95,
    num_return_sequences: int = 1
):
    """
    Generate text completion for a prompt using pre-trained GPT-2.
    """
    try:
        import torch
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
    except ImportError:
        print("[!] PyTorch or Transformers library not installed.")
        print("[!] Please run: pip install torch transformers")
        sys.exit(1)

    print(f"[+] Loading model '{model_name}' and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Set padding token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]

    print(f"[+] Generating text (max_length={max_length}, temp={temperature})...\n")
    
    with torch.no_grad():
        output_sequences = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.2,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id
        )

    results = []
    for i, seq in enumerate(output_sequences):
        decoded = tokenizer.decode(seq, skip_special_tokens=True)
        results.append(decoded)
        print(f"--- Sample {i+1} ---")
        print(decoded)
        print("-" * 30)

    return results


def fine_tune_gpt2_skeleton(dataset_path: str, output_dir: str = "./gpt2_finetuned"):
    """
    Fine-tuning workflow pipeline template for custom dataset training.
    """
    print(f"[+] Initializing Fine-Tuning Pipeline on '{dataset_path}'...")
    print(f"[+] Models will be saved to '{output_dir}'.")
    code_template = """
    # HuggingFace Fine-Tuning Recipe:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=dataset_path,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer.train()
    trainer.save_model(output_dir)
    """
    print(code_template)


def main():
    parser = argparse.ArgumentParser(description="PRODIGY_GA_01: GPT-2 Text Generation")
    parser.add_argument("--prompt", type=str, default="In the era of artificial intelligence, future technology will", help="Text prompt")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name (e.g. gpt2, gpt2-medium)")
    parser.add_argument("--max_length", type=int, default=80, help="Maximum token length")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    args = parser.parse_args()

    print("=" * 60)
    print("PRODIGY_GA_01: Text Generation with GPT-2")
    print("=" * 60)

    generate_text_gpt2(
        prompt=args.prompt,
        model_name=args.model,
        max_length=args.max_length,
        temperature=args.temp
    )


if __name__ == "__main__":
    main()
