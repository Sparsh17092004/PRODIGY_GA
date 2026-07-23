"""
PRODIGY_GA_03: Text Generation with Markov Chains
Author: Prodigy InfoTech Intern
Description: A statistical n-gram Markov Chain model for text generation based on transition probabilities.
"""

import random
import re
import argparse
from collections import defaultdict
from typing import List, Dict, Tuple


class MarkovChainTextGenerator:
    """Statistical N-gram Markov Chain text generator."""

    def __init__(self, n: int = 2):
        """
        Initialize Markov Chain model.
        
        :param n: Order of the Markov Chain (number of preceding words used as state key).
        """
        self.n = n
        self.transitions: Dict[Tuple[str, ...], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.starts: List[Tuple[str, ...]] = []

    def tokenize(self, text: str) -> List[str]:
        """Clean and tokenize input text into words."""
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", text)
        return tokens

    def train(self, corpus: str) -> None:
        """
        Build state transition matrix from text corpus.
        
        :param corpus: Input text dataset for learning probability distribution.
        """
        tokens = self.tokenize(corpus)
        if len(tokens) <= self.n:
            raise ValueError(f"Corpus must contain at least {self.n + 1} tokens.")

        for i in range(len(tokens) - self.n):
            state = tuple(tokens[i:i + self.n])
            next_token = tokens[i + self.n]

            self.transitions[state][next_token] += 1
            if i == 0 or tokens[i - 1] in ['.', '!', '?']:
                self.starts.append(state)

    def generate(self, prompt: str = "", max_tokens: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text starting from optional prompt or random start state.
        
        :param prompt: Initial text prompt to start generation.
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Softmax temperature parameter (higher = more creative/random).
        :return: Generated text string.
        """
        if not self.transitions:
            return "Model has not been trained on any corpus."

        tokens = self.tokenize(prompt) if prompt else []

        if len(tokens) >= self.n:
            current_state = tuple(tokens[-self.n:])
        elif self.starts:
            current_state = random.choice(self.starts)
            if not prompt:
                tokens = list(current_state)
        else:
            current_state = random.choice(list(self.transitions.keys()))
            if not prompt:
                tokens = list(current_state)

        for _ in range(max_tokens):
            if current_state not in self.transitions:
                # Fallback to random state if dead end reached
                current_state = random.choice(list(self.transitions.keys()))

            candidates = self.transitions[current_state]
            words = list(candidates.keys())
            counts = list(candidates.values())

            # Apply temperature scaling
            if temperature != 1.0 and temperature > 0:
                weights = [c ** (1.0 / temperature) for c in counts]
            else:
                weights = counts

            next_token = random.choices(words, weights=weights, k=1)[0]
            tokens.append(next_token)
            current_state = tuple(tokens[-self.n:])

        # Reconstruct readable string
        generated_text = ""
        for i, token in enumerate(tokens):
            if i > 0 and token not in ['.', ',', '!', '?', ';', ':', "'"]:
                generated_text += " "
            generated_text += token

        return generated_text


DEMO_CORPUS = """
Generative Artificial Intelligence is transforming technology and society. Machine learning models learn patterns 
from vast data to create original content. Neural networks process complex information and produce text, images, 
and music. Deep learning algorithms continue to advance rapidly. Artificial intelligence systems assist humans 
in solving challenging problems across medicine, engineering, science, and creative arts. The future of AI 
depends on responsible development, ethical considerations, and innovative software engineering. Machine learning 
empowers developers to solve complex real-world problems with data-driven models.
"""


def main():
    parser = argparse.ArgumentParser(description="PRODIGY_GA_03: Markov Chain Text Generator")
    parser.add_argument("--prompt", type=str, default="Artificial intelligence", help="Seed prompt for text generation")
    parser.add_argument("--n", type=int, default=2, help="N-gram order (default: 2)")
    parser.add_argument("--length", type=int, default=40, help="Maximum generated tokens")
    parser.add_argument("--temp", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    print("=" * 60)
    print("PRODIGY_GA_03: Text Generation with Markov Chains")
    print("=" * 60)

    model = MarkovChainTextGenerator(n=args.n)
    model.train(DEMO_CORPUS)

    print(f"\n[+] Trained Markov Chain (order={args.n}) on sample corpus.")
    print(f"[+] Prompt: '{args.prompt}'")
    print(f"[+] Generated Output:\n")

    result = model.generate(prompt=args.prompt, max_tokens=args.length, temperature=args.temp)
    print(f'"{result}"')
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
