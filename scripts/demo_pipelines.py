"""Pretrained Transformers in a few lines: sentiment analysis, chat generation and sampling strategies."""

import argparse

import torch
from transformers import pipeline

SENTENCES = [
    "I loved the theater yesterday",
    "The service was slow and the food was cold.",
    "The acting was brilliant but the plot was terrible.",
    "It was not bad.",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument(
        "--prompt", default="Complete the sentence: The best way to get a good grade in mathematics is to"
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = 0 if torch.cuda.is_available() else -1

    print("=== Sentiment analysis (DistilBERT fine-tuned on SST-2)")
    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )
    for sentence, result in zip(SENTENCES, classifier(SENTENCES)):
        print(f"{result['label']:8s} {result['score']:.3f}  {sentence}")

    print(f"\n=== Chat generation ({args.generator})")
    generator = pipeline("text-generation", model=args.generator, device=device)
    request = [
        {"role": "user", "content": "Translate to French: The farmers take the cows up to the mountains."}
    ]
    print(generator(request, max_new_tokens=60, do_sample=False)[0]["generated_text"][-1]["content"])

    print("\n=== Sampling: temperature and top-p")
    prompt = [{"role": "user", "content": args.prompt}]
    for temperature, top_p in [(0.2, 1.0), (1.0, 0.9), (1.8, 1.0)]:
        print(f"--- temperature={temperature} top_p={top_p}")
        outputs = generator(
            prompt,
            max_new_tokens=30,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=3,
        )
        for out in outputs:
            print("  -", out["generated_text"][-1]["content"].replace("\n", " "))


if __name__ == "__main__":
    main()
