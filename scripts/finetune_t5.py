"""Fine-tune T5-small for English -> French translation on Europarl and measure the BLEU gain."""

import argparse
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from minitransformer.training import save_json, set_seed

PREFIX = "translate English to French: "
EXAMPLES = [
    "The farmers took the cows up to the mountains.",
    "He finally kicked the bucket after years of saying he wasn't ready to.",
    "When the engineer spoke to the manager about the delay, he admitted it was his fault.",
    "It's not so much that the plan failed as that it was never really tested.",
    "By the time she realized what had happened, the opportunity had already slipped away.",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="google-t5/t5-small")
    p.add_argument("--train-size", type=int, default=100_000)
    p.add_argument("--eval-size", type=int, default=1_000)
    p.add_argument("--epochs", type=float, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/t5_translation"))
    return p.parse_args()


def translate(model, tokenizer, sentences, max_length=128):
    model.eval()
    inputs = tokenizer([PREFIX + s for s in sentences], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_length, num_beams=4)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def main():
    args = parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    if torch.cuda.is_available():
        model = model.cuda()

    raw = load_dataset("Helsinki-NLP/europarl", "en-fr")["train"].train_test_split(
        test_size=0.1, seed=args.seed
    )
    train = raw["train"].shuffle(seed=args.seed).select(range(args.train_size))
    val = raw["test"].select(range(args.eval_size))

    def preprocess(batch):
        sources = [PREFIX + pair["en"] for pair in batch["translation"]]
        targets = [pair["fr"] for pair in batch["translation"]]
        return tokenizer(sources, text_target=targets, max_length=args.max_length, truncation=True)

    train_tok = train.map(preprocess, batched=True, remove_columns=train.column_names)
    val_tok = val.map(preprocess, batched=True, remove_columns=val.column_names)

    bleu = evaluate.load("sacrebleu")

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = [p.strip() for p in tokenizer.batch_decode(preds, skip_special_tokens=True)]
        decoded_labels = [
            [label.strip()] for label in tokenizer.batch_decode(labels, skip_special_tokens=True)
        ]
        return {"bleu": round(bleu.compute(predictions=decoded_preds, references=decoded_labels)["score"], 2)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        eval_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=2 * args.batch_size,
        weight_decay=0.01,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        generation_max_length=args.max_length,
        fp16=torch.cuda.is_available(),
        save_strategy="no",
        logging_steps=200,
        report_to="none",
        seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    before = trainer.evaluate()["eval_bleu"]
    examples_before = translate(model, tokenizer, EXAMPLES)
    print(f"BLEU before fine-tuning: {before}")
    trainer.train()
    after = trainer.evaluate()["eval_bleu"]
    examples_after = translate(model, tokenizer, EXAMPLES)
    print(f"BLEU after fine-tuning: {after}")
    for src, b, a in zip(EXAMPLES, examples_before, examples_after):
        print(f"\nEN    : {src}\nbefore: {b}\nafter : {a}")

    save_json(
        {
            "args": {k: str(v) for k, v in vars(args).items()},
            "bleu_before": before,
            "bleu_after": after,
            "examples": [
                {"en": s, "before": b, "after": a}
                for s, b, a in zip(EXAMPLES, examples_before, examples_after)
            ],
        },
        args.out / "metrics.json",
    )
    trainer.save_model(str(args.out / "model"))


if __name__ == "__main__":
    main()
