"""Train a from-scratch Transformer encoder on 6-way emotion classification (dair-ai/emotion)."""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorWithPadding

from minitransformer import TransformerClassifier
from minitransformer.training import count_parameters, get_device, plot_history, save_json, set_seed

EXAMPLES = [
    "I feel so happy and grateful today",
    "I am terrified of what might happen tomorrow",
    "I miss you so much, I feel lonely without you",
    "I can't believe they lied to me again, I'm furious",
    "I was shocked when I saw the results",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--pre-norm", action="store_true", help="LayerNorm before each sublayer")
    p.add_argument("--no-padding-mask", action="store_true", help="let tokens attend to [PAD] (ablation)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/classifier"))
    return p.parse_args()


def run_epoch(model, loader, criterion, device, use_mask, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, leave=False, desc="train" if training else "eval"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device) if use_mask else None
            y = batch["labels"].to(device)
            logits = model(ids, mask)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds += logits.argmax(-1).tolist()
            labels += y.tolist()
    acc = sum(p == y for p, y in zip(preds, labels)) / len(labels)
    return total_loss / len(labels), acc, preds, labels


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    dataset = load_dataset("dair-ai/emotion")
    label_names = dataset["train"].features["label"].names
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = dataset.map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=args.max_length), batched=True
    )
    dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    collate = DataCollatorWithPadding(tokenizer)
    loaders = {
        split: torch.utils.data.DataLoader(
            dataset[split], batch_size=args.batch_size, shuffle=split == "train", collate_fn=collate
        )
        for split in ["train", "validation", "test"]
    }

    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        max_length=args.max_length,
        num_classes=len(label_names),
        hidden_dim=args.hidden_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
        norm_first=args.pre_norm,
    ).to(device)
    print(f"{count_parameters(model):,} parameters on {device}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    use_mask = not args.no_padding_mask
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "model.pt"

    history, best_val = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc, *_ = run_epoch(model, loaders["train"], criterion, device, use_mask, optimizer)
        val_loss, val_acc, *_ = run_epoch(model, loaders["validation"], criterion, device, use_mask)
        history.append(
            dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        )
        print(
            f"epoch {epoch} | {time.time() - start:.0f}s | train loss {train_loss:.3f} acc {train_acc:.3f}"
            f" | val loss {val_loss:.3f} acc {val_acc:.3f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_loss, test_acc, preds, labels = run_epoch(model, loaders["test"], criterion, device, use_mask)
    macro_f1 = f1_score(labels, preds, average="macro")
    print(f"\nTest accuracy {test_acc:.3f} | macro-F1 {macro_f1:.3f} (best checkpoint by validation loss)")
    print(classification_report(labels, preds, target_names=label_names, digits=3))

    model.eval()
    enc = tokenizer(EXAMPLES, padding=True, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"] if use_mask else None), dim=-1)
    print("Examples:")
    examples = []
    for sentence, p in zip(EXAMPLES, probs):
        label = label_names[p.argmax().item()]
        examples.append({"text": sentence, "prediction": label, "confidence": round(p.max().item(), 3)})
        print(f"  {label:9s} {p.max().item():.0%}  {sentence}")

    save_json(
        {
            "args": {k: str(v) for k, v in vars(args).items()},
            "history": history,
            "test": {"loss": test_loss, "accuracy": test_acc, "macro_f1": macro_f1},
            "examples": examples,
        },
        args.out / "metrics.json",
    )
    plot_history(history, args.out / "curves.png", "Emotion classifier (encoder from scratch)")


if __name__ == "__main__":
    main()
