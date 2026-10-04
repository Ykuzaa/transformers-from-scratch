"""Train a from-scratch encoder-decoder Transformer for dialogue summarization (DialogSum)."""

import argparse
import time
from pathlib import Path

import evaluate
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from minitransformer import Seq2SeqTransformer
from minitransformer.training import count_parameters, get_device, plot_history, save_json, set_seed

IGNORE = -100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--src-max-length", type=int, default=512)
    p.add_argument("--tgt-max-length", type=int, default=64)
    p.add_argument("--train-size", type=int, default=None, help="use only the first N training dialogues")
    p.add_argument("--eval-size", type=int, default=200, help="test dialogues scored with ROUGE")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("results/summarizer"))
    return p.parse_args()


def make_collate(pad_id):
    """Pad each batch to its longest sequence (dynamic padding)."""

    def pad(seqs, value):
        return nn.utils.rnn.pad_sequence(
            [torch.tensor(s) for s in seqs], batch_first=True, padding_value=value
        )

    def collate(batch):
        src = pad([b["src"] for b in batch], pad_id)
        tgt_in = pad([b["tgt"][:-1] for b in batch], pad_id)  # <s> w1 ... wn
        labels = pad([b["tgt"][1:] for b in batch], IGNORE)  # w1 ... wn </s>
        return {
            "src": src,
            "src_mask": (src != pad_id).long(),
            "tgt_in": tgt_in,
            "tgt_mask": (labels != IGNORE).long(),
            "labels": labels,
        }

    return collate


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for batch in tqdm(loader, leave=False, desc="train" if training else "eval"):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch["src"], batch["tgt_in"], batch["src_mask"], batch["tgt_mask"])
            loss = criterion(logits.flatten(0, 1), batch["labels"].flatten())
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            n = (batch["labels"] != IGNORE).sum().item()
            total_loss += loss.item() * n
            total_correct += (
                ((logits.argmax(-1) == batch["labels"]) & (batch["labels"] != IGNORE)).sum().item()
            )
            total_tokens += n
    return total_loss / total_tokens, total_correct / total_tokens


def summarize(model, tokenizer, dialogues, device, args, batch_size=32):
    model.eval()
    outputs = []
    for start in range(0, len(dialogues), batch_size):
        enc = tokenizer(
            dialogues[start : start + batch_size],
            truncation=True,
            max_length=args.src_max_length,
            padding=True,
            return_tensors="pt",
        ).to(device)
        ids = model.generate(
            enc["input_ids"], enc["attention_mask"], tokenizer.bos_token_id, tokenizer.eos_token_id
        )
        outputs += tokenizer.batch_decode(ids, skip_special_tokens=True)
    return [o.strip() for o in outputs]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    raw = load_dataset("knkarthick/dialogsum")
    if args.train_size:
        raw["train"] = raw["train"].select(range(args.train_size))

    def tokenize(batch):
        src = tokenizer(batch["dialogue"], truncation=True, max_length=args.src_max_length)["input_ids"]
        tgt = tokenizer(batch["summary"], truncation=True, max_length=args.tgt_max_length + 1)["input_ids"]
        return {"src": src, "tgt": tgt}

    data = raw.map(tokenize, batched=True, remove_columns=raw["train"].column_names)
    collate = make_collate(tokenizer.pad_token_id)
    loaders = {
        split: torch.utils.data.DataLoader(
            data[split], batch_size=args.batch_size, shuffle=split == "train", collate_fn=collate
        )
        for split in ["train", "validation"]
    }

    model = Seq2SeqTransformer(
        vocab_size=len(tokenizer),
        src_max_length=args.src_max_length,
        tgt_max_length=args.tgt_max_length,
        hidden_dim=args.hidden_dim,
        num_heads=args.heads,
        num_layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    print(f"{count_parameters(model):,} parameters on {device}, {len(data['train'])} training dialogues")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out / "model.pt"

    history, best_val = [], float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["validation"], criterion, device)
        history.append(
            dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc)
        )
        print(
            f"epoch {epoch} | {time.time() - start:.0f}s | train loss {train_loss:.3f} acc {train_acc:.3f}"
            f" | val loss {val_loss:.3f} acc {val_acc:.3f}",
            flush=True,
        )
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))

    # DialogSum's test set lists each dialogue three times with different reference summaries.
    test = raw["test"]
    references: dict[str, list[str]] = {}
    for dialogue, summary in zip(test["dialogue"], test["summary"]):
        references.setdefault(dialogue, []).append(summary)
    dialogues = list(references)[: args.eval_size]
    predictions = summarize(model, tokenizer, dialogues, device, args)
    rouge = evaluate.load("rouge").compute(
        predictions=predictions, references=[references[d] for d in dialogues]
    )
    rouge = {k: round(float(v) * 100, 2) for k, v in rouge.items()}
    print("\nROUGE on", len(dialogues), "test dialogues:", rouge)

    samples = [
        {"dialogue": d, "reference": references[d][0], "prediction": p}
        for d, p in zip(dialogues[:5], predictions[:5])
    ]
    for s in samples[:3]:
        print("\nREFERENCE :", s["reference"], "\nPREDICTION:", s["prediction"])

    save_json(
        {
            "args": {k: str(v) for k, v in vars(args).items()},
            "history": history,
            "test_rouge": rouge,
            "samples": samples,
        },
        args.out / "metrics.json",
    )
    plot_history(
        history,
        args.out / "curves.png",
        "Dialogue summarizer (encoder-decoder from scratch)",
        metric_label="Next-token accuracy",
    )


if __name__ == "__main__":
    main()
