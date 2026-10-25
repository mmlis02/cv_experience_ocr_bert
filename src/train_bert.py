"""Fine-tuning of a lightweight BERT-based model as a regressor.

distilbert-base-uncased with a single regression output
(AutoModelForSequenceClassification, num_labels=1, problem_type="regression"), trained in
fp32 on MPS. The target is divided by EXP_CAP (20) during training; predictions are
multiplied back and clipped to [0, 20].

Long-text strategy (--strategy):
- head       keep the first max_len tokens (default: no CV exceeds 162 tokens, so with
             max_len 256 nothing is cut and every job date is kept)
- head_tail  keep the first quarter and the last three quarters of the budget
- chunk_avg  split into overlapping windows, train on every window, average at inference

Early stopping on validation MAE (patience 2, at most 8 epochs). The best epoch is saved
in models/bert/<run>/, its validation predictions in output/predictions/val/ and the
configuration, curve, timing and hardware in output/metrics/bert/.

Usage:
    python -m src.train_bert                                   # clean text, MSE, head
    python -m src.train_bert --variant raw
    python -m src.train_bert --loss huber
    python -m src.train_bert --limit-train 200 --max-epochs 1 --run-name quick_test
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import platform
import re
import subprocess
import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src import utils
from src.evaluation import regression_metrics

MODEL_NAME = "distilbert-base-uncased"
STRATEGIES = ("head", "head_tail", "chunk_avg")
LOSSES = ("mse", "huber")


@dataclass
class Config:
    variant: str = "clean"
    strategy: str = "head"
    loss: str = "mse"
    huber_delta: float = 0.05  # in scaled units: 0.05 x 20 = 1 year
    max_len: int = 256
    chunk_stride: int = 64
    lr: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 8
    grad_accum: int = 2  # effective batch 16
    max_epochs: int = 8
    patience: int = 2
    seed: int = utils.SEED
    limit_train: int | None = None
    run_name: str | None = None

    @property
    def run(self) -> str:
        if self.run_name:
            return self.run_name
        suffix = "" if (self.loss, self.strategy) == ("mse", "head") else f"_{self.loss}_{self.strategy}"
        return f"bert{suffix}__{self.variant}"

    @property
    def model_label(self) -> str:
        return self.run.split("__")[0]


# ------------------------------------------------------------------ tokenisation

def encode(texts, tokenizer, cfg: Config) -> list[dict]:
    """Token windows per text according to the strategy: [{"doc": i, "input_ids": [...]}, ...]."""
    body = cfg.max_len - 2  # room for [CLS] and [SEP]
    out = []
    for i, ids in enumerate(tokenizer(list(texts), add_special_tokens=False)["input_ids"]):
        if len(ids) <= body:
            windows = [ids]
        elif cfg.strategy == "head":
            windows = [ids[:body]]
        elif cfg.strategy == "head_tail":
            head = body // 4
            windows = [ids[:head] + ids[-(body - head):]]
        else:  # chunk_avg: windows of `body` tokens overlapping by chunk_stride (at most half a window)
            overlap = min(cfg.chunk_stride, body // 2)
            windows = [ids[s:s + body] for s in range(0, max(len(ids) - overlap, 1), body - overlap)]
        for w in windows:
            out.append({"doc": i, "input_ids": [tokenizer.cls_token_id] + w + [tokenizer.sep_token_id]})
    return out


def truncation_report(texts, tokenizer, cfg: Config) -> dict:
    """How many CVs are cut, and how many lose at least one year that the full text contains."""
    year_re = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
    windows = encode(texts, tokenizer, cfg)
    seen: dict[int, str] = {}
    for w in windows:
        seen[w["doc"]] = seen.get(w["doc"], "") + " " + tokenizer.decode(w["input_ids"], skip_special_tokens=True)
    n_tokens = [len(ids) + 2 for ids in tokenizer(list(texts), add_special_tokens=False)["input_ids"]]
    lost = sum(bool(set(year_re.findall(t)) - set(year_re.findall(seen[i]))) for i, t in enumerate(texts))
    return {"max_tokens": int(max(n_tokens)), "cvs_truncated": int(sum(n > cfg.max_len for n in n_tokens)),
            "cvs_losing_a_year": int(lost), "windows_per_cv": len(windows) / len(texts)}


def make_loader(windows, labels, tokenizer, batch_size, shuffle, seed):
    from transformers import DataCollatorWithPadding

    collate = DataCollatorWithPadding(tokenizer)
    items = [{"input_ids": w["input_ids"], "doc": w["doc"],
              "labels": float(labels[w["doc"]]) if labels is not None else 0.0} for w in windows]

    def collate_fn(batch):
        # Read, never pop: the items are reused every epoch (train and val loaders)
        docs = torch.tensor([b["doc"] for b in batch])
        labels_ = torch.tensor([b["labels"] for b in batch], dtype=torch.float32)
        enc = collate([{"input_ids": b["input_ids"]} for b in batch])
        return enc, labels_, docs

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, generator=generator)


# ------------------------------------------------------------------ model

def load_model(path=None):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        path or MODEL_NAME, num_labels=1, problem_type="regression", dtype=torch.float32)
    return model.float().to(utils.get_device())


@torch.no_grad()
def predict_windows(model, loader, n_docs: int) -> np.ndarray:
    """Average the (scaled) predictions of every window of a document, return years."""
    model.eval()
    device = next(model.parameters()).device
    total, count = np.zeros(n_docs), np.zeros(n_docs)
    for enc, _, docs in loader:
        out = model(**{k: v.to(device) for k, v in enc.items()}).logits.squeeze(-1).float().cpu().numpy()
        np.add.at(total, docs.numpy(), out)
        np.add.at(count, docs.numpy(), 1)
    return utils.clip_years(total / count * utils.EXP_CAP)


def loss_fn(cfg: Config):
    if cfg.loss == "huber":
        return torch.nn.HuberLoss(delta=cfg.huber_delta)
    return torch.nn.MSELoss()


def hardware() -> dict:
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    import transformers
    return {"device": utils.get_device(), "cpu": cpu or platform.processor(), "torch": torch.__version__,
            "transformers": transformers.__version__, "dtype": "float32"}


# ------------------------------------------------------------------ train

def train(cfg: Config) -> dict:
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    utils.set_seed(cfg.seed)
    device = utils.get_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_df = utils.load_split_texts("train", cfg.variant)
    val_df = utils.load_split_texts("val", cfg.variant)
    if cfg.limit_train:
        train_df = train_df.sample(cfg.limit_train, random_state=cfg.seed).reset_index(drop=True)

    trunc = {"train": truncation_report(train_df["text"].tolist(), tokenizer, cfg),
             "val": truncation_report(val_df["text"].tolist(), tokenizer, cfg)}
    y_train = (train_df["exp_year"].to_numpy() / utils.EXP_CAP).astype(np.float32)
    train_loader = make_loader(encode(train_df["text"], tokenizer, cfg), y_train, tokenizer,
                               cfg.batch_size, shuffle=True, seed=cfg.seed)
    val_loader = make_loader(encode(val_df["text"], tokenizer, cfg), None, tokenizer,
                             cfg.batch_size * 4, shuffle=False, seed=cfg.seed)

    model = load_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum)
    total_updates = updates_per_epoch * cfg.max_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total_updates), total_updates)
    criterion = loss_fn(cfg)
    out_dir = utils.BERT_DIR / cfg.run
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best_mae, best_epoch, bad_epochs = [], math.inf, 0, 0
    start_all = time.perf_counter()
    print(f"[{cfg.run}] {len(train_df)} train CVs · {len(train_loader)} batches/epoch · device {device} · "
          f"max tokens {trunc['train']['max_tokens']}", flush=True)
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        start, running, n_batches = time.perf_counter(), 0.0, 0
        optimizer.zero_grad()
        for step, (enc, labels, _) in enumerate(train_loader, 1):
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits.squeeze(-1)
            loss = criterion(logits, labels.to(device)) / cfg.grad_accum
            loss.backward()
            running += loss.item() * cfg.grad_accum
            n_batches += 1
            if step % cfg.grad_accum == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        train_seconds = time.perf_counter() - start

        y_val = predict_windows(model, val_loader, len(val_df))
        val = regression_metrics(val_df["exp_year"], y_val)
        history.append({"epoch": epoch, "train_loss": running / n_batches, "val_mae": val["mae"],
                        "val_rmse": val["rmse"], "val_acc_2": val["acc_2"], "epoch_seconds": train_seconds,
                        "lr_end": scheduler.get_last_lr()[0]})
        print(f"  epoch {epoch}: train loss {running / n_batches:.5f} · val MAE {val['mae']:.3f} · "
              f"{train_seconds:.0f}s", flush=True)
        if val["mae"] < best_mae - 1e-4:
            best_mae, best_epoch, bad_epochs = val["mae"], epoch, 0
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"  early stopping: no improvement for {cfg.patience} epochs", flush=True)
                break
    total_seconds = time.perf_counter() - start_all

    # Reload the best epoch for the final validation predictions and the inference timing
    best = load_model(out_dir)
    start = time.perf_counter()
    y_val = predict_windows(best, make_loader(encode(val_df["text"], tokenizer, cfg), None, tokenizer,
                                              cfg.batch_size * 4, False, cfg.seed), len(val_df))
    infer = (time.perf_counter() - start) / len(val_df)

    # Quick tests on a train subsample are kept apart from the real results
    subdir = "bert_tests" if cfg.limit_train else "bert"
    if not cfg.limit_train:
        utils.save_predictions(cfg.model_label, cfg.variant, "val", val_df, y_val)
    hist = pd.DataFrame(history)
    (utils.METRICS_DIR / subdir).mkdir(parents=True, exist_ok=True)
    hist.to_csv(utils.METRICS_DIR / subdir / f"{cfg.run}_history.csv", index=False)
    (out_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    metrics = {"model": cfg.model_label, "variant": cfg.variant, "run": cfg.run, "base_model": MODEL_NAME,
               "config": asdict(cfg), "effective_batch_size": cfg.batch_size * cfg.grad_accum,
               "epochs_trained": len(history), "best_epoch": best_epoch, "n_train": len(train_df),
               "truncation": trunc, "train_seconds": total_seconds,
               "seconds_per_epoch": float(hist["epoch_seconds"].mean()),
               "inference_seconds_per_cv": infer, "hardware": hardware(),
               "val": regression_metrics(val_df["exp_year"], y_val), "history": history}
    utils.save_metrics(cfg.run, metrics, subdir=subdir)
    print(f"[{cfg.run}] best epoch {best_epoch} · val MAE {metrics['val']['mae']:.3f} · "
          f"{total_seconds / 60:.1f} min · {infer * 1000:.1f} ms/CV inference", flush=True)
    return metrics


# ------------------------------------------------------------------ inference

_LOADED = {}


def predict(texts, variant: str = "clean", run: str | None = None) -> np.ndarray:
    """Predicted years for raw strings with the saved model of a run (default: bert__<variant>)."""
    from transformers import AutoTokenizer

    run = run or f"bert__{variant}"
    path = utils.BERT_DIR / run
    if run not in _LOADED:
        cfg = Config(**json.loads((path / "train_config.json").read_text()))
        _LOADED[run] = (load_model(path), AutoTokenizer.from_pretrained(path), cfg)
    model, tokenizer, cfg = _LOADED[run]
    texts = list(texts)
    loader = make_loader(encode(texts, tokenizer, cfg), None, tokenizer, cfg.batch_size * 4, False, cfg.seed)
    return predict_windows(model, loader, len(texts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=utils.VARIANTS, default="clean")
    parser.add_argument("--strategy", choices=STRATEGIES, default="head")
    parser.add_argument("--loss", choices=LOSSES, default="mse")
    parser.add_argument("--huber-delta", type=float, default=Config.huber_delta)
    parser.add_argument("--max-len", type=int, default=Config.max_len)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--grad-accum", type=int, default=Config.grad_accum)
    parser.add_argument("--max-epochs", type=int, default=Config.max_epochs)
    parser.add_argument("--patience", type=int, default=Config.patience)
    parser.add_argument("--limit-train", type=int, default=None, help="subsample train (quick tests)")
    parser.add_argument("--run-name", default=None, help="output name (default bert[_loss_strategy]__variant)")
    args = parser.parse_args()
    train(Config(variant=args.variant, strategy=args.strategy, loss=args.loss, huber_delta=args.huber_delta,
                 max_len=args.max_len, lr=args.lr, batch_size=args.batch_size, grad_accum=args.grad_accum,
                 max_epochs=args.max_epochs, patience=args.patience, limit_train=args.limit_train,
                 run_name=args.run_name))


if __name__ == "__main__":
    main()
