"""Baseline 2: frozen sentence embeddings + Ridge regression.

Each CV is encoded with a frozen pretrained encoder on MPS. CVs longer than the
encoder's limit are split into chunks of whole lines that fit the limit; the chunk
embeddings are averaged (weighted by token count) and L2-normalised. Ridge's alpha is
tuned on validation. Embeddings are cached in data/processed/embeddings/.

Both candidate encoders are compared on validation; the best one is saved.

Usage:
    python -m src.baseline_embeddings [--variant clean] [--encoders bge-small minilm]
"""

import argparse
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src import utils
from src.evaluation import regression_metrics

NAME = "embeddings"
ENCODERS = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}
ALPHAS = np.logspace(-4, 2, 13)
CACHE_DIR = utils.PROCESSED_DIR / "embeddings"
BATCH_SIZE = 64

_ENCODER_CACHE = {}


def get_encoder(key: str):
    from sentence_transformers import SentenceTransformer

    if key not in _ENCODER_CACHE:
        _ENCODER_CACHE[key] = SentenceTransformer(ENCODERS[key], device=utils.get_device())
    return _ENCODER_CACHE[key]


def chunk_text(text: str, tokenizer, max_tokens: int) -> list[str]:
    """Greedy chunks of whole lines with at most max_tokens tokens each (a line longer than that is cut)."""
    chunks, current, size = [], [], 0
    for line in text.split("\n"):
        n = len(tokenizer.tokenize(line))
        if current and size + n > max_tokens:
            chunks.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += n
    if current:
        chunks.append("\n".join(current))
    return chunks


def embed(texts, key: str) -> np.ndarray:
    """One L2-normalised vector per text: token-weighted mean of its chunk embeddings."""
    encoder = get_encoder(key)
    tokenizer = encoder.tokenizer
    limit = encoder.max_seq_length - 2  # [CLS] and [SEP]
    chunks, owner, weights = [], [], []
    for i, text in enumerate(texts):
        for chunk in chunk_text(text, tokenizer, limit):
            chunks.append(chunk)
            owner.append(i)
            weights.append(max(len(tokenizer.tokenize(chunk)), 1))
    vectors = encoder.encode(chunks, batch_size=BATCH_SIZE, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=False)
    out = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    total = np.zeros(len(texts))
    for vec, i, w in zip(vectors, owner, weights):
        out[i] += w * vec
        total[i] += w
    out /= total[:, None]
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def cached_embeddings(key: str, variant: str) -> pd.DataFrame:
    """Embeddings of all 4,000 CVs for one encoder and text variant (computed once)."""
    path = CACHE_DIR / f"{key}__{variant}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    texts = pd.read_csv(utils.PROCESSED_DIR / f"{variant}.csv", dtype={"id": str})
    start = time.perf_counter()
    emb = embed(texts["text"].tolist(), key)
    seconds = time.perf_counter() - start
    df = pd.DataFrame(emb, index=texts["id"], columns=[f"e{i}" for i in range(emb.shape[1])])
    df.attrs["seconds"] = seconds
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"  encoded {len(df)} CVs with {key} in {seconds:.0f}s")
    return df


def model_path(variant: str):
    return utils.BASELINES_DIR / f"{NAME}__{variant}.joblib"


def load(variant: str = "clean") -> dict:
    return joblib.load(model_path(variant))


def predict(texts, variant: str = "clean", model: dict | None = None) -> np.ndarray:
    """Encode the texts with the saved encoder and apply the saved Ridge."""
    model = model or load(variant)
    return utils.clip_years(model["ridge"].predict(embed(list(texts), model["encoder"])))


def main(variant: str, encoders) -> dict:
    utils.set_seed()
    train = utils.load_split_texts("train", variant)
    val = utils.load_split_texts("val", variant)

    rows, fitted = [], {}
    for key in encoders:
        emb = cached_embeddings(key, variant)
        x_train, x_val = emb.loc[train["id"]].to_numpy(), emb.loc[val["id"]].to_numpy()
        for alpha in ALPHAS:
            ridge = Ridge(alpha=alpha).fit(x_train, train["exp_year"])
            mae = regression_metrics(val["exp_year"], ridge.predict(x_val))["mae"]
            rows.append({"encoder": key, "alpha": alpha, "val_mae": mae})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid["val_mae"].idxmin()]
    key = best["encoder"]

    # Training time = encoding the training CVs + fitting Ridge
    start = time.perf_counter()
    x_train = embed(train["text"].tolist(), key)
    ridge = Ridge(alpha=best["alpha"]).fit(x_train, train["exp_year"])
    train_seconds = time.perf_counter() - start
    model = {"encoder": key, "encoder_name": ENCODERS[key], "alpha": float(best["alpha"]), "ridge": ridge}
    utils.BASELINES_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path(variant))

    start = time.perf_counter()
    y_val = predict(val["text"], model=model)
    infer = (time.perf_counter() - start) / len(val)

    utils.save_predictions(NAME, variant, "val", val, y_val)
    grid.round(4).to_csv(utils.TABLES_DIR / f"embeddings_grid__{variant}.csv", index=False)
    best_by_encoder = grid.loc[grid.groupby("encoder")["val_mae"].idxmin()]
    metrics = {"model": NAME, "variant": variant, "encoder": ENCODERS[key], "alpha": float(best["alpha"]),
               "device": utils.get_device(), "best_val_mae_by_encoder": best_by_encoder.round(4).to_dict("records"),
               "max_seq_length": get_encoder(key).max_seq_length,
               "train_seconds": train_seconds, "inference_seconds_per_cv": infer,
               "val": regression_metrics(val["exp_year"], y_val)}
    utils.save_metrics(f"{NAME}__{variant}", metrics)
    print(best_by_encoder.round(4).to_string(index=False))
    print(f"[{NAME}/{variant}] {key} alpha {best['alpha']:.4g} · val MAE {metrics['val']['mae']:.3f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=utils.VARIANTS, default="clean")
    parser.add_argument("--encoders", nargs="+", choices=sorted(ENCODERS), default=sorted(ENCODERS))
    args = parser.parse_args()
    main(args.variant, args.encoders)
