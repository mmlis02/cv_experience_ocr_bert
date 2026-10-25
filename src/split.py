"""Stratified train / validation / test split by CV ID.

70% / 15% / 15%, stratified by occupation x experience band, seed 42. Every model uses
these files; the test set is only used in the final evaluation.

Usage:
    python -m src.split
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import utils

SPLITS = ("train", "val", "test")


def make_splits(df: pd.DataFrame, seed: int = utils.SEED) -> dict[str, pd.DataFrame]:
    """Split CVs 70/15/15 stratified by occupation x band (exp_year already clipped)."""
    df = df[["id", "occupation", "exp_year"]].copy()
    df["band"] = utils.to_band(df["exp_year"])
    strata = df["occupation"] + "|" + df["band"]
    train, rest = train_test_split(df, test_size=0.30, stratify=strata, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=strata[rest.index], random_state=seed)
    return {name: part.sort_values("id").reset_index(drop=True) for name, part in zip(SPLITS, (train, val, test))}


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(utils.SPLITS_DIR / f"{name}.csv", dtype={"id": str})


def distribution(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """CVs per split by occupation and band, with the target mean."""
    rows = []
    for name, df in splits.items():
        row = {"split": name, "n": len(df), "share": len(df) / sum(len(d) for d in splits.values()),
               "exp_year_mean": df["exp_year"].mean(), "exp_year_median": df["exp_year"].median()}
        row.update({f"band_{b}": (df["band"] == b).mean() for b in utils.BAND_LABELS})
        row.update({o: (df["occupation"] == o).sum() for o in utils.OCCUPATIONS})
        rows.append(row)
    return pd.DataFrame(rows).round(3)


def main() -> None:
    df = pd.read_csv(utils.PROCESSED_DIR / "clean.csv", dtype={"id": str})
    splits = make_splits(df)
    ids = [set(s["id"]) for s in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]), "splits overlap"
    assert sum(map(len, ids)) == len(df), "some CVs are missing from the splits"

    utils.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(utils.SPLITS_DIR / f"{name}.csv", index=False)
    table = distribution(splits)
    table.to_csv(utils.TABLES_DIR / "split_distribution.csv", index=False)
    print(table.T.to_string())


if __name__ == "__main__":
    main()
