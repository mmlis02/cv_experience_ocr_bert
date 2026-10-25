# CLAUDE.md — Project rules (cv_experience_ocr_bert)

The assignment is in `docs/assignment.md`. These rules are permanent and apply to every stage.

## Language
- Code, comments, README and the Streamlit app are written in English.

## Data rules
- The reference `text` column is NEVER a model input. It is used only to compute CER/WER and for the Question 2 evaluation (OCR text vs reference text).
- Never use `age`, `sex`, `race`, names, `state` or `region` as features. `sex` and `race` are used only for the fairness check in Question 4.
- Use only images with `has_photo == False`.

## Reproducibility and evaluation
- `SEED = 42` everywhere (`src.utils.SEED`, `src.utils.set_seed`).
- All models use the same splits from `data/splits/`.
- Tuning is done only on validation. Test is used only in the final evaluation.
- The target `exp_year` is clipped to 0–20 (unless we explicitly decide on another cap), and predictions are clipped to the same range.

## Hardware
- PyTorch uses MPS if available (`src.utils.get_device()`). PaddleOCR runs on CPU.
- Do not use fp16 on MPS.

## Code organisation
- Every script runs as `python -m src.<name>`, caches its results, is resumable, and saves tables, figures and metrics to `output/`.
- Each stage has its logic in `src/*.py` and a notebook in `notebooks/` that uses it. Notebooks only call `src/` and explore results with tables, plots and markdown.
- Long processes (full OCR, BERT training) run as scripts, not inside notebooks.
- Paths come from `src/utils.py`; do not hard-code them.
- Notebook kernel: `cv_experience` (the project `.venv`, Python 3.11).

## Git workflow
- Never commit directly to `main`. Each stage lives on its own `feature/<stage>` branch and is merged through a Pull Request.
- Never commit images, raw data or model weights (see `.gitignore`).
