"""Streamlit app (4 tabs): methodology, OCR & EDA, model results, interactive estimator.

Reads the results already computed in output/ and the saved models; it never retrains.

Run:
    streamlit run app.py
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src import utils
from src import visualization as viz

st.set_page_config(page_title="CV experience estimator", page_icon="📄", layout="wide")

T, F, M = utils.TABLES_DIR, utils.FIGURES_DIR, utils.METRICS_DIR
MODELS = ["median", "tfidf", "embeddings", "bert", "rules"]
LABELS = {"median": "Median", "tfidf": "TF-IDF + Ridge", "embeddings": "Embeddings + Ridge",
          "bert": "Fine-tuned BERT", "rules": "Date rules (bonus)"}


# ------------------------------------------------------------------ cached data

@st.cache_data
def table(name: str) -> pd.DataFrame:
    return pd.read_csv(T / name, dtype={"id": str})


@st.cache_data
def metrics_json(path: str) -> dict:
    return json.loads((M / path).read_text())


@st.cache_data
def test_predictions() -> dict:
    return {m: pd.read_csv(utils.PREDICTIONS_DIR / "test" / f"{m}__clean.csv", dtype={"id": str}) for m in MODELS}


@st.cache_data
def targets() -> pd.DataFrame:
    """exp_year per CV: raw values from metadata if the raw data is present, else the clipped target."""
    path = utils.RAW_DIR / "metadata.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"id": str})[["id", "occupation", "exp_year"]]
    return pd.read_csv(utils.PROCESSED_DIR / "clean.csv", dtype={"id": str})[["id", "occupation", "exp_year"]]


@st.cache_data
def clean_texts() -> pd.Series:
    return pd.read_csv(utils.PROCESSED_DIR / "clean.csv", dtype={"id": str}).set_index("id")["text"]


def figure(fig):
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def why(text: str):
    st.caption(text)


# ------------------------------------------------------------------ cached models (8 GB RAM: load once)

@st.cache_resource(show_spinner="Loading PaddleOCR (mobile, CPU)...")
def ocr_engine():
    from src.ocr_extraction import build_ocr
    return build_ocr("mobile", cpu_threads=4)


@st.cache_resource(show_spinner="Loading the trained models...")
def trained_models() -> dict:
    from src import baseline_embeddings, baseline_median, baseline_tfidf
    return {"median": baseline_median.load("clean"), "tfidf": baseline_tfidf.load("clean"),
            "embeddings": baseline_embeddings.load("clean")}


def predict_all(text: str) -> dict:
    """Predicted years from every clean-trained model for one cleaned text."""
    from src import baseline_embeddings, baseline_rules, baseline_tfidf, train_bert

    models = trained_models()
    return {
        "median": float(models["median"]["median"]),
        "tfidf": float(baseline_tfidf.predict([text], model=models["tfidf"])[0]),
        "embeddings": float(baseline_embeddings.predict([text], model=models["embeddings"])[0]),
        "bert": float(train_bert.predict([text], variant="clean")[0]),
        "rules": float(baseline_rules.predict([text])[0]),
    }


def analyze_image(path: Path) -> dict:
    """Full pipeline for one CV image: PaddleOCR -> reading order -> cleaning -> predictions."""
    from src.ocr_extraction import reading_order, run_ocr
    from src.preprocessing import detect_name, ocr_lines, process_lines

    record = run_ocr(ocr_engine(), path, "mobile")
    order = reading_order(record)
    lines = ocr_lines(record)
    clean, info = process_lines(lines, clean=True, name=detect_name(lines))
    return {"record": record, "order": order, "raw_text": "\n".join(lines), "clean_text": clean,
            "info": info, "predictions": predict_all(clean)}


def pdf_to_png(data: bytes, path: Path, dpi: int = 150) -> Path:
    """First page of a PDF rendered with PyMuPDF (150 dpi ~ the 1241 px width of the dataset images)."""
    import fitz

    with fitz.open(stream=data, filetype="pdf") as doc:
        doc[0].get_pixmap(dpi=dpi).save(path)
    return path


# ------------------------------------------------------------------ tab 1

def tab_methodology():
    download = metrics_json("download_stats.json")
    ocr = metrics_json("ocr_summary.json")
    prep = metrics_json("preprocessing_report.json")
    dist = table("exp_year_distribution.csv").set_index("occupation")
    cmp_ = table("model_comparison.csv").set_index("model")

    st.header("Problem & Methodology")
    st.markdown(f"""
**Objective.** Read a CV *image*, extract its text with PaddleOCR, and estimate the candidate's **years of work
experience (0–20)**.

**Why it is hard.** The number of years is almost never written on the CV. A reader has to connect information
spread over the page: the start and end date of every job, how many jobs there are, whether they overlap or
leave gaps, and when education ended. The model gets only the OCR text, so it must learn that arithmetic from
examples, and any digit the OCR misreads changes the answer.
""")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Data")
        st.markdown(f"""
- **Dataset:** PopResume (Yu, Park & Moon, EMNLP 2026, CC BY 4.0), 60,884 synthetic resumes generated from
  U.S. population statistics. No real person's data.
- **Sample:** {download['n_per_occupation']} CVs per occupation × 5 occupations = **{5 * download['n_per_occupation']:,}
  CVs**. Only **no-photo** images (`has_photo == False`). Each occupation is streamed, shuffled with seed
  {download['seed']} (buffer {download['buffer_size']:,}), and the first 800 are taken. The sampled IDs are saved.
- **Target:** `exp_year`, continuous, clipped to **0–20** ("20 = 20 or more"); predictions are clipped the
  same way. Only **{int(dist.loc['all', 'n_ge_20'])} CVs ({dist.loc['all', 'pct_ge_20']:.1f}%)** sit at ≥ 20,
  far below the 15% threshold, so the cap stays at 20.
- **Never used as features:** age, sex, race, names, state, region. Sex and race are used only for the fairness
  check. The reference `text` column is never a model input; it is used only to measure OCR quality and in
  Question 2.
- **Masking rule:** phrases stating total experience, age or date of birth are replaced with `[MASKED]`.
  **{prep['cvs_with_direct_answer_masked']} CVs were affected**: none of the 4,000 CVs (OCR or reference) states
  them. Name, email, phone and address are removed from every CV.
""")
    with c2:
        st.subheader("OCR and reading order")
        st.markdown(f"""
- **PaddleOCR 3.x on CPU**: `{ocr['det_model']}` + `{ocr['rec_model']}`, no orientation classification, no
  unwarping. Mobile was chosen over server after a 20-CV benchmark: it is 32× faster (5.8 vs 187 s per CV) and
  just as accurate once date ranges are repaired.
- **Full run:** {ocr['n_ok']:,} / {ocr['n_images']:,} images, {ocr['n_failed']} failures, mean line confidence
  {ocr['mean_line_confidence']:.3f}, about {ocr['wall_hours_from_file_times']:.1f} h on an Apple M2 (4 processes × 2 threads).
- **Reading order** is rebuilt from the boxes:
  - find a vertical gap between columns;
  - full-width boxes cut the page into bands;
  - in each band, read the left column, then the right one;
  - boxes on the same visual line are joined, so a job title stays with its dates.

  All PopResume CVs are single-column; a synthetic two-column page checks the logic.
- **Cleaning:** the OCR drops the dash in about half of the date ranges (`2020.112023.01`). A regex re-inserts it
  in {prep['cvs_with_glued_date_ranges_repaired']:,} CVs. Bullets and dashes are normalised and broken lines joined.
""")
    st.subheader("Models and split")
    st.markdown(f"""
| Model | Intuition | Test MAE |
|---|---|---|
| Median | always predicts the training median (5.5 years) | {cmp_.loc['Median', 'mae_years']:.2f} |
| TF-IDF + Ridge | counts words, word pairs and character pieces (including years) and weights them | {cmp_.loc['TF-IDF + Ridge', 'mae_years']:.2f} |
| Embeddings + Ridge | a frozen `bge-small-en-v1.5` encoder summarises the CV's meaning; Ridge maps it to years | {cmp_.loc['Embeddings + Ridge', 'mae_years']:.2f} |
| Fine-tuned BERT | `distilbert-base-uncased` trained end to end as a regressor (target / 20, MSE, early stopping) | {cmp_.loc['Fine-tuned BERT', 'mae_years']:.2f} |
| Date rules (bonus) | regex date ranges, merge overlaps, add the months; no training | {cmp_.loc['Date rules (bonus)', 'mae_years']:.2f} |

**Split:** 70 / 15 / 15 by CV ID (2,800 / 600 / 600), stratified by occupation × experience band, seed 42. Every
model uses the same split. Tuning uses validation only, and the test set was used once, in the final evaluation.
""")
    st.subheader("Limitations")
    st.markdown("""
- **Synthetic resumes from a single template.** The target is exactly the sum of the listed job durations, so
  simple date arithmetic nearly solves the task. Real CVs have free-form dates ("since 2019", "3 yrs"),
  part-time and overlapping jobs, and multi-column layouts.
- **Narrow scope:** 5 occupations, U.S. population statistics, English only.
- **One training run per model and variant:** differences of a few tenths of a year between text variants may
  partly be training noise.
- **Not a decision system.** An experience estimate can pre-fill a profile for a human to check. It must never
  decide on its own who gets an interview or a job.
""")


# ------------------------------------------------------------------ tab 2

def tab_ocr_eda():
    st.header("OCR & Exploratory Analysis")
    meta = targets()
    dist = table("exp_year_distribution.csv")

    st.subheader("Years of experience")
    figure(viz.plot_exp_year_distribution(meta))
    why("Why this chart: a 1-year histogram shows the shape of the target directly: where most CVs sit, how long "
        "the right tail is, and how many reach the cap (dashed line). The small panels repeat it per occupation "
        "because the split and the error analysis are stratified by occupation. Most CVs have 2–8 years; "
        "construction laborers are the most junior.")
    st.dataframe(dist, hide_index=True, width="stretch")

    st.subheader("OCR quality")
    figure(viz.plot_ocr_quality(table("ocr_quality_per_cv.csv"), table("ocr_quality_by_occupation.csv")))
    why("Why this chart: box plots show the whole per-CV distribution of character and word error rates by "
        "occupation, and the bars compare the share of reference years found in the OCR text before and after "
        "repairing date ranges, because years are the evidence the models need. Errors are tiny (CER 0.2%), but "
        "the dropped dash hides about 20% of the years until it is repaired.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Where the OCR fails** (character errors by type of reference line)")
        st.dataframe(table("ocr_errors_by_line_type.csv"), hide_index=True, width="stretch")
    with c2:
        st.markdown("**Most frequent word errors** (digits as #)")
        st.dataframe(table("ocr_top_word_errors.csv").head(8), hide_index=True, width="stretch")

    st.subheader("A CV with its PaddleOCR boxes")
    from src.ocr_extraction import load_ocr, reading_order

    ids_with_images = [p.stem for p in utils.RAW_IMAGES_DIR.glob("*.png")]
    if ids_with_images:
        meta_ids = meta[meta["id"].isin(ids_with_images)]
        occ = st.selectbox("Occupation", utils.OCCUPATIONS, key="eda_occ")
        cv_id = st.selectbox("CV id", sorted(meta_ids.loc[meta_ids["occupation"] == occ, "id"])[:200], key="eda_id")
        record = load_ocr(cv_id)
        order = reading_order(record)
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(7, 7))
            viz.plot_reading_order(record, order, Image.open(utils.RAW_IMAGES_DIR / f"{cv_id}.png"), ax,
                                   title=f"id {cv_id} · layout {order['layout']}")
            figure(fig)
        with c2:
            st.markdown("**Extracted text, in reading order** (before cleaning)")
            st.code("\n".join(l["text"] for l in order["lines"]), language=None)
            st.markdown(f"OCR time {record['seconds']:.1f} s · {record['n_lines']} lines · "
                        f"mean confidence {record['mean_score']:.3f}")
    else:
        st.info("Raw images are not in the repository; showing the saved figures. Run `python -m src.download_data` "
                "to browse any CV.")
        name = st.selectbox("Example", ["registered_nurses", "software_developers", "construction_laborers",
                                        "two_column_check"])
        st.image(str(F / f"ocr_reading_order_{name}.png"), width="stretch")
    why("Each number is one step of the rebuilt reading order. Boxes on the same visual line share a number, so a "
        "job title and its dates are read together. Colours mark full-width, left and right columns.")


# ------------------------------------------------------------------ tab 3

def tab_results():
    st.header("Model Results (test set, 600 CVs)")
    preds = test_predictions()

    st.subheader("Model comparison")
    st.dataframe(table("model_comparison.csv"), hide_index=True, width="stretch")
    why("All models are scored on the same 600 test CVs, never used before the final evaluation. BERT is off by 0.25 "
        "years on average and every prediction is within ±2 years; the date rules, which need no training, are "
        "10× more accurate still.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Predicted vs true")
        model = st.selectbox("Model", MODELS, index=3, format_func=LABELS.get, key="pvt_model")
        figure(viz.plot_pred_vs_true({LABELS[model]: preds[model]}))
        why("Points on the dashed 45° line are perfect. A fitted slope below 1 means predictions are pulled toward "
            "the middle (juniors overestimated, seniors underestimated): embeddings 0.86, TF-IDF 0.92, BERT 0.99.")
    with c2:
        st.subheader("Band confusion matrix")
        model_cm = st.selectbox("Model", MODELS, index=3, format_func=LABELS.get, key="cm_model")
        figure(viz.plot_band_confusion({LABELS[model_cm]: preds[model_cm]}))
        why("Rows are the true band (0–2, 3–5, 6–10, 11–20 years), columns the predicted one, as row percentages. "
            "BERT's mistakes only spill into the neighbouring band.")

    st.subheader("MAE by experience band and by occupation")
    figure(viz.plot_mae_by_group(table("mae_by_band.csv"), "band", utils.BAND_LABELS))
    figure(viz.plot_mae_by_group(table("mae_by_occupation.csv"), "occupation", utils.OCCUPATIONS))
    why("Grouped bars put all models side by side within each group. BERT's gain over TF-IDF is largest at the "
        "extremes (seniors 1.16 → 0.29 years), where the baselines are pulled toward the middle, and it is uniform "
        "across occupations.")

    st.subheader("Question 1 · BERT's gain and what it learns")
    gain = table("bert_gain_bootstrap.csv")
    figure(viz.plot_bert_gain(gain))
    st.dataframe(gain, hide_index=True, width="stretch")
    why("Paired bootstrap, 2,000 resamples of the same test CVs, 95% CI. BERT's reduction is significant against every "
        "learned baseline (−0.59 years vs TF-IDF); against the date rules it is significantly worse (+0.23).")
    st.markdown("**Probe on 20 test CVs.** The expected change comes from the date rules: delete one job → minus its "
                "duration; shift every date +5 years → 0 (durations do not change).")
    st.dataframe(table("q1_probe_summary.csv"), hide_index=True, width="stretch")
    st.markdown("BERT moves in the right direction when a job is deleted (20/20), but by less than the job's length "
                "and uncorrelated with it. Shifting dates into the future makes it drop 1.45 years. It relies partly "
                "on absolute years and on the number of jobs; only the rules read dates like a recruiter.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Question 2 · OCR vs reference text")
        st.dataframe(table("q2_ocr_vs_reference.csv"), hide_index=True, width="stretch")
        why("Each clean-trained model scored on the OCR text and on the reference text of the same test CVs. "
            "After the date repair, OCR noise costs less than 0.015 years for every model.")
    with c2:
        st.subheader("Question 3 · Text cleaning sensitivity")
        st.dataframe(table("q3_variant_mae.csv"), hide_index=True, width="stretch")
        st.dataframe(table("q3_variant_bootstrap.csv"), hide_index=True, width="stretch")
        why("Each model trained and tested on raw OCR, cleaned OCR, and cleaned OCR without low-confidence lines. "
            "BERT loses 0.17 years on raw text (glued dates). That is significant against test noise, but each "
            "variant was trained once, so the size should be confirmed with more seeds.")

    st.subheader("Question 4 · Fairness check")
    fair = table("q4_fairness.csv")
    figure(viz.plot_fairness(fair))
    st.dataframe(fair, hide_index=True, width="stretch")
    why("BERT's mean error (predicted − true) per group, 95% bootstrap CI. MAE is the same in every group (0.24–0.26). "
        "The mean error ranges from −0.02 (Black, n=47) to +0.15 years (Asian/Pacific Islander), about 2 months; "
        "the date rules show no difference. Sex and race are not inputs, but career patterns correlate with them.")


# ------------------------------------------------------------------ tab 4

def show_predictions(preds: dict, true=None):
    cols = st.columns(len(MODELS))
    for col, m in zip(cols, MODELS):
        delta = None if true is None else f"{preds[m] - true:+.2f} vs true"
        col.metric(LABELS[m], f"{preds[m]:.1f} yrs", delta=delta, delta_color="off")


def tab_estimator():
    st.header("Interactive Experience Estimator")

    st.subheader("Estimate from your own CV")
    upload = st.file_uploader("Upload a CV image (PNG/JPG) or a PDF (first page is used)",
                              type=["png", "jpg", "jpeg", "pdf"])
    if upload is not None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cv.png"
            if upload.name.lower().endswith(".pdf"):
                pdf_to_png(upload.getvalue(), path)
            else:
                Image.open(upload).convert("RGB").save(path)
            with st.spinner("Running PaddleOCR, reading order, cleaning and all models..."):
                result = analyze_image(path)
            image = Image.open(path).copy()
        show_predictions(result["predictions"])
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(7, 7))
            viz.plot_reading_order(result["record"], result["order"], image, ax,
                                   title=f"layout {result['order']['layout']} · OCR {result['record']['seconds']:.1f}s")
            figure(fig)
        with c2:
            st.markdown("**Cleaned text given to the models** (name and contact removed)")
            st.code(result["clean_text"], language=None)
            with st.expander("Raw OCR text (reading order)"):
                st.code(result["raw_text"], language=None)
        st.caption("The date rules add up the job date ranges they find; if a CV writes dates in another format "
                   "(e.g. 'since 2019'), they find nothing and predict 0. That is where a learned model is still needed.")

    st.subheader("Browse the test CVs")
    preds = test_predictions()
    texts = clean_texts()
    df = preds["bert"].rename(columns={"y_pred": "bert"})
    for m in MODELS:
        if m != "bert":
            df[m] = preds[m]["y_pred"].to_numpy()
    df["bert_abs_error"] = (df["bert"] - df["y_true"]).abs()
    c1, c2, c3 = st.columns(3)
    occ = c1.multiselect("Occupation", utils.OCCUPATIONS, default=utils.OCCUPATIONS)
    years = c2.slider("True years", 0.0, 20.0, (0.0, 20.0), 0.5)
    err = c3.slider("BERT absolute error (years)", 0.0, float(np.ceil(df["bert_abs_error"].max() * 10) / 10),
                    (0.0, float(np.ceil(df["bert_abs_error"].max() * 10) / 10)), 0.05)
    view = df[df["occupation"].isin(occ) & df["y_true"].between(*years) & df["bert_abs_error"].between(*err)]
    st.caption(f"{len(view)} of {len(df)} test CVs")
    st.dataframe(view[["id", "occupation", "y_true"] + MODELS + ["bert_abs_error"]].round(2)
                 .sort_values("bert_abs_error", ascending=False), hide_index=True, width="stretch")
    if len(view):
        cv_id = st.selectbox("Show a CV", view.sort_values("bert_abs_error", ascending=False)["id"])
        row = df.set_index("id").loc[cv_id]
        show_predictions({m: row[m] for m in MODELS}, true=row["y_true"])
        c1, c2 = st.columns([3, 2])
        image_path = utils.RAW_IMAGES_DIR / f"{cv_id}.png"
        with c1:
            if image_path.exists():
                st.image(str(image_path), width="stretch")
            else:
                st.info("Image not available locally (raw images are not committed).")
        with c2:
            st.code(texts[cv_id], language=None)

    st.subheader("Good and bad predictions")
    top = table("bert_top10_errors.csv")
    best = df.nsmallest(400, "bert_abs_error")
    good = pd.concat([best[utils.to_band(best["y_true"]) == b].head(1) for b in utils.BAND_LABELS])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Good: BERT within a few days**, one per experience band")
        for _, r in good.iterrows():
            with st.expander(f"id {r['id']} · {r['occupation']} · true {r['y_true']:.2f} · BERT {r['bert']:.2f}"):
                st.markdown("Contiguous jobs with complete `YYYY.MM-YYYY.MM` dates that run up to the data snapshot: "
                            "the pattern BERT saw most often in training. The rules agree "
                            f"({r['rules']:.2f}).")
                st.code(texts[r["id"]], language=None)
    with c2:
        st.markdown("**Bad: BERT's 10 largest errors**: all career gaps")
        for _, r in top.iterrows():
            with st.expander(f"id {r['id']} · true {r['y_true']:.2f} · BERT {r['bert_pred']:.2f} · rules {r['rules_pred']:.2f}"):
                st.markdown(f"**{r['cause']}**: {r['explanation']}")
                st.code(r["text"], language=None)


# ------------------------------------------------------------------ layout

st.title("Predicting years of experience from CV images")
st.caption("PaddleOCR + fine-tuned DistilBERT on 4,000 synthetic PopResume CVs · all results read from output/")
tabs = st.tabs(["Problem & Methodology", "OCR & Exploratory Analysis", "Model Results", "Interactive Estimator"])
with tabs[0]:
    tab_methodology()
with tabs[1]:
    tab_ocr_eda()
with tabs[2]:
    tab_results()
with tabs[3]:
    tab_estimator()
