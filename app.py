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
    st.info("All CV images, OCR boxes and texts shown here come from **PopResume**, a **synthetic** dataset "
            "(CC BY 4.0) generated from aggregate U.S. population statistics. The names, phone numbers, emails "
            "and employers on these CVs are invented and do not belong to real people.")
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
        "job title and its dates are read together. Colours mark full-width, left and right columns. "
        "The CV is synthetic (PopResume): its name and contact details are not those of a real person.")


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

    question_1()
    question_2()
    question_3()
    question_4()


PROBE_VERDICTS = {
    "Median": "Ignores the text: never moves. Reads nothing.",
    "TF-IDF + Ridge": "Moves on deletion (fewer job lines) but not by the job's length, and drops when years "
                      "shift: reads **vocabulary**, including year tokens as words.",
    "Embeddings + Ridge": "Same pattern: reacts to having less text, not to the dates' arithmetic: **vocabulary / "
                          "meaning**, not dates.",
    "Fine-tuned BERT": "Right direction on every deletion but only ~2/3 of the size and uncorrelated with the job's "
                       "length; drops ~1.5 years when dates shift: **partly dates, partly absolute-year and "
                       "job-count cues**. It has not learned the arithmetic.",
    "Date rules (bonus)": "Moves exactly as expected in both edits: **reads dates** like a recruiter.",
}


def question_1():
    st.subheader("Question 1 · What does BERT learn that the baselines cannot?")
    gain = table("bert_gain_bootstrap.csv").set_index("baseline")
    st.markdown("**How much does BERT reduce MAE, and is it significant?** Paired bootstrap, 2,000 resamples of the "
                "same 600 test CVs, 95% CI:")
    st.markdown("\n".join(
        f"- vs **{b}**: {r['mae_reduction']:+.2f} years [{r['ci95_low']:+.2f}, {r['ci95_high']:+.2f}]: {r['conclusion']}"
        for b, r in gain.iterrows()))
    figure(viz.plot_bert_gain(gain.reset_index()))

    st.markdown("**Where does the gain come from?** Test MAE by true experience band:")
    band = table("mae_by_band.csv").pivot(index="model", columns="band", values="mae")
    band = band.reindex(list(LABELS.values()))[utils.BAND_LABELS]
    st.dataframe(band.round(2), width="stretch")
    st.markdown(f"BERT's gain over TF-IDF is largest for **seniors** ({band.loc['TF-IDF + Ridge', '11-20']:.2f} → "
                f"{band.loc['Fine-tuned BERT', '11-20']:.2f} years) and **juniors** "
                f"({band.loc['TF-IDF + Ridge', '0-2']:.2f} → {band.loc['Fine-tuned BERT', '0-2']:.2f}). Those are the "
                "extremes where the baselines are pulled toward the middle (fitted slope: embeddings 0.86, TF-IDF "
                "0.92, BERT 0.99). Mid-career CVs gain too.")

    st.markdown("**Probe on 20 test CVs: does each model move like a recruiter?** The expected change comes from "
                "the date rules: delete one job → minus its duration; shift every date +5 years → 0 (durations do "
                "not change, and there are no 'Present' end dates).")
    probe = table("q1_probe_summary.csv")
    wide = probe.pivot(index="model", columns="experiment", values="mean_model_change").reindex(list(LABELS.values()))
    delete = probe[probe["experiment"] == "delete_one_job"].set_index("model")
    view = pd.DataFrame({
        "delete one job: expected": delete["mean_expected_change"],
        "delete one job: model moves": wide["delete_one_job"],
        "right direction": delete["share_moving_right_direction"],
        "corr. with job length": delete["corr_with_expected"],
        "shift +5 years: expected": 0.0,
        "shift +5 years: model moves": wide["shift_plus_5_years"],
        "verdict": pd.Series(PROBE_VERDICTS),
    }).reindex(list(LABELS.values()))
    st.dataframe(view.round(2), width="stretch")
    st.markdown("**Conclusion.** Only the rules read dates the way a recruiter does. TF-IDF and the embeddings react "
                "to vocabulary and text length. BERT is the best learned model, but its predictions also depend on "
                "absolute years (later years look like a recent start, because the data ends in 2023.01) and on "
                "the number of job lines, not on subtracting dates. That is why its 10 largest errors are all career "
                "gaps, where the span of years and the sum of job durations differ.")


def question_2():
    st.subheader("Question 2 · How much does OCR noise cost?")
    q = table("ocr_quality_by_occupation.csv").set_index("occupation").loc["all"]
    st.markdown(f"""
**How good is the OCR?** CER **{100 * q['cer_mean']:.2f}%**, WER **{100 * q['wer_mean']:.1f}%**. Only
**{q['pct_years_correct']:.1f}%** of reference years appear as is, and **{q['pct_years_correct_after_date_fix']:.3f}%**
after re-inserting the dash the OCR drops in date ranges (`2020.112023.01` → `2020.11-2023.01`). It fails almost
only there (98.5% of character errors are on job title + date lines), plus `i`→`l` in acronyms and dates cut off
in the image itself.
""")
    q2 = table("q2_ocr_vs_reference.csv")
    st.markdown("**Each clean-trained model evaluated twice:** on the OCR test text and on the reference text of "
                "the same CVs.")
    st.dataframe(q2, hide_index=True, width="stretch")
    q3 = table("q3_variant_mae.csv").set_index("model")
    st.markdown(f"""
**Which model is most robust, and why?** Among the learned models, **TF-IDF + Ridge** is the most robust.
- Its OCR-vs-reference gap is +0.002 years (not significant).
- On the unrepaired raw OCR it loses only {q3.loc['TF-IDF + Ridge', 'raw'] - q3.loc['TF-IDF + Ridge', 'clean']:+.3f}
  years.
- Its character n-grams are redundant: a misread or missing character changes a handful of n-grams out of hundreds,
  and the pieces of a glued date (`2020.`, `.11`, `2023`) still match the training vocabulary.

**BERT** is robust once dates are repaired (−0.007 years, slightly *better* on OCR text, which is what it was trained
on), but it is the **least** robust to unrepaired noise: {q3.loc['Fine-tuned BERT', 'raw'] - q3.loc['Fine-tuned BERT', 'clean']:+.2f}
years on raw text (Question 3). It needs well-separated dates to approximate the arithmetic. The **date rules** are
equally robust by design: their regex accepts a missing separator.
""")


def question_3():
    st.subheader("Question 3 · Methodological sensitivity: OCR cleaning")
    st.markdown("Each model is trained **and** tested on raw OCR (no repair), cleaned OCR, and cleaned OCR without "
                "lines below 0.97 confidence. BERT was trained on raw and clean only.")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(table("q3_variant_mae.csv"), hide_index=True, width="stretch")
    with c2:
        st.dataframe(table("q3_variant_bootstrap.csv"), hide_index=True, width="stretch")
    st.markdown("""
- **BERT** is the most sensitive model: 0.42 with raw vs 0.25 with clean (+0.17 years, CI [0.14, 0.20]).
- **Embeddings** lose 0.08 years with raw (significant). **TF-IDF** loses only +0.01 (not significant).
- **Dropping low-confidence lines** slightly hurts (TF-IDF +0.02, rules +0.015): those lines are correct, and some
  carry job dates.

The bootstrap covers test noise only. Each variant was trained once, and BERT's validation MAE moved by 0.05–0.2
years between epochs, so the size of the raw-vs-clean gap should be confirmed with more seeds.
""")


def question_4():
    st.subheader("Question 4 · Business value, fairness and ethics")
    cost = table("q4_error_costs.csv").set_index("model")
    counts = [c for c in cost.columns if c.startswith(("wrongly_", "errors_"))]
    # Plain dicts keep the counts as ints (a mixed-type row would turn them into floats)
    b = {**cost.loc["Fine-tuned BERT"].to_dict(), **cost.loc["Fine-tuned BERT", counts].astype(int).to_dict()}
    t = {**cost.loc["TF-IDF + Ridge"].to_dict(), **cost.loc["TF-IDF + Ridge", counts].astype(int).to_dict()}
    st.markdown(f"""
**Who could use it, and for what?**
- **Recruiting agencies and job boards:** pre-fill the "years of experience" field of a profile from an uploaded CV,
  for the candidate or a recruiter to confirm.
- **Public employment services:** the same, during registration.
- **HR teams screening thousands of CVs:** sort or route CVs for a human to read.

Pre-filling a field is low risk. **Deciding who gets an interview is not**: a wrong number becomes a silent
rejection.

**What does an error of ±2 or ±5 years cost?** (600 test CVs)
- **±2 years** hardly matters for a profile field, but it matters at hard requirements. At a **"5+ years required"**
  cut-off, BERT puts **{b['wrongly_rejected_5y'] + b['wrongly_passed_5y']} CVs** on the wrong side
  ({b['wrongly_passed_5y']} wrongly passed, {b['wrongly_rejected_5y']} wrongly rejected); TF-IDF puts
  {t['wrongly_rejected_5y'] + t['wrongly_passed_5y']} ({t['wrongly_passed_5y']} passed, {t['wrongly_rejected_5y']} rejected).
- BERT makes **{b['errors_over_2y']} errors above 2 years** (max {b['max_abs_error']:.1f}); TF-IDF makes {t['errors_over_2y']}.
- **±5 years** misplaces a candidate by a whole seniority level (junior vs senior): wrong interviews, wrong salary
  band. BERT makes {b['errors_over_5y']} such errors here, TF-IDF {t['errors_over_5y']},
  embeddings {int(cost.loc['Embeddings + Ridge', 'errors_over_5y'])} and the median
  {int(cost.loc['Median', 'errors_over_5y'])}. On messier real CVs they would be more frequent.

**Which error is worse?** In screening, **underestimating** is worse for the candidate: the rejection is invisible
and cannot be contested, and it falls hardest on people with career gaps or non-standard dates. **Overestimating**
mostly costs the employer an unsuitable interview. BERT overestimates {100 * b['share_overestimated']:.0f}% of the time,
so its threshold mistakes are mostly "wrongly passed", the less harmful kind. Any use should still show the
extracted dates next to the number and let a person correct it.
""")
    st.markdown("**Fairness check.** Sex and race are never model inputs, and names are removed.")
    fair = table("q4_fairness.csv")
    figure(viz.plot_fairness(fair))
    st.dataframe(fair, hide_index=True, width="stretch")
    st.markdown("""
- **MAE is the same in every group** (0.24–0.26). BERT slightly **overestimates** most groups: women +0.08, men
  +0.14, Asian/Pacific Islander +0.15, White +0.11. For Black candidates the mean error is −0.02, but n = 47.
- The sex intervals overlap; the largest gap is about 0.18 years (≈ 2 months). The date rules show no group
  difference.
- **Why it can happen without the attributes:** BERT's bias depends on career patterns, which correlate with group
  membership in this population-based data. It overestimates careers with a gap after the last job, and men have
  that pattern more often (19.9% vs 16.7%). Mean experience also differs (Asian/PI 7.5 vs White 6.3 years).
""")
    st.markdown("""
**Limitations, and why it must never decide on hiring.**
- **Synthetic CVs from a single, single-column template.** The target is exactly the sum of the listed job
  durations, so these numbers are an upper bound for real CVs (free-form dates, part-time and overlapping jobs,
  multi-column layouts, scans).
- **Narrow scope:** only **5 occupations**, generated from **U.S.** statistics, **English only**.
- **One training run** per model and variant.
- **Never a decision system.** Its errors concentrate on non-standard careers (gaps), the fairness check shows it
  can treat groups slightly differently without seeing sex or race, and a candidate cannot see or contest an
  automatic number. It may pre-fill or route; **a person must make every hiring decision**.
""")


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
