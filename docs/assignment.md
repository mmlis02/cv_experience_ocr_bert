Homework: Predicting Years of Experience from CV Images with PaddleOCR and BERT
Oct 18, 2026 · @jean

Course: Machine Learning / NLP / Applied AI Total score: 20 points — technical implementation 12 points, presentation video 8 points Deadline: Friday — 11:59 PM

Description
You will build an end-to-end document-AI system that reads a CV image, extracts its text with PaddleOCR, and predicts the candidate's years of work experience (0 to 20) with a fine-tuned BERT-based regression model.

This is deliberately not a keyword task. The number of years is rarely written on the CV. The model must infer it by connecting information spread across the document: the dates of each job, how many positions there are, how job titles progress, and when education ended. A recruiter does this in their head; your model must learn to do it from text.

You will compare BERT against simpler models that cannot make those connections (a median guess, TF-IDF and sentence embeddings) and measure, with numbers, how much the transformer actually helps.

Main Objective
Build a pipeline that:

Downloads a stratified sample of CV images from the PopResume dataset
Extracts the text of each CV with PaddleOCR and measures OCR quality
Cleans the text and removes any line that states the answer directly
Creates a train / validation / test split shared by every model
Trains three baselines: median predictor, TF-IDF + Ridge and sentence embeddings + Ridge
Fine-tunes one lightweight BERT-based model as a regressor
Evaluates all models with MAE, accuracy within ±k years, band accuracy and R²
Explains where BERT helps, where it fails, and whether the system would be useful in practice
You are given the dataset and the target. You are NOT given OCR settings, cleaning rules, truncation strategies or hyperparameters. Those decisions are part of the assignment and must be justified.

Dataset
All groups use PopResume (Yu, Park & Moon, EMNLP 2026; license CC BY 4.0): 60,884 synthetic resumes for 5 occupations, each released as text and as images with and without a synthetic profile photo. The resumes are generated from U.S. population statistics, so they contain no real person's data, and every resume comes with its true years of experience.

Occupation	Resumes in the dataset	CVs you sample
Registered nurses	17,632	800
Elementary / middle school teachers	16,188	800
Software developers	13,370	800
Accountants & auditors	7,784	800
Construction laborers	5,910	800
Total	60,884	4,000
Access
Create a Hugging Face account, open the dataset page and accept its access conditions (you must share contact information).
Create an access token and log in with huggingface-cli login.
Stream one occupation at a time instead of downloading all images (about 20 GB). Image shards are named data/images/<occupation>-NNNN.parquet:
from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files="hf://datasets/sumin-yu/PopResume/data/images/registered_nurses-*.parquet",
    split="train",
    streaming=True,
)
ds = ds.filter(lambda r: not r["has_photo"]).shuffle(seed=42, buffer_size=5000)
sample = list(ds.take(800))   # save each r["image"] as data/raw/images/<id>.png
Load the text config (load_dataset("sumin-yu/PopResume", "text")) and keep only the id and text of your sampled CVs. You will use it only to measure OCR quality.
Target variable
Target: exp_year, the candidate's years of work experience.
Range: clip to 0–20. Any value above 20 becomes 20 (read it as "20 or more"); clip predictions to the same range.
Check the cap: report how many sampled CVs sit at 20. If more than 15% do, you may raise the cap to 25 or 30, but you must justify it and use the same cap for every model.
Rules for using the data
Use only the no-photo images (has_photo == False). The photo adds nothing to the task and is a source of bias.
Model input must come from PaddleOCR on the image. The text column is a reference, not a feature: use it only to compute OCR error, never to train or test a model.
Never use age, sex, race, names, state or region as features. Age is especially forbidden: it is strongly tied to experience and would let the model cheat.
sex and race may be used only for the fairness check in Question 4.
Remove direct answers. Before modeling, read at least 50 random OCR texts. If a CV states total experience ("12 years of experience"), age or date of birth, mask that phrase with a regex and report how many CVs were affected. Job dates and graduation years stay: they are the evidence the model is supposed to reason from.
Sample with a fixed seed and save the list of sampled IDs so your results can be reproduced.
Compute budget
OCR is the slowest stage. Run it once, cache the output (one JSON per CV) and reuse it. A GPU (for example Google Colab) is recommended for OCR and BERT. If you cannot process 4,000 CVs, use at least 2,000 (400 per occupation) and explain the choice.

Expected Repository Structure
Create a repository named exactly cv_experience_ocr_bert with this structure:

cv_experience_ocr_bert/
│
├── app.py                          # Streamlit app (4 tabs)
├── README.md                       # Project explanation
├── requirements.txt                # Pinned versions, incl. paddlepaddle + paddleocr
│
├── src/
│   ├── download_data.py            # Stratified sampling and download from PopResume
│   ├── ocr_extraction.py           # PaddleOCR on each image, caching, confidence
│   ├── ocr_quality.py              # CER / WER against the reference text
│   ├── preprocessing.py            # Cleaning, reading order, masking of direct answers
│   ├── split.py                    # Stratified train / val / test split
│   ├── baseline_median.py          # Median predictor
│   ├── baseline_tfidf.py           # TF-IDF + Ridge
│   ├── baseline_embeddings.py      # Sentence embeddings + Ridge
│   ├── train_bert.py               # BERT fine-tuning as a regressor
│   ├── evaluation.py               # MAE, ±k accuracy, bands, R², bootstrap CIs
│   ├── visualization.py            # Figures
│   └── utils.py                    # Seeds, paths, helpers
│
├── data/
│   ├── raw/                        # Sampled images + metadata (not committed)
│   ├── ocr/                        # Cached PaddleOCR output
│   ├── processed/                  # Clean text per CV + target
│   └── splits/                     # train.csv, val.csv, test.csv (IDs + target)
│
├── models/
│   ├── baselines/
│   └── bert/
│
├── output/
│   ├── figures/
│   ├── tables/
│   └── metrics/
│
└── video/
    └── link.txt
Images and model weights must not be pushed to GitHub. Use .gitignore and explain in the README how to regenerate them.

Required Pipeline
One OCR output feeds four models trained and tested on the same split, so any difference in results comes from the model, not the data.

[embedded content: pipeline · OCR, split, four models, one evaluation]

Stage 1 — Text Extraction with PaddleOCR
Install PaddlePaddle first (CPU or GPU build, from the PaddlePaddle package index), then PaddleOCR 3.x. Pin both versions in requirements.txt.

from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
result = ocr.predict("data/raw/images/000123.png")[0]
texts, scores, boxes = result["rec_texts"], result["rec_scores"], result["rec_boxes"]
State which recognition model you used (for example PP-OCRv5_mobile or PP-OCRv5_server) and why.
Save, for every text line: the text, its confidence score and its bounding box (one JSON per CV in data/ocr/).
Rebuild the reading order from the boxes (top to bottom, left to right) and explain how you handle two-column layouts. Getting the order wrong can separate a job title from its dates.
Report: OCR time per CV, hardware used, mean confidence and any failed images.
Stage 2 — OCR Quality and Preprocessing
Measure OCR quality against the reference text with character error rate (CER) and word error rate (WER) (for example with jiwer), after normalizing both texts the same way. Report the distribution and the mean per occupation.
Check specifically how well dates are read (for example "2014 – 2019"). A misread digit changes the answer, so report the share of reference years that appear correctly in the OCR text.
Clean the text: fix broken lines and hyphenation, decide whether to drop low-confidence lines (justify any threshold), and remove the candidate's name, email, phone and address.
Apply the masking rule from the Dataset section and report how many CVs were affected.
Stage 3 — Train / Validation / Test Split
Split by CV ID: 70% / 15% / 15%, stratified by occupation and experience band, with a fixed seed.
Save the IDs in data/splits/. All models must use the same split.
The test set stays unseen until the final evaluation. All tuning uses the validation set.
Stage 4 — Baseline Models
All baselines predict a number and are tuned on the validation set only.

Baseline 0 — Median predictor. Always predict the median years of the training set. Every other model must beat it; otherwise it learned nothing.
Baseline 1 — TF-IDF + Ridge regression. Word n-grams (for example 1–2), optionally character n-grams. Decide and justify whether to keep numbers and years as tokens. Tune the regularization strength.
Baseline 2 — Sentence embeddings + Ridge regression. Encode each CV with a frozen pretrained encoder (for example all-MiniLM-L6-v2, bge-small-en-v1.5 or e5-small-v2), then fit Ridge on the vectors. Explain how you handle CVs longer than the encoder's limit (for example chunk and average). The encoder is not fine-tuned.
Optional bonus: a rule-based model that extracts date ranges with regular expressions and adds up the job durations (merging overlaps). It is an honest test of whether BERT does better than simple arithmetic.

Stage 5 — Fine-Tuning BERT as a Regressor
Fine-tune one lightweight transformer: DistilBERT, ModernBERT, MiniLM or DeBERTa-v3-small, with HuggingFace Transformers.
Use a single output: AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, problem_type="regression"). The loss is mean squared error; you may try Huber loss and compare.
Scale the target (for example divide by 20) during training and convert predictions back to years, clipped to 0–20.
CVs may exceed 512 tokens. Choose and justify a strategy: head truncation, head + tail, or chunking with averaged predictions. Say whether the dates you need survive your truncation.
Use early stopping on validation MAE. Report learning rate, batch size, epochs, max length, seed, training time and hardware.
Stage 6 — Evaluation and Error Analysis
Evaluate every model on the same test set with these metrics:

MAE (main metric): average number of years the prediction is off.
RMSE: like MAE but punishes large misses more.
Accuracy within ±k years: share of CVs whose prediction is within k years of the truth, for k = 1, 2 and 3.
Band accuracy and macro-F1: convert true and predicted values to bands 0–2, 3–5, 6–10, 11–20 and score them as a classification, with a confusion matrix.
R²: share of the variance in experience the model explains.
\text{MAE} = \frac{1}{n}\sum_{i=1}^{n} \lvert \hat{y}_i - y_i \rvert \qquad \text{Acc}_{\pm k} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}\left[\lvert \hat{y}_i - y_i \rvert \le k\right]
Fill this table in output/tables/model_comparison.csv:

Model	MAE (years)	RMSE (years)	Acc ±1	Acc ±2	Acc ±3	Band accuracy	Band macro-F1	R²	Training time	Inference time per CV
Median										
TF-IDF + Ridge										
Embeddings + Ridge										
Fine-tuned BERT										
You must also:

Quantify BERT's gain: the MAE reduction of BERT versus each baseline, with a 95% paired bootstrap confidence interval on the test set (at least 1,000 resamples). State whether the gain is statistically meaningful.
Plot predicted vs true years for each model, with the 45° line. Check whether models pull predictions toward the middle (overestimating juniors, underestimating seniors).
Report MAE by true experience band and by occupation.
Inspect the 10 largest BERT errors and group them by cause: OCR misread a date, wrong reading order, career gap, overlapping jobs, missing dates, truncation, or other.
Explain possible improvements.
Required Analytical Questions
Answer all four in the README and in the Streamlit app, supported by your own numbers.

Question 1 — What does BERT learn that the baselines cannot?
How much does BERT reduce MAE compared with the median, TF-IDF and embeddings, and is the gain statistically meaningful?
Where does the gain come from? Compare errors by experience band: does BERT mainly fix juniors, mid-career or seniors?
Run one probe: take 20 test CVs, shift every date by +5 years (or delete one job), and check whether each model's prediction moves the way a recruiter's would. What does this tell you about whether the model reads dates or just vocabulary?
Question 2 — How much does OCR noise cost?
How good is your OCR (CER, WER, share of years read correctly), and where does it fail?
Evaluate each trained model twice: on the OCR test text and on the reference text of the same CVs. How large is the gap for each model?
Which model is most robust to OCR errors, and why might that be?
Question 3 — Methodological sensitivity
Compare one baseline setup and one alternative setup for at least one of these choices, and report how every model's MAE changes:

Long-text strategy: head truncation vs head + tail vs chunking.
Numbers in TF-IDF: keeping vs removing digits and years.
Loss function for BERT: MSE vs Huber.
OCR cleaning: raw OCR vs cleaned OCR vs low-confidence lines removed.
Question 4 — Business value, fairness and ethics
Who could use an experience estimator (recruiting agencies, job boards, public employment services, HR teams screening thousands of CVs), and for what task? Pre-filling a profile is very different from deciding who gets an interview.
What does an error of ±2 or ±5 years cost in practice? Which kind of error is worse: overestimating or underestimating?
Fairness check: report BERT's MAE and its mean error (predicted minus true) separately by sex and by race. Does the model systematically under- or overestimate experience for any group? What could explain it, given that these attributes are not model inputs?
Discuss limitations: synthetic resumes, only 5 occupations, U.S. statistics, English only, and why such a system must never decide who is hired.
Technical Requirements
Required: PaddleOCR (with PaddlePaddle), HuggingFace Transformers and datasets, sentence-transformers, pandas, scikit-learn, matplotlib, seaborn, Streamlit.

Optional: PyTorch, accelerate, evaluate, jiwer (for CER/WER), plotly.

Streamlit Application
Score: 2 points. The app must contain exactly 4 tabs.

Tab 1 — Problem & Methodology
Objective and why experience is hard to read from a CV
Dataset, sampling, target definition and the masking rule
OCR setup and reading-order logic
The four models and the split
Limitations
Tab 2 — OCR & Exploratory Analysis
Distribution of years of experience, overall and by occupation (including how many sit at the cap)
OCR quality: CER/WER distribution and share of years read correctly
Side-by-side example: a CV image with its PaddleOCR boxes, and the extracted text
For each chart, one or two sentences on why it was chosen and what it shows
Tab 3 — Model Results
The model comparison table (Stage 6)
Predicted vs true scatter plot for each model, selectable
MAE by experience band and by occupation for all models side by side
Band confusion matrix for each model
BERT's MAE gain over each baseline with its confidence interval
The sensitivity experiment (Question 3) and the fairness check (Question 4)
Tab 4 — Interactive Experience Estimator
Upload a CV image (or a PDF, converted to an image): the app runs PaddleOCR and shows the extracted text
Predicted years of experience from all models side by side
Browse test CVs by occupation, true years and size of error
Examples of good and bad predictions with a short explanation
README.md Must Include
What does the project do, and for whom?
How were the CVs sampled (number per occupation, seed), and what does the experience distribution look like?
Which PaddleOCR model and settings were used, and how was reading order rebuilt?
What was the OCR quality (CER, WER, years read correctly)?
How was the text cleaned, and how many CVs needed masking?
How was the dataset split?
How were the three baselines built?
Which BERT model was used, how was regression set up, and how were long CVs handled?
Final metrics of all models and BERT's MAE gain with its confidence interval
Answers to the four analytical questions
Main limitations and possible business applications
How to reproduce the project step by step, from download to evaluation
How to run the Streamlit app
Explanatory Video
Score: 8 points. Maximum 5 minutes, presented as a pitch of the system, not a coding walkthrough. Do not show code line by line.

The video must explain:

Problem definition: why estimating experience from CVs matters and who needs it.
From image to text: why OCR is needed, what PaddleOCR extracts, and how well it reads dates.
Why this is hard: what a model must connect to infer years of experience, with one real example CV.
Four ways to guess: the intuition behind the median, TF-IDF, embeddings and fine-tuned BERT, in plain words.
Results: MAE and accuracy within ±2 years for each model, the predicted-vs-true plot, and one good and one bad prediction.
Is BERT worth it, and is it fair? The size of the gain versus its cost, the fairness check, and what the system must never be used for.
GitHub Workflow (Mandatory)
Do not work directly on main.
Create development branches, use descriptive commits, and merge through Pull Requests.
Example branches: feature/data-sampling, feature/paddleocr-extraction, feature/ocr-quality, feature/baselines, feature/bert-regression, feature/evaluation, feature/streamlit-dashboard.

Grading Rubric
Technical Implementation — 12 points
Criteria	Points
Data sampling and PaddleOCR extraction pipeline	2
OCR quality measurement, preprocessing and masking	2
Baselines: median, TF-IDF + Ridge, embeddings + Ridge	2
BERT regression fine-tuning pipeline	2
Evaluation, model comparison and error analysis	2
Streamlit app completeness	2
Presentation Video — 8 points
Clarity of the problem and business case, quality of the reasoning about OCR and model choice, honest interpretation of results, and use of the 5 minutes.

Checklist Before Submitting

Repository is named cv_experience_ocr_bert

Only no-photo images were used, sampled with a fixed seed

Model inputs come from PaddleOCR, not from the reference text

Age, sex, race, names and location were not used as features

Phrases stating total experience or age were masked and counted

OCR quality (CER, WER, years read correctly) is reported

Train / validation / test split exists and is shared by all models

Median, TF-IDF + Ridge and embeddings + Ridge baselines were trained and evaluated

BERT was fine-tuned as a regressor

Comparison table, predicted-vs-true plots and BERT's MAE gain with a confidence interval are included

All four analytical questions are answered, including the fairness check

Streamlit app contains exactly 4 tabs

No images or model weights are committed

Video link exists in video/link.txt

Work was done with branches and Pull Requests

Repository is reproducible from the README
Final Note
The lowest MAE is not the main goal. It is a valid result if TF-IDF comes close to BERT, or if a simple date-counting rule beats it. What matters is that you can show what each model actually uses to make its guess, measure what OCR noise costs, compare models fairly on the same split, and argue with numbers whether the more expensive model is worth deploying.

