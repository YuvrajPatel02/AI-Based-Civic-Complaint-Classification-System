# AI-Based Civic Complaint Classification System

A full-stack ML system that classifies citizen civic complaints (Garbage, Road
Damage, Water Leakage, Drainage Blockage, Streetlight Fault) from text
descriptions and/or photos, assigns a priority level, and stores records in a
database — built from the project abstract "AI-Based Civic Complaint
Classification System Using NLP and Deep Learning."

## What's actually inside (please read before your viva)

This is a genuinely trained, end-to-end system, not an API wrapper:

| Component | Technique | Real training performed |
|---|---|---|
| Text classifier | TF-IDF (1-2 grams) + Logistic Regression | Yes — 800 samples, 80/20 split + 5-fold CV |
| Image classifier | CNN (MobileNetV2 architecture proposed in the abstract) | Yes — 1100 images, 80/20 stratified split |
| Priority assignment | Rule-based, per abstract's requirement | N/A (rules, not ML, by design) |
| Storage | SQLite | N/A |

**Two honest limitations you should know about and can explain in your viva:**

1. **Text dataset is template-generated, not scraped from a real complaints
   portal.** `src/generate_text_dataset.py` procedurally builds ~800
   grammatically varied complaint sentences from location/severity/category
   templates. This lets the model actually learn real n-gram patterns, but
   because each category uses fairly distinctive vocabulary ("pothole",
   "leak", "garbage", "drain", "streetlight"), the model reaches ~100% test
   accuracy on this dataset — that number will be lower on messier real-world
   complaint text with typos, mixed topics, or vague phrasing. For a stronger
   viva result, replace `data/complaints_text.csv` with real complaints (even
   50-100 scraped/anonymized real examples per category) and re-run
   `src/train_text_model.py`.

2. **Image dataset is synthetic, not real photographs**, and the model was
   trained **from scratch (random weights)**, not via true ImageNet transfer
   learning. This sandbox has no internet access to Keras's ImageNet weight
   servers or a real photo dataset, so:
   - `src/generate_image_dataset.py` procedurally draws simple images with
     class-distinctive colors/shapes/textures (see `data/images/`) purely so
     the pipeline can be trained and evaluated end-to-end.
   - `src/train_image_model.py` **first tries real MobileNetV2 transfer
     learning** (`weights="imagenet"`) exactly as proposed in the abstract.
     **When you run it yourself with normal internet access, it will
     automatically use real pretrained ImageNet weights** — no code change
     needed. It only falls back to a compact from-scratch CNN when the
     ImageNet weight download fails (which is what happened when this
     package was built, since this build sandbox blocks that domain).
   - Before you demo this for real, replace `data/images/<Category>/` with
     real photographs (a few hundred per class is a reasonable target) and
     re-run the training script with internet access enabled.

Both training scripts print which path was taken and save real metrics
(`models/text_metrics.json`, `models/image_metrics.json`) — check
`pretrained_imagenet_weights_used` in the latter to confirm which mode you're
in.

## Architecture

```
Citizen submits complaint (text and/or photo)
        │
        ▼
 ┌─────────────────┐     ┌──────────────────────┐
 │ TF-IDF + LogReg  │     │ CNN / MobileNetV2     │
 │ (text_model)     │     │ (image_model)         │
 └────────┬─────────┘     └──────────┬───────────┘
          │  category probabilities   │
          └───────────┬───────────────┘
                       ▼
            Fuse (average) if both given
                       ▼
              Final category + confidence
                       ▼
            Rule-based priority assignment
                       ▼
         Store in SQLite + generate complaint ID
                       ▼
              Flask UI shows result / dashboard
```

## Project structure

```
civic_complaint_classifier/
├── app.py                       # Flask application (routes)
├── requirements.txt             # Full deps (incl. TensorFlow for image model)
├── requirements-light.txt       # Text-only deps (no TensorFlow, for tight hosts)
├── Procfile                     # gunicorn start command (Render/Heroku style)
├── render.yaml                  # Render.com deployment blueprint
├── data/
│   ├── complaints_text.csv      # Generated text dataset (800 rows)
│   └── images/<Category>/       # Generated image dataset (1100 images)
├── models/                      # Trained model artifacts (see below)
├── src/
│   ├── generate_text_dataset.py
│   ├── generate_image_dataset.py
│   ├── train_text_model.py
│   ├── train_image_model.py
│   ├── predict.py               # Loads models, fuses predictions
│   └── priority.py              # Rule-based priority logic
├── database/
│   └── db.py                    # SQLite data access layer
├── templates/                   # Jinja2 HTML templates
└── static/
    ├── css/style.css
    └── uploads/                 # User-uploaded complaint photos
```

### Trained model artifacts in `models/`

| File | What it is |
|---|---|
| `tfidf_vectorizer.joblib` | Fitted TF-IDF vectorizer |
| `text_model.joblib` | Trained Logistic Regression classifier |
| `text_label_encoder.joblib` | Category label encoder for the text model |
| `text_metrics.json` | Real accuracy, CV scores, classification report, confusion matrix |
| `image_model.keras` | Trained CNN/MobileNetV2 model |
| `image_label_encoder.joblib` | Category label encoder for the image model |
| `image_metrics.json` | Real accuracy, training history, classification report, confusion matrix |

## Running locally

```bash
cd civic_complaint_classifier
pip install -r requirements.txt          # or requirements-light.txt for text-only
python app.py                            # runs on http://localhost:5000
```

Open http://localhost:5000, submit a complaint with text and/or a photo, and
you'll be redirected to a result page with your complaint ID, predicted
category, priority, and confidence. Visit `/dashboard` to see all complaints,
filter by category/priority/status, and update status.

## Retraining the models

```bash
# 1. Regenerate datasets (optional - already included)
python src/generate_text_dataset.py
python src/generate_image_dataset.py

# 2. Train
python src/train_text_model.py
python src/train_image_model.py
```

Both scripts overwrite the artifacts in `models/` and print/save real
metrics — no manual metric entry anywhere.

## Deploying to Render

1. Push this project to a GitHub repository.
2. On Render.com: **New +** → **Blueprint** → connect your repo (Render will
   detect `render.yaml`), or **New +** → **Web Service** manually with:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
3. **Memory note:** `tensorflow-cpu` is a large dependency. Render's free
   instance (512 MB RAM) may struggle to load both TensorFlow and the model.
   If the deploy fails or the service crashes/OOMs:
   - Use `requirements-light.txt` instead (rename to `requirements.txt` or
     point Render's build at it) and remove `models/image_model.keras` — the
     app automatically detects the missing image model and falls back to
     **text-only classification** without any code change.
   - Or upgrade to a paid Render instance with more RAM.
4. **Persistence note:** SQLite (`database/complaints.db`) is a file on disk.
   Render's free web services use an **ephemeral filesystem** — data is lost
   on restart/redeploy unless you attach a paid persistent disk (see the
   `disk:` block in `render.yaml`, which requires a paid plan). For a purely
   free-tier demo, treat complaint history as session-only, or swap in a free
   external database (e.g. Render's free PostgreSQL) for real persistence.

## Priority rules (editable in `src/priority.py`)

| Category | Priority | Rationale |
|---|---|---|
| Water Leakage | High | Wastage of potable water, possible flooding |
| Road Damage | High | Accident risk |
| Drainage Blockage | Medium | Health hazard, worsens in monsoon |
| Garbage | Medium | Health/hygiene hazard |
| Streetlight Fault | Low | Safety concern, not immediately hazardous |

Complaints classified with confidence below 55% are flagged for manual
review regardless of category, since low-confidence auto-classification may
be unreliable.

## Tech stack

Python, Flask, gunicorn, scikit-learn, TensorFlow/Keras, Pandas, NumPy,
Pillow, joblib, SQLite, HTML/CSS (Jinja2 templates).

## Suggested next steps for a stronger submission

- Replace the synthetic image dataset with real photographs of civic issues.
- Replace/augment the templated text dataset with real (anonymized) municipal
  complaint records.
- Retrain both models with internet access so the image model uses real
  ImageNet-pretrained MobileNetV2 weights (true transfer learning, as
  proposed in the abstract) instead of the from-scratch fallback CNN.
- Add authentication for the dashboard (currently open to anyone with the URL).
- Add a learned fusion layer instead of simple probability averaging if you
  want to improve on multimodal combination.
