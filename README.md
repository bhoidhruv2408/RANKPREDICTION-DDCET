# DDQuest – DDCET College Predictor AI

An AI-powered counseling platform for DDCET (Diploma to Degree Common Entrance Test) students in Gujarat.

## Features
- **Rank Prediction** — Random Forest model trained on real 2022–2024 DDCET data (R² = 0.9999)
- **College Matching** — Compares predicted rank with 2024 ACPC cutoff data
- **Safe / Target / Dream** buckets with admission chance %
- **Analytics Dashboard** — Charts for marks vs rank trends, category split, branch demand
- **Admin Panel** — Upload new datasets, retrain model in one click
- **Filters** — Search, city, government/private, branch

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the ML model
```bash
python train_models.py
```
This creates `model/rank_model.pkl` and `model/label_encoder.pkl`.

### 3. Run the app
```bash
python app.py
```
Visit http://localhost:5000

## Folder Structure
```
ddcet_predictor/
├── app.py                  ← Flask backend (all routes)
├── train_models.py         ← ML training script
├── requirements.txt
├── dataset/
│   ├── marks_vs_rank.csv   ← Marks vs rank data (2022–2024)
│   └── ddcet_cutoff.csv    ← College cutoff data (2022–2024)
├── model/
│   ├── rank_model.pkl      ← Trained RandomForestRegressor
│   └── label_encoder.pkl   ← Category encoder
├── templates/
│   ├── index.html          ← Home / prediction form
│   ├── result.html         ← Results page
│   ├── analytics.html      ← Charts page
│   └── admin.html          ← Admin dashboard
└── static/
    ├── style.css
    ├── script.js
    └── charts.js
```

## Categories
| Code | Meaning |
|------|---------|
| OP   | Open / General |
| EW   | EWS (Economically Weaker Section) |
| SC   | Scheduled Caste |
| SE   | SEBC |
| ST   | Scheduled Tribe |
| TF   | Tuition Fee Waiver |

## Prediction Logic
- **Safe**: Your predicted rank ≤ 80% of closing rank
- **Target**: Your rank is within ±10% of closing rank
- **Dream**: Your rank is within 135% of closing rank

## Deployment (Production)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:8000 app:app
```

## Adding New Year Data
1. Append rows to `dataset/marks_vs_rank.csv` and `dataset/ddcet_cutoff.csv`
2. Go to Admin panel → Retrain Model
3. Done!

## Tech Stack
- **Backend**: Python Flask
- **ML**: scikit-learn RandomForestRegressor
- **Frontend**: HTML5 / CSS3 / Vanilla JS
- **Charts**: Chart.js (CDN)
- **Data**: Pandas / NumPy
