"""
DDQuest – DDCET College Predictor AI — Flask Backend

KEY FIX: ddcet_cutoff.csv stores MARKS (0-200) in opening_rank/closing_rank columns.
         We compare student's MARKS directly against cutoff marks — NOT rank vs rank.

Routes:
  GET  /                → Home page
  POST /predict-rank    → Predict rank from marks, then show colleges
  POST /predict-college → Predict colleges from marks (AJAX)
  GET  /analytics       → Analytics page
  GET  /admin           → Admin panel
  POST /upload          → Upload new dataset
  POST /retrain         → Retrain model
  GET  /api/cutoff-data → Returns cutoff data as JSON
"""

import os, pickle, json
import pandas as pd
import numpy as np
from flask import (Flask, render_template, request,
                   jsonify, redirect, url_for, flash)
from werkzeug.utils import secure_filename

# ── App setup ──────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "ddquest_secret_2024"

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "model")
DATA_DIR     = os.path.join(BASE_DIR, "dataset")
UPLOAD_DIR   = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

CURRENT_YEAR = 2024

# ── Load model & encoder ───────────────────────────────────────────────────
def load_model():
    with open(os.path.join(MODEL_DIR, "rank_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
        le = pickle.load(f)
    return model, le

rank_model, label_encoder = None, None
try:
    rank_model, label_encoder = load_model()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"[WARN] Model not loaded: {e}")

# ── Load cutoff data ───────────────────────────────────────────────────────
def load_cutoff():
    df = pd.read_csv(os.path.join(DATA_DIR, "ddcet_cutoff.csv"))
    # NOTE: opening_rank and closing_rank in the CSV are actually MARKS (0-200 scale)
    # They represent the marks of the first and last student admitted
    # opening_rank = marks of best student admitted (highest marks)
    # closing_rank = marks of last/weakest student admitted (lowest marks = cutoff)
    df["opening_rank"] = pd.to_numeric(df["opening_rank"], errors="coerce").fillna(0).astype(float)
    df["closing_rank"] = pd.to_numeric(df["closing_rank"], errors="coerce").fillna(0).astype(float)
    # Ensure opening >= closing (opening = topper marks, closing = cutoff marks)
    # If data has them swapped (lower value first), fix it
    mask = df["opening_rank"] < df["closing_rank"]
    df.loc[mask, ["opening_rank", "closing_rank"]] = df.loc[mask, ["closing_rank", "opening_rank"]].values
    return df

# ── Rank prediction helper ─────────────────────────────────────────────────
def predict_rank(marks: float, category: str, year: int = CURRENT_YEAR):
    """
    Returns (best_rank, avg_rank, worst_rank) as integers.
    Uses ML model when available; falls back to linear approximation.
    """
    if not MODEL_LOADED:
        # Fallback linear approximation: 190 marks → rank ~1, 100 marks → rank ~5000
        base = max(1, int(17000 - marks * 87))
        best  = max(1, base - int(base * 0.12))
        worst = base + int(base * 0.18)
        return best, base, worst

    known = list(label_encoder.classes_)
    cat   = category if category in known else "OP"
    cat_enc = label_encoder.transform([cat])[0]

    X       = pd.DataFrame([[year, marks, cat_enc]], columns=["year", "marks", "category_enc"])
    avg_rank = max(1, int(rank_model.predict(X)[0]))

    X_best  = pd.DataFrame([[year, marks + 1.5, cat_enc]], columns=["year", "marks", "category_enc"])
    X_worst = pd.DataFrame([[year, marks - 2.0, cat_enc]], columns=["year", "marks", "category_enc"])

    best_rank  = max(1, int(rank_model.predict(X_best)[0]))
    worst_rank = max(1, int(rank_model.predict(X_worst)[0]))

    best_rank  = min(best_rank, avg_rank)
    worst_rank = max(avg_rank, worst_rank)

    return best_rank, avg_rank, worst_rank


# ── College prediction helper ──────────────────────────────────────────────
def predict_colleges(marks: float, category: str,
                     branch: str = None, city: str = None,
                     college_type: str = None):
    """
    Returns {"safe": [...], "target": [...], "dream": [...]}

    HOW MATCHING WORKS:
    The cutoff CSV stores MARKS (not ranks) in opening_rank/closing_rank columns:
      opening_rank = marks of the first/best student admitted (e.g. 165)
      closing_rank = marks of the last/weakest student admitted (e.g. 120) ← actual cutoff

    A student with `marks` M is eligible if M >= closing_rank (their marks meet cutoff).

    Buckets:
      SAFE   → M >= opening_rank * 0.97   (near or above the topper marks)
      TARGET → closing_rank <= M < opening_rank * 0.97
      DREAM  → closing_rank * 0.88 <= M < closing_rank  (slightly below cutoff, stretch)
    """
    df = load_cutoff()
    df = df[df["year"] == CURRENT_YEAR].copy()

    # ── Category filter with OP fallback ──
    cat_df = df[df["category"] == category].copy()
    if cat_df.empty:
        cat_df = df[df["category"] == "OP"].copy()

    # ── Optional filters (only narrow if data exists for that filter) ──
    if branch and branch.strip().lower() not in ("", "all"):
        mask = cat_df["branch"].str.contains(branch, case=False, na=False)
        if mask.any():
            cat_df = cat_df[mask]

    if city and city.strip().lower() not in ("", "all"):
        mask = cat_df["city"].str.contains(city, case=False, na=False)
        if mask.any():
            cat_df = cat_df[mask]

    if college_type and college_type.strip().lower() not in ("", "all"):
        mask = cat_df["college_type"].str.contains(college_type, case=False, na=False)
        if mask.any():
            cat_df = cat_df[mask]

    results = {"safe": [], "target": [], "dream": []}

    for _, row in cat_df.iterrows():
        open_marks  = float(row["opening_rank"])   # marks of best student admitted
        close_marks = float(row["closing_rank"])   # cutoff marks (min marks to get in)

        if close_marks <= 0:
            continue

        # ── Bucket assignment based on marks comparison ──
        if marks >= open_marks * 0.97:
            # Student marks ≈ or above topper marks → very safe
            margin = marks - close_marks
            chance = min(97, int(85 + min(12, margin / max(1, open_marks - close_marks) * 15)))
            bucket = "safe"
        elif marks >= close_marks:
            # Student marks between cutoff and topper → target
            span = max(1, open_marks * 0.97 - close_marks)
            pos  = marks - close_marks
            chance = max(40, int(40 + (pos / span) * 40))
            bucket = "target"
        elif marks >= close_marks * 0.88:
            # Student marks slightly below cutoff → dream (stretch)
            span = max(1, close_marks - close_marks * 0.88)
            pos  = marks - close_marks * 0.88
            chance = max(8, int(8 + (pos / span) * 27))
            bucket = "dream"
        else:
            continue  # Too far below cutoff — skip

        results[bucket].append({
            "college_name"  : row["college_name"],
            "branch"        : row["branch"],
            "city"          : row["city"],
            "college_type"  : row["college_type"],
            "opening_marks" : int(open_marks),
            "closing_marks" : int(close_marks),
            "chance"        : chance,
            "category"      : row["category"],
        })

    # Sort: safe by opening_marks desc, target/dream by closing_marks desc
    results["safe"].sort(key=lambda x: -x["opening_marks"])
    results["target"].sort(key=lambda x: -x["closing_marks"])
    results["dream"].sort(key=lambda x: -x["closing_marks"])

    # Cap at 20 per bucket
    for k in results:
        results[k] = results[k][:20]

    return results


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict-rank", methods=["POST"])
def predict_rank_route():
    try:
        marks_str = request.form.get("marks", "").strip()
        if not marks_str:
            flash("Please enter your marks.", "error")
            return redirect(url_for("index"))

        marks    = float(marks_str)
        category = request.form.get("category", "OP").upper().strip()
        branch   = request.form.get("branch", "all")
        city     = request.form.get("city", "all")
        col_type = request.form.get("college_type", "all")

        if not (0 <= marks <= 200):
            flash("Marks must be between 0 and 200.", "error")
            return redirect(url_for("index"))

        best, avg, worst = predict_rank(marks, category)
        colleges = predict_colleges(marks, category, branch, city, col_type)

        total_colleges = (len(colleges["safe"]) +
                          len(colleges["target"]) +
                          len(colleges["dream"]))

        return render_template(
            "result.html",
            marks      = marks,
            category   = category,
            branch     = branch,
            city       = city,
            col_type   = col_type,
            best_rank  = best,
            avg_rank   = avg,
            worst_rank = worst,
            colleges   = colleges,
            total      = total_colleges
        )
    except ValueError:
        flash("Invalid marks value. Please enter a valid number.", "error")
        return redirect(url_for("index"))
    except Exception as e:
        flash(f"Prediction error: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/predict-college", methods=["POST"])
def predict_college_route():
    data = request.get_json() or {}
    try:
        marks = float(data.get("marks", 100))
    except (ValueError, TypeError):
        marks = 100.0
    category   = str(data.get("category", "OP")).upper().strip()
    branch     = data.get("branch", "all")
    city       = data.get("city", "all")
    col_type   = data.get("college_type", "all")

    try:
        colleges = predict_colleges(marks, category, branch, city, col_type)
        return jsonify({"status": "ok", "colleges": colleges})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/analytics")
def analytics():
    try:
        df        = pd.read_csv(os.path.join(DATA_DIR, "marks_vs_rank.csv"))
        cutoff_df = load_cutoff()

        scatter_data = {}
        for yr in df["year"].unique():
            sub = df[(df["year"] == yr) & (df["category"] == "OP")]
            n   = min(120, len(sub))
            if n > 0:
                sample = sub.sample(n, random_state=1)
                scatter_data[int(yr)] = sample[["marks", "rank"]].values.tolist()

        cat_counts    = df[df["year"] == CURRENT_YEAR]["category"].value_counts().to_dict()
        branch_counts = (cutoff_df[cutoff_df["year"] == CURRENT_YEAR]["branch"]
                         .value_counts().head(10).to_dict())
        type_counts   = (cutoff_df[cutoff_df["year"] == CURRENT_YEAR]["college_type"]
                         .value_counts().to_dict())
        city_counts   = (cutoff_df[cutoff_df["year"] == CURRENT_YEAR]["city"]
                         .value_counts().head(8).to_dict())

        return render_template(
            "analytics.html",
            scatter_data  = json.dumps(scatter_data),
            cat_counts    = json.dumps(cat_counts),
            branch_counts = json.dumps(branch_counts),
            type_counts   = json.dumps(type_counts),
            city_counts   = json.dumps(city_counts)
        )
    except Exception as e:
        flash(f"Analytics error: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/admin")
def admin():
    try:
        df     = pd.read_csv(os.path.join(DATA_DIR, "marks_vs_rank.csv"))
        cutoff = load_cutoff()
        stats  = {
            "rank_rows"   : len(df),
            "cutoff_rows" : len(cutoff),
            "years"       : sorted(df["year"].unique().tolist(), reverse=True),
            "categories"  : sorted(df["category"].unique().tolist()),
            "colleges"    : cutoff["college_name"].nunique(),
            "model_loaded": MODEL_LOADED
        }
    except Exception as e:
        flash(f"Admin load error: {str(e)}", "error")
        stats = {"rank_rows": 0, "cutoff_rows": 0, "years": [],
                 "categories": [], "colleges": 0, "model_loaded": MODEL_LOADED}
    return render_template("admin.html", stats=stats)


@app.route("/upload", methods=["POST"])
def upload():
    file_type = request.form.get("file_type", "rank")
    f = request.files.get("file")
    if not f or f.filename == "":
        flash("No file selected.", "error")
        return redirect(url_for("admin"))

    fname = secure_filename(f.filename)
    if not fname.lower().endswith(".csv"):
        flash("Only CSV files are accepted.", "error")
        return redirect(url_for("admin"))

    dest      = ("marks_vs_rank.csv" if file_type == "rank" else "ddcet_cutoff.csv")
    save_path = os.path.join(DATA_DIR, dest)

    try:
        f.save(save_path)
        test_df  = pd.read_csv(save_path, nrows=2)
        required = (["year", "marks", "category", "rank"]
                    if file_type == "rank"
                    else ["year", "college_name", "branch", "category",
                          "opening_rank", "closing_rank", "city", "college_type"])
        missing = [c for c in required if c not in test_df.columns]
        if missing:
            flash(f"❌ Missing required columns: {', '.join(missing)}", "error")
            return redirect(url_for("admin"))
        flash(f"✅ {dest} updated successfully!", "success")
    except Exception as e:
        flash(f"❌ Upload failed: {e}", "error")

    return redirect(url_for("admin"))


@app.route("/retrain", methods=["POST"])
def retrain():
    import subprocess
    try:
        result = subprocess.run(
            ["python", os.path.join(BASE_DIR, "train_models.py")],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            flash(f"❌ Training error:\n{result.stderr}", "error")
            return redirect(url_for("admin"))
        global rank_model, label_encoder, MODEL_LOADED
        rank_model, label_encoder = load_model()
        MODEL_LOADED = True
        flash("✅ Model retrained successfully!\n" + result.stdout, "success")
    except subprocess.TimeoutExpired:
        flash("❌ Retraining timed out (>120s).", "error")
    except Exception as e:
        flash(f"❌ Retraining failed: {e}", "error")
    return redirect(url_for("admin"))


@app.route("/api/cutoff-data")
def api_cutoff():
    try:
        cutoff = load_cutoff()
        return jsonify(cutoff.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
