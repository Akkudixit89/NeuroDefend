from flask import Flask, render_template, request
from ml_model import train_models, predict_url, evaluate_model, get_metrics_summary
import os
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cache trained objects to avoid retraining every time
MODELS, VECTORIZER, ACCURACIES, URLS_DATA = None, None, None, None


# -----------------------
# Landing Page
# -----------------------
@app.route("/", methods=["GET"])
def index():
    """Landing page with 3 input bars and central Neuro Defend title."""
    return render_template("index.html")


# -----------------------
# Analyze Route
# -----------------------
@app.route("/analyze", methods=["POST"])
def analyze():
    """Handles URL / PE / APK input and displays dashboard results."""
    global MODELS, VECTORIZER, ACCURACIES, URLS_DATA

    # Train or reuse trained models
    if MODELS is None:
        MODELS, VECTORIZER, ACCURACIES, URLS_DATA = train_models()

    chosen_model = request.form.get("model_choice", "Random Forest")
    which_input = request.form.get("which_input", "url")
    prediction = None
    input_summary = ""
    uploaded_filename = None

    # URL analysis
    if which_input == "url":
        url_input = request.form.get("url_input", "").strip()
        if url_input:
            prediction = predict_url(url_input, MODELS[chosen_model], VECTORIZER)
            input_summary = f"URL scanned: {url_input}"

    # PE upload
    elif which_input == "pe":
        pe_file = request.files.get("pe_input")
        if pe_file and pe_file.filename != "":
            safe_name = f"{int(time.time())}_pe_{pe_file.filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            pe_file.save(save_path)
            uploaded_filename = safe_name
            prediction = "PE uploaded — analysis pending"
            input_summary = f"PE file: {pe_file.filename}"

    # APK upload
    elif which_input == "apk":
        apk_file = request.files.get("apk_input")
        if apk_file and apk_file.filename != "":
            safe_name = f"{int(time.time())}_apk_{apk_file.filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            apk_file.save(save_path)
            uploaded_filename = safe_name
            prediction = "APK uploaded — analysis pending"
            input_summary = f"APK file: {apk_file.filename}"

    # Evaluate model for dashboard
    f1, cm = evaluate_model(MODELS[chosen_model], VECTORIZER, URLS_DATA)
    metrics = get_metrics_summary(chosen_model, MODELS[chosen_model], ACCURACIES, f1, cm)

    return render_template(
        "dashboard.html",
        prediction=prediction,
        metrics=metrics,
        accuracies=ACCURACIES,
        cm=cm,
        input_summary=input_summary,
        uploaded_filename=uploaded_filename
    )


# -----------------------
# Run App
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)
