# from flask import Flask, render_template, request
# from ml_model import train_models, predict_url, evaluate_model, get_metrics_summary
# import os
# import time

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = "uploads"
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Cache trained objects to avoid retraining every time
# MODELS, VECTORIZER, ACCURACIES, URLS_DATA = None, None, None, None


# # -----------------------
# # Landing Page
# # -----------------------
# @app.route("/", methods=["GET"])
# def index():
#     """Landing page with 3 input bars and central Neuro Defend title."""
#     return render_template("index.html")


# # -----------------------
# # Analyze Route
# # -----------------------
# @app.route("/analyze", methods=["POST"])
# def analyze():
#     """Handles URL / PE / APK input and displays dashboard results."""
#     global MODELS, VECTORIZER, ACCURACIES, URLS_DATA

#     # Train or reuse trained models
#     if MODELS is None:
#         MODELS, VECTORIZER, ACCURACIES, URLS_DATA = train_models()

#     chosen_model = request.form.get("model_choice", "Random Forest")
#     which_input = request.form.get("which_input", "url")
#     prediction = None
#     input_summary = ""
#     uploaded_filename = None

#     # URL analysis
#     if which_input == "url":
#         url_input = request.form.get("url_input", "").strip()
#         if url_input:
#             prediction = predict_url(url_input, MODELS[chosen_model], VECTORIZER)
#             input_summary = f"URL scanned: {url_input}"

#     # PE upload
#     elif which_input == "pe":
#         pe_file = request.files.get("pe_input")
#         if pe_file and pe_file.filename != "":
#             safe_name = f"{int(time.time())}_pe_{pe_file.filename}"
#             save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
#             pe_file.save(save_path)
#             uploaded_filename = safe_name
#             prediction = "PE uploaded — analysis pending"
#             input_summary = f"PE file: {pe_file.filename}"

#     # APK upload
#     elif which_input == "apk":
#         apk_file = request.files.get("apk_input")
#         if apk_file and apk_file.filename != "":
#             safe_name = f"{int(time.time())}_apk_{apk_file.filename}"
#             save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
#             apk_file.save(save_path)
#             uploaded_filename = safe_name
#             prediction = "APK uploaded — analysis pending"
#             input_summary = f"APK file: {apk_file.filename}"

#     # Evaluate model for dashboard
#     f1, cm = evaluate_model(MODELS[chosen_model], VECTORIZER, URLS_DATA)
#     metrics = get_metrics_summary(chosen_model, MODELS[chosen_model], ACCURACIES, f1, cm)

#     return render_template(
#         "dashboard.html",
#         prediction=prediction,
#         metrics=metrics,
#         accuracies=ACCURACIES,
#         cm=cm,
#         input_summary=input_summary,
#         uploaded_filename=uploaded_filename
#     )


# # -----------------------
# # Run App
# # -----------------------
# if __name__ == "__main__":
#     app.run(debug=True)









from flask import Flask, render_template, request
from ml_model import train_models, predict_url, evaluate_model, get_metrics_summary
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# -----------------------
# Flask App Setup
# -----------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Cache to avoid retraining models each time
MODELS, VECTORIZER, ACCURACIES, URLS_DATA = None, None, None, None


# -----------------------
# Visualization Functions
# -----------------------

def plot_performance_bar(metrics):
    """Creates a bar chart for performance metrics (Accuracy, F1, etc.)."""
    clean_metrics = {}

    for k, v in metrics.items():
        if isinstance(v, (list, tuple)):
            flat = []
            for item in v:
                if isinstance(item, (list, tuple)):
                    flat.extend(item)
                elif isinstance(item, (int, float)):
                    flat.append(item)
            if len(flat) > 0:
                clean_metrics[k] = sum(flat) / len(flat)
        elif isinstance(v, (int, float)):
            clean_metrics[k] = v

    if not clean_metrics:
        clean_metrics = {"No Data": 0}

    keys = list(clean_metrics.keys())
    values = list(clean_metrics.values())

    plt.figure(figsize=(6, 4))
    sns.barplot(x=keys, y=values, palette="mako")
    plt.title("📊 Model Performance Metrics", fontsize=12)
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    encoded = base64.b64encode(img.getvalue()).decode('utf-8')
    plt.close()
    return encoded


def plot_accuracy_pie(accuracies):
    """Creates a pie chart of model accuracy distribution."""
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = list(accuracies.keys())
    sizes = list(accuracies.values())

    ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Set3.colors,
        shadow=True
    )
    ax.set_title("🎯 Accuracy Distribution", fontsize=12)

    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return encoded


# -----------------------
# Routes
# -----------------------

@app.route("/", methods=["GET"])
def index():
    """Landing page (input form for URL, PE, or APK)."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    """Handles URL / PE / APK input and displays the dynamic dashboard."""
    global MODELS, VECTORIZER, ACCURACIES, URLS_DATA

    # Train models only once (cached for reuse)
    if MODELS is None:
        MODELS, VECTORIZER, ACCURACIES, URLS_DATA = train_models()

    chosen_model = request.form.get("model_choice", "Random Forest")
    which_input = request.form.get("which_input", "url")
    prediction, input_summary, uploaded_filename = None, "", None

    # -----------------------
    # Handle Input Types
    # -----------------------
    if which_input == "url":
        url_input = request.form.get("url_input", "").strip()
        if url_input:
            prediction = predict_url(url_input, MODELS[chosen_model], VECTORIZER)
            input_summary = f"URL scanned: {url_input}"

    elif which_input == "pe":
        pe_file = request.files.get("pe_input")
        if pe_file and pe_file.filename != "":
            safe_name = f"{int(time.time())}_pe_{pe_file.filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            pe_file.save(save_path)
            uploaded_filename = safe_name
            prediction = "🧩 PE uploaded — analysis pending"
            input_summary = f"PE file: {pe_file.filename}"

    elif which_input == "apk":
        apk_file = request.files.get("apk_input")
        if apk_file and apk_file.filename != "":
            safe_name = f"{int(time.time())}_apk_{apk_file.filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_name)
            apk_file.save(save_path)
            uploaded_filename = safe_name
            prediction = "📦 APK uploaded — analysis pending"
            input_summary = f"APK file: {apk_file.filename}"

    # -----------------------
    # Evaluate & Visualize
    # -----------------------
    f1, cm, precision, recall = evaluate_model(MODELS[chosen_model], VECTORIZER, URLS_DATA)
    metrics = get_metrics_summary(chosen_model, MODELS[chosen_model], ACCURACIES, f1, cm, precision, recall)

    # Generate charts
    accuracy_chart = plot_accuracy_pie(ACCURACIES)
    performance_chart = plot_performance_bar(metrics)

    # Render dashboard
    return render_template(
        "dashboard.html",
        prediction=prediction,
        metrics=metrics,
        accuracies=ACCURACIES,
        input_summary=input_summary,
        uploaded_filename=uploaded_filename,
        accuracy_chart=accuracy_chart,
        performance_chart=performance_chart,
        cm=cm
    )


@app.route("/_debug_css", methods=["GET"])
def _debug_css():
    """Debug helper: return the static CSS file contents directly from disk.

    This bypasses Flask's static file handler and is only used for debugging.
    """
    css_path = os.path.join(app.root_path, 'static', 'style.css')
    try:
        with open(css_path, 'r', encoding='utf-8') as f:
            content = f.read()
        from flask import Response
        return Response(content, mimetype='text/css')
    except Exception as e:
        return (f"Error reading {css_path}: {e}", 500)


# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
