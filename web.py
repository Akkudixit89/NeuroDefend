from flask import Flask, render_template, request, redirect, url_for
from ml_model import train_models, predict_url, evaluate_model
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# global variables to store trained models
MODELS, VECTORIZER, ACCURACIES, URLS_DATA = None, None, None, None


@app.route("/", methods=["GET", "POST"])
def index():
    global MODELS, VECTORIZER, ACCURACIES, URLS_DATA

    prediction = None
    chosen_model = None
    f1 = None
    cm = None

    if request.method == "POST":
        # train/retrain models
        MODELS, VECTORIZER, ACCURACIES, URLS_DATA = train_models()

        # get selected model
        chosen_model = request.form.get("model_choice", "Random Forest")

        # URL input
        input_url = request.form.get("url_input")
        if input_url:
            prediction = predict_url(input_url, MODELS[chosen_model], VECTORIZER)

        # APK / PE uploads (stub — real feature extraction required)
        if "apk_input" in request.files:
            apk_file = request.files["apk_input"]
            if apk_file and apk_file.filename != "":
                path = os.path.join(app.config['UPLOAD_FOLDER'], apk_file.filename)
                apk_file.save(path)
                prediction = "APK uploaded (feature extraction needed)"

        if "pe_input" in request.files:
            pe_file = request.files["pe_input"]
            if pe_file and pe_file.filename != "":
                path = os.path.join(app.config['UPLOAD_FOLDER'], pe_file.filename)
                pe_file.save(path)
                prediction = "PE file uploaded (feature extraction needed)"

        # evaluate model (f1 + confusion matrix)
        f1, cm = evaluate_model(MODELS[chosen_model], VECTORIZER, URLS_DATA)

    return render_template(
        "index.html",
        accuracies=ACCURACIES,
        prediction=prediction,
        model_choice=chosen_model,
        f1=f1,
        cm=cm
    )


if __name__ == "__main__":
    app.run(debug=True)
