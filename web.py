from flask import Flask, render_template
from ml_model import train_models, get_all_predictions  # updated import

app = Flask(__name__)

@app.route('/')
def index():
    # Train all models (LogReg, RandomForest, XGBoost)
    models, vectorizer, accuracies, urls_data = train_models()

    # Get predictions for ALL dataset URLs
    results = get_all_predictions(models, vectorizer, urls_data)

    # No need to repackage again, results already has [(url, label)] format
    return render_template("index.html", accuracies=accuracies, results=results)

if __name__ == '__main__':
    app.run(debug=True)
