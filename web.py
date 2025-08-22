from flask import Flask, render_template
from ml_model import train_model, get_test_predictions  # uses your code.py logic

app = Flask(__name__)

@app.route('/')
def index():
    # Train the model and get vectorizer and accuracy
    model, vectorizer, accuracy = train_model()

    # Get test URLs and predictions
    test_urls, predictions = get_test_predictions(model, vectorizer)

    # Combine URLs and predictions for the HTML template
    results = list(zip(test_urls, predictions))

    return render_template('index.html', accuracy=round(accuracy * 100, 2), results=results)

if __name__ == '__main__':
    app.run(debug=True)
