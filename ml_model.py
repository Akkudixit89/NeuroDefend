import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------
# Tokenizer for URLs
# -----------------------
def makeTokens(f):
    """Custom tokenizer for splitting URLs into tokens."""
    tkns_BySlash = str(f.encode('utf-8')).split('/')
    total_Tokens = []
    for i in tkns_BySlash:
        tokens = str(i).split('-')
        tkns_ByDot = []
        for j in range(0, len(tokens)):
            temp_Tokens = str(tokens[j]).split('.')
            tkns_ByDot = tkns_ByDot + temp_Tokens
        total_Tokens = total_Tokens + tokens + tkns_ByDot
    total_Tokens = list(set(total_Tokens))
    if 'com' in total_Tokens:
        total_Tokens.remove('com')
    return total_Tokens


# -----------------------
# Train models
# -----------------------
def train_models():
    """Train Random Forest and XGBoost models on the dataset."""
    urls_data = pd.read_csv("url_features.csv")

    y = urls_data["malicious"].astype(int)
    url_list = urls_data["URL"]

    vectorizer = TfidfVectorizer(tokenizer=makeTokens)
    X = vectorizer.fit_transform(url_list)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)

    # XGBoost
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_acc = xgb_model.score(X_test, y_test)

    models = {
        "Random Forest": rf,
        "XGBoost": xgb_model
    }
    accuracies = {
        "Random Forest": round(rf_acc * 100, 2),
        "XGBoost": round(xgb_acc * 100, 2)
    }

    return models, vectorizer, accuracies, urls_data


# -----------------------
# Predict single URL safely
# -----------------------
def predict_url(url, model, vectorizer):
    """Predict whether a single URL is benign or malicious."""
    if not url.startswith("http"):
        url = "http://" + url
    X = vectorizer.transform([url])
    pred = model.predict(X)[0]
    return "Malicious" if pred == 1 else "Benign"


# -----------------------
# Evaluate model
# -----------------------
def evaluate_model(model, vectorizer, urls_data):
    """Evaluate model on full dataset to compute F1 and confusion matrix."""
    y = urls_data["malicious"].astype(int)
    url_list = urls_data["URL"]
    X = vectorizer.transform(url_list)
    preds = model.predict(X)

    f1 = round(f1_score(y, preds, average='weighted'), 3)
    precision = round(precision_score(y, preds, average='weighted', zero_division=0), 3)
    recall = round(recall_score(y, preds, average='weighted', zero_division=0), 3)
    cm = confusion_matrix(y, preds).tolist()
    return f1, cm, precision, recall


# -----------------------
# Dashboard metrics summary
# -----------------------
def get_metrics_summary(model_name, model, accuracies, f1, cm, precision=None, recall=None):
    """Return summary dictionary for dynamic dashboard display.

    precision and recall are optional (fractions 0-1). If provided, include them.
    """
    summary = {
        "model": model_name,
        "accuracy": accuracies.get(model_name, "N/A"),
        "f1_score": f1,
        "confusion_matrix": cm
    }
    # include precision/recall if available
    if precision is not None:
        summary["precision"] = precision
    if recall is not None:
        summary["recall"] = recall
    return summary
