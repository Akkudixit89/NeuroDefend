import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import xgboost as xgb
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# -----------------------
# Tokenizer for URLs
# -----------------------
def makeTokens(f):
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
    urls_data = pd.read_csv("url_features.csv")
    y = urls_data["malicious"]
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
# Predict single URL
# -----------------------
def predict_url(url, model, vectorizer):
    X = vectorizer.transform([url])
    pred = model.predict(X)[0]
    return "Malicious" if pred == 1 else "Benign"


# -----------------------
# Evaluate model (F1 + Confusion Matrix)
# -----------------------
def evaluate_model(model, vectorizer, urls_data):
    y = urls_data["malicious"]
    url_list = urls_data["URL"]
    X = vectorizer.transform(url_list)

    preds = model.predict(X)
    f1 = round(f1_score(y, preds), 3)
    cm = confusion_matrix(y, preds).tolist()
    return f1, cm
